#!/bin/bash
# AWS EC2 Setup for Simplex Pilot Training
# Creates infrastructure for GPU-accelerated pilot runs

set -e

REGION="us-east-1"
INSTANCE_TYPE="g4dn.xlarge"
KEY_NAME="simplex-pilot-key"
SECURITY_GROUP_NAME="simplex-pilot-sg"
INSTANCE_NAME="simplex-pilot"
VPC_NAME="simplex-pilot-vpc"

echo "=== Simplex Pilot AWS Setup ==="
echo "Region: $REGION"
echo "Instance: $INSTANCE_TYPE (NVIDIA T4 GPU)"
echo "Estimated cost: ~\$0.19/hour (spot)"
echo ""

# Check AWS CLI
if ! aws sts get-caller-identity &>/dev/null; then
    echo "ERROR: AWS CLI not configured. Run 'aws configure' first."
    exit 1
fi

# Create key pair if not exists
if ! aws ec2 describe-key-pairs --key-names $KEY_NAME --region $REGION &>/dev/null; then
    echo "Creating SSH key pair..."
    aws ec2 create-key-pair --key-name $KEY_NAME --region $REGION \
        --query 'KeyMaterial' --output text > ~/.ssh/${KEY_NAME}.pem
    chmod 400 ~/.ssh/${KEY_NAME}.pem
    echo "Key saved to ~/.ssh/${KEY_NAME}.pem"
else
    echo "Key pair $KEY_NAME already exists"
fi

# Check for existing VPC
VPC_ID=$(aws ec2 describe-vpcs --region $REGION \
    --filters "Name=tag:Name,Values=$VPC_NAME" \
    --query 'Vpcs[0].VpcId' --output text 2>/dev/null || echo "None")

if [ "$VPC_ID" = "None" ] || [ -z "$VPC_ID" ]; then
    echo "Creating VPC..."
    VPC_ID=$(aws ec2 create-vpc \
        --cidr-block 10.0.0.0/16 \
        --region $REGION \
        --query 'Vpc.VpcId' --output text)

    aws ec2 create-tags --resources $VPC_ID --region $REGION \
        --tags Key=Name,Value=$VPC_NAME

    # Enable DNS hostnames
    aws ec2 modify-vpc-attribute --vpc-id $VPC_ID --region $REGION \
        --enable-dns-hostnames '{"Value":true}'

    echo "VPC created: $VPC_ID"
else
    echo "VPC exists: $VPC_ID"
fi

# Check for existing subnet
SUBNET_ID=$(aws ec2 describe-subnets --region $REGION \
    --filters "Name=vpc-id,Values=$VPC_ID" "Name=tag:Name,Values=${VPC_NAME}-subnet" \
    --query 'Subnets[0].SubnetId' --output text 2>/dev/null || echo "None")

if [ "$SUBNET_ID" = "None" ] || [ -z "$SUBNET_ID" ]; then
    echo "Creating subnet..."
    SUBNET_ID=$(aws ec2 create-subnet \
        --vpc-id $VPC_ID \
        --cidr-block 10.0.1.0/24 \
        --availability-zone ${REGION}a \
        --region $REGION \
        --query 'Subnet.SubnetId' --output text)

    aws ec2 create-tags --resources $SUBNET_ID --region $REGION \
        --tags Key=Name,Value=${VPC_NAME}-subnet

    # Auto-assign public IPs
    aws ec2 modify-subnet-attribute --subnet-id $SUBNET_ID --region $REGION \
        --map-public-ip-on-launch

    echo "Subnet created: $SUBNET_ID"
else
    echo "Subnet exists: $SUBNET_ID"
fi

# Check for internet gateway
IGW_ID=$(aws ec2 describe-internet-gateways --region $REGION \
    --filters "Name=attachment.vpc-id,Values=$VPC_ID" \
    --query 'InternetGateways[0].InternetGatewayId' --output text 2>/dev/null || echo "None")

if [ "$IGW_ID" = "None" ] || [ -z "$IGW_ID" ]; then
    echo "Creating internet gateway..."
    IGW_ID=$(aws ec2 create-internet-gateway \
        --region $REGION \
        --query 'InternetGateway.InternetGatewayId' --output text)

    aws ec2 attach-internet-gateway --internet-gateway-id $IGW_ID \
        --vpc-id $VPC_ID --region $REGION

    aws ec2 create-tags --resources $IGW_ID --region $REGION \
        --tags Key=Name,Value=${VPC_NAME}-igw

    echo "Internet gateway created: $IGW_ID"
else
    echo "Internet gateway exists: $IGW_ID"
fi

# Setup route table
RTB_ID=$(aws ec2 describe-route-tables --region $REGION \
    --filters "Name=vpc-id,Values=$VPC_ID" "Name=association.main,Values=true" \
    --query 'RouteTables[0].RouteTableId' --output text)

# Add route to internet gateway
aws ec2 create-route --route-table-id $RTB_ID --region $REGION \
    --destination-cidr-block 0.0.0.0/0 \
    --gateway-id $IGW_ID 2>/dev/null || true

echo "Route table configured: $RTB_ID"

# Create security group in VPC
SG_ID=$(aws ec2 describe-security-groups --region $REGION \
    --filters "Name=vpc-id,Values=$VPC_ID" "Name=group-name,Values=$SECURITY_GROUP_NAME" \
    --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || echo "None")

if [ "$SG_ID" = "None" ] || [ -z "$SG_ID" ]; then
    echo "Creating security group..."
    SG_ID=$(aws ec2 create-security-group \
        --group-name $SECURITY_GROUP_NAME \
        --description "Simplex pilot training security group" \
        --vpc-id $VPC_ID \
        --region $REGION \
        --query 'GroupId' --output text)

    # Allow SSH
    aws ec2 authorize-security-group-ingress \
        --group-id $SG_ID \
        --protocol tcp \
        --port 22 \
        --cidr 0.0.0.0/0 \
        --region $REGION

    echo "Security group created: $SG_ID"
else
    echo "Security group exists: $SG_ID"
fi

# Get latest Deep Learning AMI
AMI_ID=$(aws ec2 describe-images \
    --region $REGION \
    --owners amazon \
    --filters "Name=name,Values=Deep Learning AMI GPU PyTorch*Ubuntu 22.04*" \
    --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
    --output text)

if [ "$AMI_ID" = "None" ] || [ -z "$AMI_ID" ]; then
    # Fallback to Ubuntu 22.04 with CUDA
    AMI_ID=$(aws ec2 describe-images \
        --region $REGION \
        --owners amazon \
        --filters "Name=name,Values=*ubuntu*22.04*amd64*server*" \
        --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
        --output text)
fi

echo "Using AMI: $AMI_ID"

# Save configuration
cat > /Users/rod/code/simplex-lang/training/aws/config.env << EOF
REGION=$REGION
INSTANCE_TYPE=$INSTANCE_TYPE
KEY_NAME=$KEY_NAME
SECURITY_GROUP_ID=$SG_ID
SUBNET_ID=$SUBNET_ID
VPC_ID=$VPC_ID
AMI_ID=$AMI_ID
INSTANCE_NAME=$INSTANCE_NAME
EOF

echo ""
echo "=== Setup Complete ==="
echo "VPC: $VPC_ID"
echo "Subnet: $SUBNET_ID"
echo "Security Group: $SG_ID"
echo "AMI: $AMI_ID"
echo ""
echo "Configuration saved to aws/config.env"
echo ""
echo "Next steps:"
echo "  ./start-pilot.sh sentiment   # Launch instance and run sentiment pilot"
echo "  ./stop-pilot.sh              # Stop instance (saves money)"
echo "  ./connect-pilot.sh           # SSH into running instance"
