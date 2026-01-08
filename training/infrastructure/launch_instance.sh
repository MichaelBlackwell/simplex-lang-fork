#!/bin/bash
# Simple AWS CLI script to launch training instance
# No credentials stored - uses AWS CLI configuration

set -e

# Configuration
INSTANCE_TYPE="${INSTANCE_TYPE:-g5.xlarge}"
REGION="${AWS_REGION:-us-east-1}"
KEY_NAME="${KEY_NAME:-}"  # Your existing EC2 key pair name
VOLUME_SIZE="${VOLUME_SIZE:-200}"
USE_SPOT="${USE_SPOT:-true}"
SPOT_PRICE="${SPOT_PRICE:-0.50}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}Simplex Cognitive Training - Instance Launch${NC}"
echo -e "${GREEN}============================================${NC}"

# Check AWS CLI
if ! command -v aws &> /dev/null; then
    echo -e "${RED}Error: AWS CLI not installed${NC}"
    exit 1
fi

# Verify AWS credentials
echo "Verifying AWS credentials..."
if ! aws sts get-caller-identity &> /dev/null; then
    echo -e "${RED}Error: AWS credentials not configured${NC}"
    echo "Run: aws configure"
    exit 1
fi

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo -e "Using AWS Account: ${GREEN}${ACCOUNT_ID}${NC}"

# Check for key pair
if [ -z "$KEY_NAME" ]; then
    echo -e "${YELLOW}Warning: KEY_NAME not set${NC}"
    echo "Available key pairs:"
    aws ec2 describe-key-pairs --query 'KeyPairs[*].KeyName' --output table
    read -p "Enter key pair name: " KEY_NAME
fi

# Find Deep Learning AMI
echo "Finding Deep Learning AMI..."
AMI_ID=$(aws ec2 describe-images \
    --owners amazon \
    --filters "Name=name,Values=Deep Learning OSS Nvidia Driver AMI GPU PyTorch *-Ubuntu 22.04-*" \
    --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
    --output text)

if [ "$AMI_ID" == "None" ] || [ -z "$AMI_ID" ]; then
    echo -e "${RED}Error: Could not find Deep Learning AMI${NC}"
    exit 1
fi
echo -e "Using AMI: ${GREEN}${AMI_ID}${NC}"

# Get default VPC and subnet
VPC_ID=$(aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" --query 'Vpcs[0].VpcId' --output text)
SUBNET_ID=$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=${VPC_ID}" --query 'Subnets[0].SubnetId' --output text)

# Create security group if needed
SG_NAME="simplex-training-sg"
SG_ID=$(aws ec2 describe-security-groups --filters "Name=group-name,Values=${SG_NAME}" --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || echo "")

if [ -z "$SG_ID" ] || [ "$SG_ID" == "None" ]; then
    echo "Creating security group..."
    SG_ID=$(aws ec2 create-security-group \
        --group-name "$SG_NAME" \
        --description "Simplex training instance" \
        --vpc-id "$VPC_ID" \
        --query 'GroupId' \
        --output text)

    aws ec2 authorize-security-group-ingress \
        --group-id "$SG_ID" \
        --protocol tcp \
        --port 22 \
        --cidr 0.0.0.0/0
fi
echo -e "Security Group: ${GREEN}${SG_ID}${NC}"

# User data script
USER_DATA=$(cat << 'EOF'
#!/bin/bash
exec > >(tee /var/log/simplex-setup.log) 2>&1
apt-get update && apt-get upgrade -y
apt-get install -y awscli tmux htop nvtop
mkdir -p /home/ubuntu/simplex-training
cd /home/ubuntu/simplex-training
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate datasets peft bitsandbytes trl wandb tensorboard pandas numpy jsonlines tqdm scikit-learn scipy faker pyyaml
chown -R ubuntu:ubuntu /home/ubuntu/simplex-training
echo "Setup complete!" >> /var/log/simplex-setup.log
EOF
)

USER_DATA_B64=$(echo "$USER_DATA" | base64)

# Launch instance
echo ""
echo -e "Launching ${GREEN}${INSTANCE_TYPE}${NC} instance..."

if [ "$USE_SPOT" == "true" ]; then
    echo -e "Using ${YELLOW}spot instance${NC} (max price: \$${SPOT_PRICE}/hr)"

    INSTANCE_ID=$(aws ec2 run-instances \
        --image-id "$AMI_ID" \
        --instance-type "$INSTANCE_TYPE" \
        --key-name "$KEY_NAME" \
        --security-group-ids "$SG_ID" \
        --subnet-id "$SUBNET_ID" \
        --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":${VOLUME_SIZE},\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]" \
        --user-data "$USER_DATA_B64" \
        --instance-market-options "MarketType=spot,SpotOptions={MaxPrice=${SPOT_PRICE},SpotInstanceType=one-time}" \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=simplex-training}]" \
        --query 'Instances[0].InstanceId' \
        --output text)
else
    echo "Using on-demand instance"

    INSTANCE_ID=$(aws ec2 run-instances \
        --image-id "$AMI_ID" \
        --instance-type "$INSTANCE_TYPE" \
        --key-name "$KEY_NAME" \
        --security-group-ids "$SG_ID" \
        --subnet-id "$SUBNET_ID" \
        --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":${VOLUME_SIZE},\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]" \
        --user-data "$USER_DATA_B64" \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=simplex-training}]" \
        --query 'Instances[0].InstanceId' \
        --output text)
fi

echo -e "Instance ID: ${GREEN}${INSTANCE_ID}${NC}"

# Save instance ID for termination script
echo "$INSTANCE_ID" > .instance_id

# Wait for instance to be running
echo "Waiting for instance to start..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID"

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}Instance launched successfully!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo -e "Instance ID: ${GREEN}${INSTANCE_ID}${NC}"
echo -e "Public IP:   ${GREEN}${PUBLIC_IP}${NC}"
echo ""
echo "Wait ~5 minutes for setup to complete, then:"
echo ""
echo -e "  ${YELLOW}ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@${PUBLIC_IP}${NC}"
echo ""
echo "Upload training code:"
echo ""
echo -e "  ${YELLOW}scp -i ~/.ssh/${KEY_NAME}.pem -r ../scripts ../configs ../requirements.txt ubuntu@${PUBLIC_IP}:~/simplex-training/${NC}"
echo ""
echo "To terminate:"
echo ""
echo -e "  ${YELLOW}./terminate_instance.sh${NC}"
echo ""
