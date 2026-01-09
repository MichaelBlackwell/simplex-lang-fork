#!/bin/bash
# Start AWS EC2 on-demand instance and run pilot training
# Use when spot instances aren't available

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.env" 2>/dev/null || {
    echo "Run ./setup-pilot.sh first"
    exit 1
}

SPECIALIST="${1:-sentiment}"
MODE="${2:---baseline-only}"

echo "=== Starting Simplex Pilot (On-Demand) ==="
echo "Specialist: $SPECIALIST"
echo "Mode: $MODE"
echo "Instance: $INSTANCE_TYPE @ $REGION"
echo "Cost: ~\$0.52/hour (on-demand g4dn.xlarge)"
echo ""

# Check for existing instance
EXISTING=$(aws ec2 describe-instances \
    --region $REGION \
    --filters "Name=tag:Name,Values=$INSTANCE_NAME" "Name=instance-state-name,Values=running,pending" \
    --query 'Reservations[0].Instances[0].InstanceId' \
    --output text 2>/dev/null || echo "None")

if [ "$EXISTING" != "None" ] && [ -n "$EXISTING" ]; then
    echo "Instance already running: $EXISTING"
    PUBLIC_IP=$(aws ec2 describe-instances \
        --region $REGION \
        --instance-ids $EXISTING \
        --query 'Reservations[0].Instances[0].PublicIpAddress' \
        --output text)
    echo "Public IP: $PUBLIC_IP"
    echo ""
    echo "Use ./connect-pilot.sh to connect or ./stop-pilot.sh to stop"
    exit 0
fi

# Create user data script
USER_DATA=$(cat << 'USERDATA'
#!/bin/bash
set -ex

# Log everything
exec > >(tee /var/log/pilot-setup.log) 2>&1

echo "=== Pilot Setup Starting ==="
cd /home/ubuntu

mkdir -p training

cat > /home/ubuntu/run-pilot.sh << 'RUNSCRIPT'
#!/bin/bash
set -e
cd /home/ubuntu/training

# Activate conda environment with PyTorch
source /opt/conda/etc/profile.d/conda.sh
conda activate pytorch

# Install additional dependencies
pip install datasets peft trl faker sqlparse accelerate transformers --quiet

# Run the pilot
SPECIALIST="${1:-sentiment}"
MODE="${2:---baseline-only}"

echo "Running $SPECIALIST pilot with mode: $MODE"
cd validation
python pilots/train_pilot.py --specialist $SPECIALIST $MODE

echo "=== Pilot Complete ==="
RUNSCRIPT
chmod +x /home/ubuntu/run-pilot.sh

echo "=== Setup Complete - Ready for pilot ==="
USERDATA
)

# Launch on-demand instance
echo "Launching on-demand instance..."
INSTANCE_ID=$(aws ec2 run-instances \
    --region $REGION \
    --image-id $AMI_ID \
    --instance-type $INSTANCE_TYPE \
    --key-name $KEY_NAME \
    --subnet-id $SUBNET_ID \
    --security-group-ids $SECURITY_GROUP_ID \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
    --user-data "$USER_DATA" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$INSTANCE_NAME}]" \
    --query 'Instances[0].InstanceId' \
    --output text)

echo "Instance launched: $INSTANCE_ID"

# Wait for instance to be running
echo "Waiting for instance to be running..."
aws ec2 wait instance-running --instance-ids $INSTANCE_ID --region $REGION

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
    --region $REGION \
    --instance-ids $INSTANCE_ID \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo ""
echo "=== Instance Ready ==="
echo "Instance ID: $INSTANCE_ID"
echo "Public IP: $PUBLIC_IP"
echo ""

# Save instance info
echo "INSTANCE_ID=$INSTANCE_ID" > "$SCRIPT_DIR/instance.env"
echo "PUBLIC_IP=$PUBLIC_IP" >> "$SCRIPT_DIR/instance.env"

echo "Waiting 90s for instance to initialize..."
sleep 90

# Sync training code
echo "Syncing training code to instance..."
rsync -avz --progress \
    -e "ssh -i ~/.ssh/${KEY_NAME}.pem -o StrictHostKeyChecking=no -o ConnectTimeout=30" \
    /Users/rod/code/simplex-lang/training/ \
    ubuntu@${PUBLIC_IP}:/home/ubuntu/training/

echo ""
echo "=== Ready to Run Pilot ==="
echo ""
echo "Run pilot now:"
echo "  ./run-remote-pilot.sh $SPECIALIST $MODE"
echo ""
echo "Or connect manually:"
echo "  ./connect-pilot.sh"
echo ""
echo "IMPORTANT: Stop instance when done to save costs:"
echo "  ./stop-pilot.sh"
