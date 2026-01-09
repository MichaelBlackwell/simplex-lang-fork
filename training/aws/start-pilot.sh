#!/bin/bash
# Start AWS EC2 spot instance and run pilot training

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.env" 2>/dev/null || {
    echo "Run ./setup-pilot.sh first"
    exit 1
}

SPECIALIST="${1:-sentiment}"
MODE="${2:---baseline-only}"

echo "=== Starting Simplex Pilot ==="
echo "Specialist: $SPECIALIST"
echo "Mode: $MODE"
echo "Instance: $INSTANCE_TYPE @ $REGION"
echo ""

# Check for existing instance
EXISTING=$(aws ec2 describe-instances \
    --region $REGION \
    --filters "Name=tag:Name,Values=$INSTANCE_NAME" "Name=instance-state-name,Values=running,pending" \
    --query 'Reservations[0].Instances[0].InstanceId' \
    --output text 2>/dev/null || echo "None")

if [ "$EXISTING" != "None" ] && [ -n "$EXISTING" ]; then
    echo "Instance already running: $EXISTING"
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

# Clone or update training code
if [ ! -d "training" ]; then
    mkdir -p training
fi

# Create the validation pipeline (will be synced via S3 or uploaded)
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

# Request spot instance
echo "Requesting spot instance..."
SPOT_REQUEST=$(aws ec2 request-spot-instances \
    --region $REGION \
    --instance-count 1 \
    --type "one-time" \
    --launch-specification "{
        \"ImageId\": \"$AMI_ID\",
        \"InstanceType\": \"$INSTANCE_TYPE\",
        \"KeyName\": \"$KEY_NAME\",
        \"SubnetId\": \"$SUBNET_ID\",
        \"SecurityGroupIds\": [\"$SECURITY_GROUP_ID\"],
        \"BlockDeviceMappings\": [{
            \"DeviceName\": \"/dev/sda1\",
            \"Ebs\": {
                \"VolumeSize\": 100,
                \"VolumeType\": \"gp3\",
                \"DeleteOnTermination\": true
            }
        }],
        \"UserData\": \"$(echo "$USER_DATA" | base64)\"
    }" \
    --query 'SpotInstanceRequests[0].SpotInstanceRequestId' \
    --output text)

echo "Spot request: $SPOT_REQUEST"
echo "Waiting for instance to launch..."

# Wait for spot request to be fulfilled
sleep 10
for i in {1..30}; do
    INSTANCE_ID=$(aws ec2 describe-spot-instance-requests \
        --region $REGION \
        --spot-instance-request-ids $SPOT_REQUEST \
        --query 'SpotInstanceRequests[0].InstanceId' \
        --output text 2>/dev/null || echo "None")

    if [ "$INSTANCE_ID" != "None" ] && [ -n "$INSTANCE_ID" ]; then
        break
    fi
    echo "  Waiting for spot fulfillment... ($i/30)"
    sleep 5
done

if [ "$INSTANCE_ID" = "None" ] || [ -z "$INSTANCE_ID" ]; then
    echo "ERROR: Spot request not fulfilled. Check AWS console."
    exit 1
fi

echo "Instance launched: $INSTANCE_ID"

# Tag the instance
aws ec2 create-tags \
    --region $REGION \
    --resources $INSTANCE_ID \
    --tags Key=Name,Value=$INSTANCE_NAME

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

echo "Waiting 60s for instance to initialize..."
sleep 60

# Sync training code
echo "Syncing training code to instance..."
rsync -avz --progress \
    -e "ssh -i ~/.ssh/${KEY_NAME}.pem -o StrictHostKeyChecking=no" \
    /Users/rod/code/simplex-lang/training/ \
    ubuntu@${PUBLIC_IP}:/home/ubuntu/training/

echo ""
echo "=== Ready to Run Pilot ==="
echo ""
echo "Connect and run manually:"
echo "  ./connect-pilot.sh"
echo "  ./run-pilot.sh $SPECIALIST $MODE"
echo ""
echo "Or run directly:"
echo "  ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@${PUBLIC_IP} '/home/ubuntu/run-pilot.sh $SPECIALIST $MODE'"
echo ""
echo "When done, stop the instance to save costs:"
echo "  ./stop-pilot.sh"
