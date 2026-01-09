#!/bin/bash
# SSH into the pilot EC2 instance

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.env" 2>/dev/null || {
    echo "No config found. Run ./setup-pilot.sh first"
    exit 1
}

# Get instance IP
if [ -f "$SCRIPT_DIR/instance.env" ]; then
    source "$SCRIPT_DIR/instance.env"
else
    PUBLIC_IP=$(aws ec2 describe-instances \
        --region $REGION \
        --filters "Name=tag:Name,Values=$INSTANCE_NAME" "Name=instance-state-name,Values=running" \
        --query 'Reservations[0].Instances[0].PublicIpAddress' \
        --output text 2>/dev/null || echo "None")
fi

if [ "$PUBLIC_IP" = "None" ] || [ -z "$PUBLIC_IP" ]; then
    echo "No running instance found. Start one with ./start-pilot.sh"
    exit 1
fi

echo "Connecting to $PUBLIC_IP..."
ssh -i ~/.ssh/${KEY_NAME}.pem -o StrictHostKeyChecking=no ubuntu@${PUBLIC_IP}
