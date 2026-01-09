#!/bin/bash
# Run pilot on remote EC2 instance and stream output

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.env" 2>/dev/null || {
    echo "No config found. Run ./setup-pilot.sh first"
    exit 1
}

SPECIALIST="${1:-sentiment}"
MODE="${2:---baseline-only}"

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

echo "=== Running $SPECIALIST Pilot on AWS ==="
echo "Instance: $PUBLIC_IP"
echo "Mode: $MODE"
echo ""

# Run the pilot and stream output
ssh -i ~/.ssh/${KEY_NAME}.pem -o StrictHostKeyChecking=no ubuntu@${PUBLIC_IP} \
    "cd /home/ubuntu/training/validation && \
     source /opt/conda/etc/profile.d/conda.sh && \
     conda activate pytorch && \
     pip install datasets peft trl faker sqlparse accelerate transformers --quiet && \
     python pilots/train_pilot.py --specialist $SPECIALIST $MODE"

echo ""
echo "=== Pilot Complete ==="
echo "Don't forget to stop the instance: ./stop-pilot.sh"
