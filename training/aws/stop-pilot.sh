#!/bin/bash
# Stop/terminate the pilot EC2 instance to save costs

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.env" 2>/dev/null || {
    echo "No config found. Run ./setup-pilot.sh first"
    exit 1
}

ACTION="${1:-terminate}"  # terminate or stop

# Find running instance
INSTANCE_ID=$(aws ec2 describe-instances \
    --region $REGION \
    --filters "Name=tag:Name,Values=$INSTANCE_NAME" "Name=instance-state-name,Values=running,pending,stopping,stopped" \
    --query 'Reservations[0].Instances[0].InstanceId' \
    --output text 2>/dev/null || echo "None")

if [ "$INSTANCE_ID" = "None" ] || [ -z "$INSTANCE_ID" ]; then
    echo "No running instance found"
    exit 0
fi

echo "Found instance: $INSTANCE_ID"

if [ "$ACTION" = "stop" ]; then
    echo "Stopping instance (can be restarted)..."
    aws ec2 stop-instances --instance-ids $INSTANCE_ID --region $REGION
    echo "Instance stopped. Use start-pilot.sh to restart."
else
    echo "Terminating instance (permanent)..."
    aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $REGION

    # Cancel any spot requests
    SPOT_REQUESTS=$(aws ec2 describe-spot-instance-requests \
        --region $REGION \
        --filters "Name=instance-id,Values=$INSTANCE_ID" \
        --query 'SpotInstanceRequests[*].SpotInstanceRequestId' \
        --output text 2>/dev/null || echo "")

    if [ -n "$SPOT_REQUESTS" ]; then
        aws ec2 cancel-spot-instance-requests \
            --spot-instance-request-ids $SPOT_REQUESTS \
            --region $REGION 2>/dev/null || true
    fi

    echo "Instance terminated. No more charges."
    rm -f "$SCRIPT_DIR/instance.env"
fi
