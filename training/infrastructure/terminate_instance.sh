#!/bin/bash
# Terminate the training instance

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

if [ ! -f .instance_id ]; then
    echo -e "${RED}Error: No instance ID found${NC}"
    echo "Looking for running simplex-training instances..."

    INSTANCE_ID=$(aws ec2 describe-instances \
        --filters "Name=tag:Name,Values=simplex-training" "Name=instance-state-name,Values=running" \
        --query 'Reservations[0].Instances[0].InstanceId' \
        --output text)

    if [ -z "$INSTANCE_ID" ] || [ "$INSTANCE_ID" == "None" ]; then
        echo "No running instances found."
        exit 1
    fi
else
    INSTANCE_ID=$(cat .instance_id)
fi

echo -e "Instance ID: ${YELLOW}${INSTANCE_ID}${NC}"
read -p "Are you sure you want to terminate this instance? (y/N) " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Terminating instance..."
    aws ec2 terminate-instances --instance-ids "$INSTANCE_ID"

    echo "Waiting for termination..."
    aws ec2 wait instance-terminated --instance-ids "$INSTANCE_ID"

    rm -f .instance_id

    echo -e "${GREEN}Instance terminated successfully${NC}"
else
    echo "Cancelled."
fi
