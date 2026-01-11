#!/bin/bash
# SAFE training script - syncs to S3 after each specialist
# DOES NOT auto-shutdown until all models are confirmed in S3

set -e

S3_BUCKET="s3://simplex-model-repo"
REGION="us-east-2"

# Verify AWS credentials BEFORE starting
echo "Verifying AWS credentials..."
if ! aws sts get-caller-identity --region $REGION > /dev/null 2>&1; then
    echo "ERROR: AWS credentials not configured!"
    echo "Please run: aws configure"
    echo "Or attach an instance profile with S3 access"
    exit 1
fi
echo "AWS credentials verified!"

source /opt/pytorch/bin/activate 2>/dev/null || source ~/pytorch/bin/activate 2>/dev/null || true
cd /home/ec2-user/training

# List of specialists to train
SPECIALISTS=(
    "code_documentation"
    "compliance_check"
    "email_writing"
    "fact_checking"
    "faq_answering"
    "financial_analysis"
    "intent_classification"
    "keyword_extraction"
    "legal_analysis"
    "logical_reasoning"
    "math_reasoning"
    "meeting_summarization"
    "negotiation"
    "news_summarization"
    "relation_extraction"
    "report_generation"
    "risk_assessment"
    "sentiment_analysis"
    "spam_detection"
    "support_response"
    "task_dialogue"
    "ticket_classification"
    "topic_classification"
    "web_developer"
)

BASE_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
SYNCED_COUNT=0
TOTAL=${#SPECIALISTS[@]}

for i in "${!SPECIALISTS[@]}"; do
    SPECIALIST="${SPECIALISTS[$i]}"
    NUM=$((i + 1))

    echo "========================================"
    echo "[$NUM/$TOTAL] Training: $SPECIALIST"
    echo "Started: $(date)"
    echo "========================================"

    # Check if already trained and synced to S3
    if aws s3 ls "${S3_BUCKET}/specialists/${SPECIALIST}/adapter_model.safetensors" --region $REGION 2>/dev/null; then
        echo "$SPECIALIST already in S3, skipping..."
        SYNCED_COUNT=$((SYNCED_COUNT + 1))
        continue
    fi

    # Check if we were interrupted
    if [ -f /home/ec2-user/training/interrupted.flag ]; then
        echo "Spot interruption detected, stopping training..."
        exit 1
    fi

    # Train
    python train_specialist.py \
        --specialist "$SPECIALIST" \
        --model "$BASE_MODEL" \
        --data "data/specialists/${SPECIALIST}_training.jsonl" \
        --output-dir "outputs/${SPECIALIST}" \
        --epochs 1 \
        --batch-size 1 \
        --learning-rate 2e-4 \
        --max-length 512 \
        --save-steps 10 2>&1 | tee "outputs/${SPECIALIST}.log"

    # IMMEDIATELY sync to S3 after training
    echo "Syncing $SPECIALIST to S3..."
    if aws s3 sync "outputs/${SPECIALIST}" "${S3_BUCKET}/specialists/${SPECIALIST}/" --region $REGION; then
        echo "$SPECIALIST synced successfully!"
        SYNCED_COUNT=$((SYNCED_COUNT + 1))

        # Verify sync
        if aws s3 ls "${S3_BUCKET}/specialists/${SPECIALIST}/adapter_model.safetensors" --region $REGION 2>/dev/null; then
            echo "Verified: $SPECIALIST is in S3"
        else
            echo "WARNING: Sync may have failed for $SPECIALIST"
        fi
    else
        echo "ERROR: Failed to sync $SPECIALIST to S3!"
        echo "Stopping to prevent data loss."
        exit 1
    fi

    echo ""
done

echo "========================================"
echo "Training complete!"
echo "Synced: $SYNCED_COUNT / $TOTAL specialists"
echo "========================================"

# Verify all models are in S3 before allowing shutdown
echo ""
echo "Verifying all models in S3..."
VERIFIED=0
for SPECIALIST in "${SPECIALISTS[@]}"; do
    if aws s3 ls "${S3_BUCKET}/specialists/${SPECIALIST}/adapter_model.safetensors" --region $REGION 2>/dev/null; then
        VERIFIED=$((VERIFIED + 1))
    else
        echo "MISSING: $SPECIALIST not in S3!"
    fi
done

echo "Verified: $VERIFIED / $TOTAL in S3"

if [ $VERIFIED -eq $TOTAL ]; then
    echo ""
    echo "All models safely in S3!"
    echo "You can now safely stop or terminate this instance."
    echo ""
    echo "To stop: sudo shutdown -h now"
else
    echo ""
    echo "WARNING: Not all models are in S3!"
    echo "DO NOT terminate this instance until resolved."
fi
