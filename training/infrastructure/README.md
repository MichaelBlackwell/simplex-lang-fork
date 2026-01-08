# Simplex Training Infrastructure

AWS infrastructure for GPU training using Terraform or AWS CLI.

## Security

**IMPORTANT**: No credentials are stored in any files. Use one of these methods:

### Option 1: Environment Variables
```bash
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_REGION="us-east-1"
```

### Option 2: AWS CLI Profile
```bash
aws configure --profile simplex-training
export AWS_PROFILE=simplex-training
```

### Option 3: IAM Role (EC2/ECS)
If running from AWS, use IAM roles attached to the compute resource.

## Quick Start with AWS CLI

For simple one-off training, use the CLI script:

```bash
# Launch instance
./launch_instance.sh

# When done
./terminate_instance.sh
```

## Terraform Deployment

For repeatable infrastructure:

```bash
cd infrastructure

# Initialize
terraform init

# Preview changes
terraform plan -var="ssh_public_key=$(cat ~/.ssh/id_rsa.pub)"

# Deploy
terraform apply -var="ssh_public_key=$(cat ~/.ssh/id_rsa.pub)"

# Get connection info
terraform output

# Destroy when done
terraform destroy
```

## Instance Types

| Type | GPU | VRAM | $/hr (On-Demand) | $/hr (Spot) | Use Case |
|------|-----|------|------------------|-------------|----------|
| g5.xlarge | 1x A10G | 24GB | $1.01 | ~$0.35 | 8B model training |
| g5.2xlarge | 1x A10G | 24GB | $1.21 | ~$0.42 | 8B + more CPU |
| g5.12xlarge | 4x A10G | 96GB | $5.67 | ~$2.00 | 32B model training |
| p3.2xlarge | 1x V100 | 16GB | $3.06 | ~$1.00 | Alternative to g5 |
| p4d.24xlarge | 8x A100 | 320GB | $32.77 | ~$12.00 | Large scale |

## Cost Estimation

For 8B model full training pipeline:
- Stage 1-3: ~12-16 hours
- Spot pricing: ~$5-8 total
- On-demand: ~$12-16 total

## Files

```
infrastructure/
├── main.tf           # Main Terraform configuration
├── variables.tf      # Input variables (no secrets!)
├── outputs.tf        # Output values
├── user_data.sh      # Instance bootstrap script
├── launch_instance.sh # Simple AWS CLI launcher
├── terminate_instance.sh
└── README.md
```
