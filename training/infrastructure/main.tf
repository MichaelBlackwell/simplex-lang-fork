# Simplex Cognitive Training Infrastructure
# Terraform configuration for AWS GPU training instance
#
# SECURITY: No credentials stored in this file.
# Use AWS CLI profile or environment variables:
#   export AWS_ACCESS_KEY_ID=xxx
#   export AWS_SECRET_ACCESS_KEY=xxx
#   export AWS_REGION=us-east-1
#
# Or use AWS CLI profile:
#   aws configure --profile simplex-training

terraform {
  required_version = ">= 1.0.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# Provider configuration - uses environment variables or AWS CLI profile
provider "aws" {
  region  = var.aws_region
  profile = var.aws_profile != "" ? var.aws_profile : null

  default_tags {
    tags = {
      Project     = "simplex-cognitive"
      Environment = "training"
      ManagedBy   = "terraform"
    }
  }
}

# Data sources
data "aws_ami" "deep_learning" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["Deep Learning OSS Nvidia Driver AMI GPU PyTorch *-Ubuntu 22.04-*"]
  }

  filter {
    name   = "architecture"
    values = ["x86_64"]
  }
}

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# Security group for training instance
resource "aws_security_group" "training" {
  name        = "simplex-training-sg"
  description = "Security group for Simplex cognitive training"
  vpc_id      = data.aws_vpc.default.id

  # SSH access
  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.allowed_ssh_cidrs
  }

  # Outbound internet access
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "simplex-training-sg"
  }
}

# IAM role for the instance
resource "aws_iam_role" "training" {
  name = "simplex-training-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })
}

# S3 access for model artifacts
resource "aws_iam_role_policy" "s3_access" {
  name = "simplex-training-s3-access"
  role = aws_iam_role.training.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket"
        ]
        Resource = [
          "arn:aws:s3:::${var.s3_bucket}",
          "arn:aws:s3:::${var.s3_bucket}/*"
        ]
      }
    ]
  })
}

resource "aws_iam_instance_profile" "training" {
  name = "simplex-training-profile"
  role = aws_iam_role.training.name
}

# SSH key pair (public key provided as variable)
resource "aws_key_pair" "training" {
  count      = var.ssh_public_key != "" ? 1 : 0
  key_name   = "simplex-training-key"
  public_key = var.ssh_public_key
}

# Spot instance request for cost savings
resource "aws_spot_instance_request" "training" {
  count = var.use_spot_instance ? 1 : 0

  ami                    = data.aws_ami.deep_learning.id
  instance_type          = var.instance_type
  key_name               = var.ssh_public_key != "" ? aws_key_pair.training[0].key_name : var.existing_key_name
  vpc_security_group_ids = [aws_security_group.training.id]
  subnet_id              = data.aws_subnets.default.ids[0]
  iam_instance_profile   = aws_iam_instance_profile.training.name

  spot_price                     = var.spot_max_price
  wait_for_fulfillment          = true
  instance_interruption_behavior = "terminate"

  root_block_device {
    volume_size           = var.root_volume_size
    volume_type           = "gp3"
    delete_on_termination = true
  }

  user_data = base64encode(templatefile("${path.module}/user_data.sh", {
    s3_bucket     = var.s3_bucket
    wandb_api_key = var.wandb_api_key
    hf_token      = var.hf_token
  }))

  tags = {
    Name = "simplex-training-spot"
  }
}

# On-demand instance (alternative to spot)
resource "aws_instance" "training" {
  count = var.use_spot_instance ? 0 : 1

  ami                    = data.aws_ami.deep_learning.id
  instance_type          = var.instance_type
  key_name               = var.ssh_public_key != "" ? aws_key_pair.training[0].key_name : var.existing_key_name
  vpc_security_group_ids = [aws_security_group.training.id]
  subnet_id              = data.aws_subnets.default.ids[0]
  iam_instance_profile   = aws_iam_instance_profile.training.name

  root_block_device {
    volume_size           = var.root_volume_size
    volume_type           = "gp3"
    delete_on_termination = true
  }

  user_data = base64encode(templatefile("${path.module}/user_data.sh", {
    s3_bucket     = var.s3_bucket
    wandb_api_key = var.wandb_api_key
    hf_token      = var.hf_token
  }))

  tags = {
    Name = "simplex-training"
  }
}

# S3 bucket for model artifacts (optional)
resource "aws_s3_bucket" "artifacts" {
  count  = var.create_s3_bucket ? 1 : 0
  bucket = var.s3_bucket

  tags = {
    Name = "simplex-training-artifacts"
  }
}

resource "aws_s3_bucket_versioning" "artifacts" {
  count  = var.create_s3_bucket ? 1 : 0
  bucket = aws_s3_bucket.artifacts[0].id

  versioning_configuration {
    status = "Enabled"
  }
}
