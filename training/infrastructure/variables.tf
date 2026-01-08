# Simplex Cognitive Training - Terraform Variables
#
# SECURITY: Sensitive values should be passed via:
#   - Environment variables: TF_VAR_xxx
#   - terraform.tfvars (git-ignored)
#   - Command line: -var="xxx=yyy"
#
# NEVER commit sensitive values to version control.

# AWS Configuration
variable "aws_region" {
  description = "AWS region for training resources"
  type        = string
  default     = "us-east-1"
}

variable "aws_profile" {
  description = "AWS CLI profile to use (leave empty to use environment variables)"
  type        = string
  default     = ""
}

# Instance Configuration
variable "instance_type" {
  description = "EC2 instance type for training"
  type        = string
  default     = "g5.xlarge" # 1x A10G 24GB - ~$1/hr

  validation {
    condition = contains([
      "g5.xlarge",    # 1x A10G 24GB - $1.01/hr
      "g5.2xlarge",   # 1x A10G 24GB + more CPU/RAM - $1.21/hr
      "g5.4xlarge",   # 1x A10G 24GB + more CPU/RAM - $1.62/hr
      "g5.12xlarge",  # 4x A10G 24GB - $5.67/hr
      "p3.2xlarge",   # 1x V100 16GB - $3.06/hr
      "p3.8xlarge",   # 4x V100 16GB - $12.24/hr
      "p4d.24xlarge", # 8x A100 40GB - $32.77/hr
    ], var.instance_type)
    error_message = "Must be a valid GPU instance type."
  }
}

variable "use_spot_instance" {
  description = "Use spot instance for cost savings (can be interrupted)"
  type        = bool
  default     = true
}

variable "spot_max_price" {
  description = "Maximum hourly price for spot instance"
  type        = string
  default     = "0.50" # 50% of on-demand for g5.xlarge
}

variable "root_volume_size" {
  description = "Size of root EBS volume in GB"
  type        = number
  default     = 200
}

# SSH Access
variable "ssh_public_key" {
  description = "SSH public key for instance access (leave empty to use existing key)"
  type        = string
  default     = ""
  sensitive   = true
}

variable "existing_key_name" {
  description = "Name of existing AWS key pair (if not providing ssh_public_key)"
  type        = string
  default     = ""
}

variable "allowed_ssh_cidrs" {
  description = "CIDR blocks allowed to SSH to the instance"
  type        = list(string)
  default     = ["0.0.0.0/0"] # Restrict this in production!
}

# S3 Configuration
variable "s3_bucket" {
  description = "S3 bucket for model artifacts"
  type        = string
  default     = "simplex-cognitive-training"
}

variable "create_s3_bucket" {
  description = "Create S3 bucket (set false if bucket already exists)"
  type        = bool
  default     = true
}

# Secrets (NEVER commit actual values)
variable "wandb_api_key" {
  description = "Weights & Biases API key for experiment tracking"
  type        = string
  default     = ""
  sensitive   = true
}

variable "hf_token" {
  description = "HuggingFace token for model access"
  type        = string
  default     = ""
  sensitive   = true
}
