# Simplex Cognitive Training - Terraform Outputs

output "instance_id" {
  description = "ID of the training instance"
  value       = var.use_spot_instance ? aws_spot_instance_request.training[0].spot_instance_id : aws_instance.training[0].id
}

output "instance_public_ip" {
  description = "Public IP of the training instance"
  value       = var.use_spot_instance ? aws_spot_instance_request.training[0].public_ip : aws_instance.training[0].public_ip
}

output "instance_public_dns" {
  description = "Public DNS of the training instance"
  value       = var.use_spot_instance ? aws_spot_instance_request.training[0].public_dns : aws_instance.training[0].public_dns
}

output "ssh_command" {
  description = "SSH command to connect to the instance"
  value       = "ssh -i <your-key.pem> ubuntu@${var.use_spot_instance ? aws_spot_instance_request.training[0].public_ip : aws_instance.training[0].public_ip}"
}

output "s3_bucket" {
  description = "S3 bucket for artifacts"
  value       = var.s3_bucket
}

output "upload_command" {
  description = "Command to upload training code to instance"
  value       = "scp -i <your-key.pem> -r ../scripts ../configs ../requirements.txt ubuntu@${var.use_spot_instance ? aws_spot_instance_request.training[0].public_ip : aws_instance.training[0].public_ip}:~/simplex-training/"
}

output "estimated_hourly_cost" {
  description = "Estimated hourly cost"
  value = var.use_spot_instance ? "~$${var.spot_max_price}/hr (spot)" : lookup({
    "g5.xlarge"    = "$1.01/hr"
    "g5.2xlarge"   = "$1.21/hr"
    "g5.4xlarge"   = "$1.62/hr"
    "g5.12xlarge"  = "$5.67/hr"
    "p3.2xlarge"   = "$3.06/hr"
    "p3.8xlarge"   = "$12.24/hr"
    "p4d.24xlarge" = "$32.77/hr"
  }, var.instance_type, "unknown")
}
