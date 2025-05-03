# --- Networking Outputs ---
output "vpc_id" {
  description = "The ID of the created VPC."
  value       = aws_vpc.main.id
}

output "public_subnet_ids" {
  description = "The IDs of the public subnets."
  value       = aws_subnet.public[*].id
}

output "private_subnet_ids" {
  description = "The IDs of the private subnets."
  value       = aws_subnet.private[*].id
}

# --- ECS Outputs ---
output "ecs_cluster_name" {
  description = "The name of the ECS cluster created."
  value       = aws_ecs_cluster.main.name
}

output "ecs_task_execution_role_arn" {
  description = "The ARN of the ECS Task Execution Role."
  value       = aws_iam_role.ecs_task_execution_role.arn
}

output "ecs_task_security_group_id" {
  description = "The ID of the Security Group attached to ECS tasks."
  value       = aws_security_group.ecs_tasks.id
}

output "ecs_service_name" {
  description = "The name of the ECS service created."
  value       = aws_ecs_service.app.name
}

# --- Load Balancer Output ---
output "alb_dns_name" {
  description = "The DNS name of the Application Load Balancer to access the frontend."
  value       = aws_lb.main.dns_name
}

output "alb_security_group_id" {
  description = "The ID of the Security Group attached to the ALB."
  value       = aws_security_group.alb.id
}

output "cloudwatch_log_group_name" {
  description = "The name of the CloudWatch Log Group for ECS container logs."
  value       = aws_cloudwatch_log_group.ecs_logs.name
}
