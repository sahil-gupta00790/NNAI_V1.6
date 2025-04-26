output "vpc_id" {
  description = "The ID of the created VPC."
  value       = aws_vpc.main.id
}

output "public_subnet_ids" {
  description = "List of IDs for the public subnets."
  value       = aws_subnet.public[*].id
}

output "private_subnet_ids" {
  description = "List of IDs for the private subnets."
  value       = aws_subnet.private[*].id
}

output "ecs_cluster_name" {
  description = "The name of the ECS cluster."
  value       = aws_ecs_cluster.main.name
}

output "ecs_task_execution_role_arn" {
  description = "ARN of the ECS Task Execution Role."
  value       = aws_iam_role.ecs_task_execution_role.arn
}

output "ecs_task_security_group_id" {
  description = "The ID of the security group for ECS tasks."
  value       = aws_security_group.ecs_tasks.id
}

output "ecs_service_name" {
  description = "The name of the example ECS service created."
  value       = aws_ecs_service.app.name # Only if creating the example service
}
