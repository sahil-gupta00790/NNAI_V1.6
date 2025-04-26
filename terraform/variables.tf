variable "aws_region" {
  description = "The AWS region to deploy resources in."
  type        = string
  default     = "us-east-1" # Choose your desired region
}

variable "project_name" {
  description = "A name prefix for resources to ensure uniqueness and grouping."
  type        = string
  default     = "my-ecs-app"
}

variable "vpc_cidr" {
  description = "The CIDR block for the VPC."
  type        = string
  default     = "10.0.0.0/16"
}

variable "public_subnet_cidrs" {
  description = "List of CIDR blocks for public subnets (one per AZ)."
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24"]
}

variable "private_subnet_cidrs" {
  description = "List of CIDR blocks for private subnets (one per AZ)."
  type        = list(string)
  default     = ["10.0.101.0/24", "10.0.102.0/24"]
}

variable "availability_zones" {
  description = "List of Availability Zones to use (must match the number of subnets)."
  type        = list(string)
  # Let Terraform determine AZs dynamically based on the chosen region
  default = []
}

variable "ecs_cluster_name" {
  description = "Name for the ECS Cluster."
  type        = string
  default     = "main-cluster"
}

# --- Example Task Definition Variables (Modify for your app) ---
variable "ecs_task_family" {
  description = "Family name for the ECS Task Definition."
  type        = string
  default     = "nginx-example-task"
}

variable "ecs_service_name" {
  description = "Name for the ECS Service."
  type        = string
  default     = "nginx-example-service"
}

variable "container_image" {
  description = "Docker image to run in the container."
  type        = string
  default     = "nginx:latest" # Replace with your image from ECR later
}

variable "container_port" {
  description = "Port the container listens on."
  type        = number
  default     = 3000
}

variable "desired_task_count" {
  description = "Number of tasks to run for the service."
  type        = number
  default     = 1
}

variable "task_cpu" {
  description = "Fargate task CPU units (e.g., 256 = 0.25 vCPU)."
  type        = number
  default     = 256
}

variable "task_memory" {
  description = "Fargate task memory MiB (e.g., 512 = 0.5 GB)."
  type        = number
  default     = 512
}
