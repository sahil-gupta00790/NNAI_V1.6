# --- General AWS Variables ---
variable "aws_region" {
  description = "The AWS region to deploy resources in."
  type        = string
  default     = "us-east-1" # Or your preferred region
}

variable "project_name" {
  description = "A name prefix for resources to ensure uniqueness and grouping."
  type        = string
  default     = "neural-nexus"
}

# --- Networking Variables ---
variable "vpc_cidr" {
  description = "The primary CIDR block for the VPC."
  type        = string
  default     = "10.0.0.0/16"
}

variable "availability_zones" {
  description = "List of Availability Zones to use. Leave empty to use defaults for the region."
  type        = list(string)
  default     = [] # Example: ["us-east-1a", "us-east-1b"]
}

variable "num_azs" {
  description = "Number of Availability Zones to use if 'availability_zones' is empty."
  type        = number
  default     = 2 # Use at least 2 for high availability
}

variable "public_subnet_cidrs" {
  description = "List of CIDR blocks for public subnets. Must match the number of AZs."
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24"] # Example for 2 AZs
}

variable "private_subnet_cidrs" {
  description = "List of CIDR blocks for private subnets. Must match the number of AZs."
  type        = list(string)
  default     = ["10.0.101.0/24", "10.0.102.0/24"] # Example for 2 AZs
}

# --- ECS Task Definition Variables ---
variable "ecs_task_family" {
  description = "Family name for the ECS Task Definition."
  type        = string
  default     = "neural-nexus-app"
}

variable "ecs_service_name" {
  description = "Name for the ECS Service."
  type        = string
  default     = "neural-nexus-service"
}

# Container Images (Will be overridden by Jenkins TF_VAR_* environment variables)
variable "frontend_image" {
  description = "Docker image for the frontend container."
  type        = string
  default     = "sohano/primary-frontend:latest" # Default if TF_VAR_frontend_image not set
}

variable "backend_image" {
  description = "Docker image for the backend container."
  type        = string
  default     = "sohano/primary-backend:latest" # Default if TF_VAR_backend_image not set
}

variable "redis_image" {
  description = "Docker image for the redis container."
  type        = string
  default     = "redis:alpine" # Default if TF_VAR_redis_image not set
}

variable "frontend_port" {
  description = "Port for the frontend container"
  type        = number
  default     = 3000
}

variable "backend_port" {
  description = "Port for the backend container"
  type        = number
  default     = 8000
}

variable "redis_port" {
  description = "Port for the redis container"
  type        = number
  default     = 6379
}


variable "desired_task_count" {
  description = "Number of tasks to run for the service."
  type        = number
  default     = 1
}

# Adjust Task CPU/Memory for 3 containers
variable "task_cpu" {
  description = "Fargate task CPU units (e.g., 1024 = 1 vCPU)."
  type        = number
  default     = 8192 # 1 vCPU
}

variable "task_memory" {
  description = "Fargate task memory MiB (e.g., 2048 = 2 GB)."
  type        = number
  default     = 16384 # 2 GB
}

# --- Tagging Variables ---
variable "common_tags" {
  description = "A map of common tags to apply to all resources."
  type        = map(string)
  default = {
    Project   = "NeuralNexus"
    ManagedBy = "Terraform"
  }
}
