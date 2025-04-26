terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0" # Use a recent version
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# --- Dynamically get Availability Zones for the chosen region ---
data "aws_availability_zones" "available" {
  state = "available"
}

locals {
  # Use explicitly provided AZs or dynamically fetch them, ensuring we only use as many as defined subnets
  azs = length(var.availability_zones) == 0 ? slice(data.aws_availability_zones.available.names, 0, length(var.public_subnet_cidrs)) : var.availability_zones
  tags = {
    Project     = var.project_name
    Environment = "dev" # Or manage via variable
    ManagedBy   = "Terraform"
  }
}

# --- Networking ---

resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = merge(local.tags, {
    Name = "${var.project_name}-vpc"
  })
}

resource "aws_internet_gateway" "gw" {
  vpc_id = aws_vpc.main.id

  tags = merge(local.tags, {
    Name = "${var.project_name}-igw"
  })
}

resource "aws_subnet" "public" {
  count                   = length(var.public_subnet_cidrs)
  vpc_id                  = aws_vpc.main.id
  cidr_block              = var.public_subnet_cidrs[count.index]
  availability_zone       = local.azs[count.index]
  map_public_ip_on_launch = true # Important for public subnets

  tags = merge(local.tags, {
    Name = "${var.project_name}-public-subnet-${local.azs[count.index]}"
  })
}

resource "aws_subnet" "private" {
  count             = length(var.private_subnet_cidrs)
  vpc_id            = aws_vpc.main.id
  cidr_block        = var.private_subnet_cidrs[count.index]
  availability_zone = local.azs[count.index]

  tags = merge(local.tags, {
    Name = "${var.project_name}-private-subnet-${local.azs[count.index]}"
  })
}

# --- NAT Gateway for Private Subnet Outbound Access ---

resource "aws_eip" "nat" {
  count = length(var.public_subnet_cidrs) # One EIP per AZ/public subnet
  domain   = "vpc" # depends_on = [aws_internet_gateway.gw] # Implicit dependency via vpc_id usage

  tags = merge(local.tags, {
    Name = "${var.project_name}-nat-eip-${local.azs[count.index]}"
  })
}

resource "aws_nat_gateway" "nat" {
  count         = length(var.public_subnet_cidrs)
  allocation_id = aws_eip.nat[count.index].id
  subnet_id     = aws_subnet.public[count.index].id

  tags = merge(local.tags, {
    Name = "${var.project_name}-nat-gw-${local.azs[count.index]}"
  })

  # Ensure IGW is created before NAT GW
  depends_on = [aws_internet_gateway.gw]
}

# --- Routing ---

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.gw.id
  }

  tags = merge(local.tags, {
    Name = "${var.project_name}-public-rt"
  })
}

resource "aws_route_table_association" "public" {
  count          = length(aws_subnet.public)
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table" "private" {
  count  = length(var.private_subnet_cidrs)
  vpc_id = aws_vpc.main.id

  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.nat[count.index].id
  }

  tags = merge(local.tags, {
    Name = "${var.project_name}-private-rt-${local.azs[count.index]}"
  })
}

resource "aws_route_table_association" "private" {
  count          = length(aws_subnet.private)
  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private[count.index].id
}

# --- Security ---

resource "aws_security_group" "ecs_tasks" {
  name        = "${var.project_name}-ecs-tasks-sg"
  description = "Allow traffic for ECS tasks"
  vpc_id      = aws_vpc.main.id

  # Default: Allow all outbound traffic
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1" # All protocols
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Add ingress rules as needed for your application
  # Example: Allow HTTP from anywhere (adjust for production, maybe only from Load Balancer)
  # ingress {
  #   from_port   = 80
  #   to_port     = 80
  #   protocol    = "tcp"
  #   cidr_blocks = ["0.0.0.0/0"]
  # }

  tags = merge(local.tags, {
    Name = "${var.project_name}-ecs-tasks-sg"
  })
}

# --- IAM Role for ECS Task Execution ---

data "aws_iam_policy_document" "ecs_task_execution_assume_role" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "ecs_task_execution_role" {
  name               = "${var.project_name}-ecsTaskExecutionRole"
  assume_role_policy = data.aws_iam_policy_document.ecs_task_execution_assume_role.json
  tags               = local.tags
}

resource "aws_iam_role_policy_attachment" "ecs_task_execution_role_policy" {
  role       = aws_iam_role.ecs_task_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# --- ECS Cluster ---

resource "aws_ecs_cluster" "main" {
  name = "${var.project_name}-${var.ecs_cluster_name}"

  tags = merge(local.tags, {
    Name = "${var.project_name}-${var.ecs_cluster_name}"
  })
}

# --- Optional: Example ECS Task Definition and Service (using Fargate) ---

resource "aws_ecs_task_definition" "app" {
  family                   = "${var.project_name}-${var.ecs_task_family}"
  network_mode             = "awsvpc" # Required for Fargate
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.task_cpu
  memory                   = var.task_memory
  execution_role_arn       = aws_iam_role.ecs_task_execution_role.arn
  # Optional: task_role_arn = aws_iam_role.ecs_task_role.arn # If your app needs AWS permissions

  # Container definition using JSON syntax within Terraform
  container_definitions = jsonencode([
    {
      name      = "${var.project_name}-container"
      image     = var.container_image
      cpu       = var.task_cpu   # Can allocate specific CPU/Memory here too
      memory    = var.task_memory
      essential = true
      portMappings = [
        {
          containerPort = var.container_port
          hostPort      = var.container_port # For awsvpc mode, hostPort is often same as containerPort
          protocol      = "tcp"
        }
      ]
      # Add log configuration if needed (e.g., awslogs)
      # logConfiguration = {
      #   logDriver = "awslogs"
      #   options = {
      #     "awslogs-group"         = aws_cloudwatch_log_group.ecs_logs.name
      #     "awslogs-region"        = var.aws_region
      #     "awslogs-stream-prefix" = "ecs"
      #   }
      # }
    }
  ])

  tags = merge(local.tags, {
    Name = "${var.project_name}-${var.ecs_task_family}"
  })
}

resource "aws_ecs_service" "app" {
  name            = "${var.project_name}-${var.ecs_service_name}"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.app.arn
  desired_count   = var.desired_task_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets = aws_subnet.private[*].id # Run tasks in private subnets
    # Alternatively use public subnets: aws_subnet.public[*].id
    security_groups = [aws_security_group.ecs_tasks.id]
    assign_public_ip = false # Set to true if using public subnets and need direct public IP
  }

  # Optional: Load Balancer configuration
  # load_balancer {
  #   target_group_arn = aws_lb_target_group.app.arn # Reference your LB Target Group
  #   container_name   = "${var.project_name}-container"
  #   container_port   = var.container_port
  # }

  # Optional: Service Discovery configuration
  # service_registries {
  #   registry_arn = aws_service_discovery_service.example.arn
  # }

  # Ensure Task Definition is created before the service
  depends_on = [aws_iam_role_policy_attachment.ecs_task_execution_role_policy] # And potentially LB/Service Discovery

  tags = merge(local.tags, {
    Name = "${var.project_name}-${var.ecs_service_name}"
  })
}

# Optional: CloudWatch Log Group for container logs
# resource "aws_cloudwatch_log_group" "ecs_logs" {
#   name = "/ecs/${var.project_name}/${var.ecs_task_family}"
#   retention_in_days = 7 # Adjust as needed
#
#   tags = merge(local.tags, {
#     Name = "${var.project_name}-ecs-logs"
#   })
# }
