# --- Provider Configuration ---
provider "aws" {
  region = var.aws_region
}

# --- Locals ---
locals {
  # Use specified AZs or dynamically get them based on num_azs
  azs      = length(var.availability_zones) > 0 ? var.availability_zones : slice(data.aws_availability_zones.available.names, 0, var.num_azs)
  num_azs  = length(local.azs)
  tags     = merge(var.common_tags, { Terraform = "true" }) # Ensure Terraform tag exists
}

# Data source to get available AZs in the region
data "aws_availability_zones" "available" {
  state = "available"
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

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = merge(local.tags, {
    Name = "${var.project_name}-igw"
  })
}

resource "aws_subnet" "public" {
  count                   = local.num_azs
  vpc_id                  = aws_vpc.main.id
  cidr_block              = element(var.public_subnet_cidrs, count.index)
  availability_zone       = element(local.azs, count.index)
  map_public_ip_on_launch = true # Often useful for public subnets

  tags = merge(local.tags, {
    Name = "${var.project_name}-public-subnet-${count.index + 1}"
    Tier = "Public"
  })
}

resource "aws_subnet" "private" {
  count             = local.num_azs
  vpc_id            = aws_vpc.main.id
  cidr_block        = element(var.private_subnet_cidrs, count.index)
  availability_zone = element(local.azs, count.index)

  tags = merge(local.tags, {
    Name = "${var.project_name}-private-subnet-${count.index + 1}"
    Tier = "Private"
  })
}

resource "aws_eip" "nat" {
  count = local.num_azs # Create one EIP per AZ for NAT Gateway
  domain = "vpc" 

  tags = merge(local.tags, {
    Name = "${var.project_name}-nat-eip-${count.index + 1}"
  })
}

resource "aws_nat_gateway" "main" {
  count         = local.num_azs
  allocation_id = aws_eip.nat[count.index].id
  subnet_id     = aws_subnet.public[count.index].id

  tags = merge(local.tags, {
    Name = "${var.project_name}-nat-gw-${count.index + 1}"
  })

  # Ensure IGW is created before NAT Gateway
  depends_on = [aws_internet_gateway.main]
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }

  tags = merge(local.tags, {
    Name = "${var.project_name}-public-rt"
  })
}

resource "aws_route_table" "private" {
  count  = local.num_azs
  vpc_id = aws_vpc.main.id

  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.main[count.index].id
  }

  tags = merge(local.tags, {
    Name = "${var.project_name}-private-rt-${count.index + 1}"
  })
}

resource "aws_route_table_association" "public" {
  count          = local.num_azs
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table_association" "private" {
  count          = local.num_azs
  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private[count.index].id
}


# --- Security ---
resource "aws_security_group" "alb" {
  name        = "${var.project_name}-alb-sg"
  description = "Allow HTTP traffic to ALB"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port   = 80 # HTTP
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"] # Allow traffic from anywhere
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(local.tags, {
    Name = "${var.project_name}-alb-sg"
  })
}

resource "aws_security_group" "ecs_tasks" {
  name        = "${var.project_name}-ecs-tasks-sg"
  description = "Allow traffic for ECS tasks"
  vpc_id      = aws_vpc.main.id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1" # All protocols
    cidr_blocks = ["0.0.0.0/0"]
  }
  # Allow traffic ONLY from the ALB Security Group
  ingress {
    from_port       = 0 # Allow any port from the ALB
    to_port         = 0
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  tags = merge(local.tags, {
    Name = "${var.project_name}-ecs-tasks-sg"
  })
}

# --- IAM for ECS Task Execution ---
resource "aws_iam_role" "ecs_task_execution_role" {
  name = "${var.project_name}-ecsTaskExecutionRole"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })

  tags = merge(local.tags, {
    Name = "${var.project_name}-ecsTaskExecutionRole"
  })
}

resource "aws_iam_role_policy_attachment" "ecs_task_execution_role_policy" {
  role       = aws_iam_role.ecs_task_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# --- ECS Cluster ---
resource "aws_ecs_cluster" "main" {
  name = "${var.project_name}-cluster"

  tags = merge(local.tags, {
    Name = "${var.project_name}-cluster"
  })
}

# --- Application Load Balancer (ALB) ---
resource "aws_lb" "main" {
  name               = "${var.project_name}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  idle_timeout=150
  subnets            = aws_subnet.public[*].id # Place ALB in public subnets

  tags = merge(local.tags, {
    Name = "${var.project_name}-alb"
  })
}

resource "aws_lb_target_group" "frontend" {
  name        = "${var.project_name}-frontend-tg"
  port        = var.frontend_port
  protocol    = "HTTP"
  vpc_id      = aws_vpc.main.id
  target_type = "ip" # Required for Fargate

  health_check {
    path                = "/" # Basic health check for frontend, adjust if needed
    protocol            = "HTTP"
    matcher             = "200-399"
    interval            = 30
    timeout             = 5
    healthy_threshold   = 2
    unhealthy_threshold = 2
  }

  tags = merge(local.tags, {
    Name = "${var.project_name}-frontend-tg"
  })
}

resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.main.arn
  port              = "80" # Listen on HTTP
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.frontend.arn
  }
}


# --- ECS Task Definition and Service ---
resource "aws_ecs_task_definition" "app" {
  family                   = "${var.project_name}-${var.ecs_task_family}"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.task_cpu
  memory                   = var.task_memory
  execution_role_arn       = aws_iam_role.ecs_task_execution_role.arn

  container_definitions = jsonencode([
    # --- Frontend Container Definition ---
    {
      name      = "frontend"
      image     = var.frontend_image
      essential = true
      environment = [
        { name = "NEXT_PUBLIC_API_URL", value = "http://localhost:${var.backend_port}/api/v1" }
      ]
      portMappings = [
        {
          containerPort = var.frontend_port
          hostPort      = var.frontend_port
          protocol      = "tcp"
        }
      ]
      healthCheck = {
        command     = ["CMD-SHELL", "curl -f http://localhost:${var.frontend_port}/ || exit 1"]
        interval    = 30 # seconds
        timeout     = 5  # seconds
        retries     = 3
        startPeriod = 400 # seconds, give frontend time to initialize (especially after backend is healthy)
      }
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.ecs_logs.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "frontend"
        }
      }
    },
    # --- Backend Container Definition ---
    {
      name      = "backend"
      image     = var.backend_image
      essential = true
      portMappings = [{ containerPort = var.backend_port, hostPort = var.backend_port, protocol = "tcp" }]
      # Internal health check for the backend
      healthCheck = {
        # IMPORTANT: Replace '/health' with your backend's actual health check endpoint
        command     = ["CMD-SHELL", "curl -f http://localhost:${var.backend_port}/health || exit 1"]
        interval    = 30 # seconds
        timeout     = 5  # seconds
        retries     = 3
        startPeriod = 400 # seconds, give backend time to initialize before checks start
      }
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.ecs_logs.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "backend"
        }
      }
    },
    # --- Redis Container Definition ---
    {
      name      = "redis"
      image     = var.redis_image
      essential = true
      portMappings = [{ containerPort = var.redis_port, hostPort = var.redis_port, protocol = "tcp" }]
      # Internal health check for Redis
      healthCheck = {
        command     = ["CMD-SHELL", "redis-cli -h localhost -p ${var.redis_port} ping | grep PONG || exit 1"]
        interval    = 30 # seconds
        timeout     = 5  # seconds
        retries     = 3
        startPeriod = 400 # seconds, give redis time to initialize
      }
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.ecs_logs.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "redis"
        }
      }
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
    subnets         = aws_subnet.private[*].id # Run tasks in private subnets
    security_groups = [aws_security_group.ecs_tasks.id]
    assign_public_ip = false # Tasks are private, accessed via ALB
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.frontend.arn # Attach to frontend TG
    container_name   = "frontend"                       # Route traffic to frontend container
    container_port   = var.frontend_port
  }

  # Ensure dependencies are created first
  depends_on = [
    aws_lb_listener.http,
    aws_iam_role_policy_attachment.ecs_task_execution_role_policy,
    aws_cloudwatch_log_group.ecs_logs # Depend on log group if defined
  ]

  # Optional: Enable deployment circuit breaker and rollback
  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }

  tags = merge(local.tags, {
    Name = "${var.project_name}-${var.ecs_service_name}"
  })
}

# --- Optional: CloudWatch Log Group for ECS Tasks ---
resource "aws_cloudwatch_log_group" "ecs_logs" {
  name = "/ecs/${var.project_name}-${var.ecs_task_family}" # Standard naming convention

  tags = merge(local.tags, {
    Name = "${var.project_name}-ecs-logs"
  })
}
