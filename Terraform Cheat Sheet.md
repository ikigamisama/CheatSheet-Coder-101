# Terraform for Data Engineering Cheat Sheet

## Terraform Basics

### Core Concepts

- **Infrastructure as Code (IaC)**: Define infrastructure using configuration files
- **Providers**: Plugins that interact with APIs (AWS, Azure, GCP)
- **Resources**: Infrastructure components (EC2, S3 bucket, etc.)
- **Data Sources**: Read-only information from existing infrastructure
- **Variables**: Input parameters for configurations
- **Outputs**: Return values from configurations
- **State**: Current state of managed infrastructure

### Essential Commands

```bash
# Initialize Terraform (download providers)
terraform init

# Validate configuration syntax
terraform validate

# Preview changes
terraform plan

# Apply changes
terraform apply

# Destroy infrastructure
terraform destroy

# Format code
terraform fmt

# Show current state
terraform show

# Import existing resources
terraform import <resource_type>.<name> <resource_id>
```

## Project Structure for Data Engineering

```
data-platform/
├── environments/
│   ├── dev/
│   ├── staging/
│   └── prod/
├── modules/
│   ├── data-lake/
│   ├── data-warehouse/
│   ├── etl-pipeline/
│   └── analytics/
├── variables.tf
├── terraform.tfvars
├── main.tf
├── outputs.tf
└── versions.tf
```

## Data Storage Infrastructure

### S3 Data Lake Setup

```hcl
# S3 Buckets for Data Lake
resource "aws_s3_bucket" "data_lake_raw" {
  bucket = "${var.project_name}-data-lake-raw-${var.environment}"

  tags = {
    Environment = var.environment
    Purpose     = "raw-data-storage"
    DataClassification = "internal"
  }
}

resource "aws_s3_bucket" "data_lake_processed" {
  bucket = "${var.project_name}-data-lake-processed-${var.environment}"

  tags = {
    Environment = var.environment
    Purpose     = "processed-data-storage"
  }
}

# Versioning
resource "aws_s3_bucket_versioning" "data_lake_raw_versioning" {
  bucket = aws_s3_bucket.data_lake_raw.id
  versioning_configuration {
    status = "Enabled"
  }
}

# Lifecycle Management
resource "aws_s3_bucket_lifecycle_configuration" "data_lake_lifecycle" {
  bucket = aws_s3_bucket.data_lake_raw.id

  rule {
    id     = "transition_to_ia"
    status = "Enabled"

    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }

    transition {
      days          = 90
      storage_class = "GLACIER"
    }

    transition {
      days          = 365
      storage_class = "DEEP_ARCHIVE"
    }
  }
}

# Server-side Encryption
resource "aws_s3_bucket_server_side_encryption_configuration" "data_lake_encryption" {
  bucket = aws_s3_bucket.data_lake_raw.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}
```

### Redshift Data Warehouse

```hcl
# Redshift Subnet Group
resource "aws_redshift_subnet_group" "data_warehouse" {
  name       = "${var.project_name}-redshift-subnet-group"
  subnet_ids = var.private_subnet_ids

  tags = {
    Name = "${var.project_name}-redshift-subnet-group"
  }
}

# Redshift Cluster
resource "aws_redshift_cluster" "data_warehouse" {
  cluster_identifier      = "${var.project_name}-data-warehouse"
  database_name          = var.redshift_database_name
  master_username        = var.redshift_master_username
  master_password        = var.redshift_master_password
  node_type              = var.redshift_node_type
  cluster_type           = var.redshift_cluster_type
  number_of_nodes        = var.redshift_number_of_nodes

  db_subnet_group_name   = aws_redshift_subnet_group.data_warehouse.name
  vpc_security_group_ids = [aws_security_group.redshift.id]

  # Backup and Maintenance
  automated_snapshot_retention_period = 7
  preferred_maintenance_window        = "sun:05:00-sun:06:00"

  # Encryption
  encrypted  = true
  kms_key_id = aws_kms_key.redshift.arn

  # Enhanced VPC Routing
  enhanced_vpc_routing = true

  skip_final_snapshot = true

  tags = {
    Environment = var.environment
    Purpose     = "data-warehouse"
  }
}
```

### RDS for Metadata Storage

```hcl
# RDS Subnet Group
resource "aws_db_subnet_group" "metadata_db" {
  name       = "${var.project_name}-metadata-db-subnet-group"
  subnet_ids = var.private_subnet_ids

  tags = {
    Name = "${var.project_name}-metadata-db-subnet-group"
  }
}

# RDS Instance
resource "aws_db_instance" "metadata_db" {
  identifier             = "${var.project_name}-metadata-db"
  engine                 = "postgres"
  engine_version         = "13.7"
  instance_class         = "db.t3.micro"
  allocated_storage      = 20
  storage_encrypted      = true

  db_name  = "metadata"
  username = var.db_username
  password = var.db_password

  db_subnet_group_name   = aws_db_subnet_group.metadata_db.name
  vpc_security_group_ids = [aws_security_group.rds.id]

  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"

  skip_final_snapshot = true

  tags = {
    Environment = var.environment
    Purpose     = "metadata-storage"
  }
}
```

## Data Processing Infrastructure

### AWS Glue Resources

```hcl
# Glue Database
resource "aws_glue_catalog_database" "data_catalog" {
  name = "${var.project_name}_data_catalog"

  description = "Data catalog for ${var.project_name}"
}

# Glue Crawler
resource "aws_glue_crawler" "s3_crawler" {
  database_name = aws_glue_catalog_database.data_catalog.name
  name          = "${var.project_name}-s3-crawler"
  role          = aws_iam_role.glue_crawler_role.arn

  s3_target {
    path = "s3://${aws_s3_bucket.data_lake_raw.bucket}/"
  }

  configuration = jsonencode({
    Version = 1.0
    CrawlerOutput = {
      Partitions = { AddOrUpdateBehavior = "InheritFromTable" }
    }
  })

  schedule = "cron(0 2 * * ? *)"  # Daily at 2 AM

  tags = {
    Environment = var.environment
    Purpose     = "schema-discovery"
  }
}

# Glue ETL Job
resource "aws_glue_job" "etl_job" {
  name     = "${var.project_name}-etl-job"
  role_arn = aws_iam_role.glue_job_role.arn

  command {
    script_location = "s3://${aws_s3_bucket.scripts.bucket}/etl_script.py"
    python_version  = "3"
  }

  default_arguments = {
    "--job-bookmark-option"              = "job-bookmark-enable"
    "--enable-metrics"                   = ""
    "--enable-continuous-cloudwatch-log" = "true"
    "--TempDir"                         = "s3://${aws_s3_bucket.temp.bucket}/"
  }

  execution_property {
    max_concurrent_runs = 2
  }

  max_retries    = 1
  timeout        = 60
  glue_version   = "3.0"
  worker_type    = "G.1X"
  number_of_workers = 2

  tags = {
    Environment = var.environment
    Purpose     = "data-transformation"
  }
}
```

### Lambda Functions for Data Processing

```hcl
# Lambda Function for Data Processing
resource "aws_lambda_function" "data_processor" {
  filename         = "data_processor.zip"
  function_name    = "${var.project_name}-data-processor"
  role            = aws_iam_role.lambda_role.arn
  handler         = "lambda_function.lambda_handler"
  runtime         = "python3.9"
  timeout         = 300
  memory_size     = 512

  environment {
    variables = {
      S3_BUCKET = aws_s3_bucket.data_lake_processed.bucket
      DB_HOST   = aws_db_instance.metadata_db.endpoint
    }
  }

  tags = {
    Environment = var.environment
    Purpose     = "data-processing"
  }
}

# S3 Event Trigger
resource "aws_s3_bucket_notification" "s3_notification" {
  bucket = aws_s3_bucket.data_lake_raw.id

  lambda_function {
    lambda_function_arn = aws_lambda_function.data_processor.arn
    events              = ["s3:ObjectCreated:*"]
    filter_prefix       = "incoming/"
    filter_suffix       = ".json"
  }

  depends_on = [aws_lambda_permission.s3_lambda_permission]
}
```

### Kinesis for Streaming Data

```hcl
# Kinesis Data Stream
resource "aws_kinesis_stream" "data_stream" {
  name        = "${var.project_name}-data-stream"
  shard_count = var.kinesis_shard_count

  shard_level_metrics = [
    "IncomingRecords",
    "OutgoingRecords",
  ]

  stream_mode_details {
    stream_mode = "PROVISIONED"
  }

  tags = {
    Environment = var.environment
    Purpose     = "real-time-data-ingestion"
  }
}

# Kinesis Firehose
resource "aws_kinesis_firehose_delivery_stream" "data_firehose" {
  name        = "${var.project_name}-data-firehose"
  destination = "s3"

  s3_configuration {
    role_arn   = aws_iam_role.firehose_role.arn
    bucket_arn = aws_s3_bucket.data_lake_raw.arn
    prefix     = "year=!{timestamp:yyyy}/month=!{timestamp:MM}/day=!{timestamp:dd}/hour=!{timestamp:HH}/"

    buffer_size     = 5
    buffer_interval = 300

    compression_format = "GZIP"
  }

  tags = {
    Environment = var.environment
    Purpose     = "streaming-data-delivery"
  }
}
```

## Analytics Infrastructure

### EMR Cluster

```hcl
# EMR Cluster
resource "aws_emr_cluster" "analytics_cluster" {
  name          = "${var.project_name}-analytics-cluster"
  release_label = "emr-6.4.0"
  applications  = ["Spark", "Hadoop", "Hive", "JupyterHub"]

  termination_protection            = false
  keep_job_flow_alive_when_no_steps = true

  ec2_attributes {
    subnet_id                         = var.private_subnet_id
    emr_managed_master_security_group = aws_security_group.emr_master.id
    emr_managed_slave_security_group  = aws_security_group.emr_slave.id
    instance_profile                  = aws_iam_instance_profile.emr_profile.arn
    key_name                         = var.key_pair_name
  }

  master_instance_group {
    instance_type = "m5.xlarge"
  }

  core_instance_group {
    instance_type  = "m5.large"
    instance_count = 2

    ebs_config {
      size                 = 40
      type                 = "gp2"
      volumes_per_instance = 1
    }
  }

  service_role = aws_iam_role.emr_service_role.arn

  tags = {
    Environment = var.environment
    Purpose     = "big-data-analytics"
  }
}
```

## Networking and Security

### VPC Setup for Data Platform

```hcl
# VPC
resource "aws_vpc" "data_platform_vpc" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "${var.project_name}-data-platform-vpc"
  }
}

# Private Subnets
resource "aws_subnet" "private_subnets" {
  count             = length(var.private_subnet_cidrs)
  vpc_id            = aws_vpc.data_platform_vpc.id
  cidr_block        = var.private_subnet_cidrs[count.index]
  availability_zone = var.availability_zones[count.index]

  tags = {
    Name = "${var.project_name}-private-subnet-${count.index + 1}"
  }
}

# NAT Gateway
resource "aws_nat_gateway" "nat_gateway" {
  allocation_id = aws_eip.nat_eip.id
  subnet_id     = aws_subnet.public_subnets[0].id

  tags = {
    Name = "${var.project_name}-nat-gateway"
  }
}

# VPC Endpoints
resource "aws_vpc_endpoint" "s3" {
  vpc_id       = aws_vpc.data_platform_vpc.id
  service_name = "com.amazonaws.${var.aws_region}.s3"

  tags = {
    Name = "${var.project_name}-s3-endpoint"
  }
}

resource "aws_vpc_endpoint" "glue" {
  vpc_id              = aws_vpc.data_platform_vpc.id
  service_name        = "com.amazonaws.${var.aws_region}.glue"
  vpc_endpoint_type   = "Interface"
  subnet_ids          = aws_subnet.private_subnets[*].id
  security_group_ids  = [aws_security_group.vpc_endpoints.id]

  tags = {
    Name = "${var.project_name}-glue-endpoint"
  }
}
```

### Security Groups

```hcl
# Redshift Security Group
resource "aws_security_group" "redshift" {
  name_prefix = "${var.project_name}-redshift-"
  vpc_id      = aws_vpc.data_platform_vpc.id

  ingress {
    from_port   = 5439
    to_port     = 5439
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project_name}-redshift-sg"
  }
}

# RDS Security Group
resource "aws_security_group" "rds" {
  name_prefix = "${var.project_name}-rds-"
  vpc_id      = aws_vpc.data_platform_vpc.id

  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }

  tags = {
    Name = "${var.project_name}-rds-sg"
  }
}
```

## IAM Roles and Policies

### Glue Service Roles

```hcl
# Glue Service Role
resource "aws_iam_role" "glue_job_role" {
  name = "${var.project_name}-glue-job-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "glue.amazonaws.com"
        }
      }
    ]
  })
}

# Glue Policy
resource "aws_iam_role_policy" "glue_job_policy" {
  name = "${var.project_name}-glue-job-policy"
  role = aws_iam_role.glue_job_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.data_lake_raw.arn,
          "${aws_s3_bucket.data_lake_raw.arn}/*",
          aws_s3_bucket.data_lake_processed.arn,
          "${aws_s3_bucket.data_lake_processed.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "glue:*",
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "*"
      }
    ]
  })
}

# Attach AWS managed policy
resource "aws_iam_role_policy_attachment" "glue_service_role" {
  role       = aws_iam_role.glue_job_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSGlueServiceRole"
}
```

## Variables and Outputs

### Variables Definition

```hcl
# variables.tf
variable "project_name" {
  description = "Name of the data engineering project"
  type        = string
}

variable "environment" {
  description = "Environment (dev, staging, prod)"
  type        = string
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-west-2"
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "private_subnet_cidrs" {
  description = "CIDR blocks for private subnets"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24"]
}

variable "redshift_node_type" {
  description = "Redshift node type"
  type        = string
  default     = "dc2.large"
}

variable "redshift_number_of_nodes" {
  description = "Number of Redshift nodes"
  type        = number
  default     = 2
}
```

### Outputs

```hcl
# outputs.tf
output "s3_data_lake_raw_bucket" {
  description = "Name of the raw data S3 bucket"
  value       = aws_s3_bucket.data_lake_raw.bucket
}

output "s3_data_lake_processed_bucket" {
  description = "Name of the processed data S3 bucket"
  value       = aws_s3_bucket.data_lake_processed.bucket
}

output "redshift_cluster_endpoint" {
  description = "Redshift cluster endpoint"
  value       = aws_redshift_cluster.data_warehouse.endpoint
  sensitive   = true
}

output "glue_catalog_database_name" {
  description = "Glue catalog database name"
  value       = aws_glue_catalog_database.data_catalog.name
}

output "kinesis_stream_name" {
  description = "Kinesis stream name"
  value       = aws_kinesis_stream.data_stream.name
}
```

## Best Practices for Data Engineering

### State Management

```hcl
# Backend configuration for remote state
terraform {
  backend "s3" {
    bucket         = "your-terraform-state-bucket"
    key            = "data-platform/terraform.tfstate"
    region         = "us-west-2"
    encrypt        = true
    dynamodb_table = "terraform-state-lock"
  }
}
```

### Environment-Specific Configurations

```hcl
# terraform.tfvars.example
project_name = "my-data-platform"
environment  = "dev"
aws_region   = "us-west-2"

# Development specific
redshift_node_type        = "dc2.large"
redshift_number_of_nodes  = 1
kinesis_shard_count      = 1

# Production would have different values
# redshift_node_type        = "ra3.xlplus"
# redshift_number_of_nodes  = 3
# kinesis_shard_count      = 5
```

### Module Structure Example

```hcl
# modules/data-lake/main.tf
resource "aws_s3_bucket" "data_lake" {
  bucket = var.bucket_name

  tags = var.tags
}

# modules/data-lake/variables.tf
variable "bucket_name" {
  description = "Name of the S3 bucket"
  type        = string
}

variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default     = {}
}

# modules/data-lake/outputs.tf
output "bucket_name" {
  value = aws_s3_bucket.data_lake.bucket
}

output "bucket_arn" {
  value = aws_s3_bucket.data_lake.arn
}
```

### Using Modules

```hcl
# main.tf
module "data_lake" {
  source = "./modules/data-lake"

  bucket_name = "${var.project_name}-data-lake-${var.environment}"

  tags = {
    Environment = var.environment
    Project     = var.project_name
    ManagedBy   = "terraform"
  }
}
```

## Common Terraform Patterns for Data Engineering

### Data Sources for Existing Resources

```hcl
# Reference existing VPC
data "aws_vpc" "existing_vpc" {
  filter {
    name   = "tag:Name"
    values = ["existing-vpc"]
  }
}

# Use existing subnets
data "aws_subnets" "existing_subnets" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.existing_vpc.id]
  }
}
```

### Conditional Resource Creation

```hcl
resource "aws_redshift_cluster" "data_warehouse" {
  count = var.create_redshift ? 1 : 0

  cluster_identifier = "${var.project_name}-data-warehouse"
  # ... other configuration
}
```

### Dynamic Blocks

```hcl
resource "aws_security_group" "dynamic_sg" {
  name_prefix = "${var.project_name}-dynamic-sg"
  vpc_id      = var.vpc_id

  dynamic "ingress" {
    for_each = var.ingress_rules
    content {
      from_port   = ingress.value.from_port
      to_port     = ingress.value.to_port
      protocol    = ingress.value.protocol
      cidr_blocks = ingress.value.cidr_blocks
    }
  }
}
```

## Troubleshooting Common Issues

### State File Issues

```bash
# Refresh state
terraform refresh

# Import existing resource
terraform import aws_s3_bucket.example bucket-name

# Remove resource from state
terraform state rm aws_s3_bucket.example

# Move resource in state
terraform state mv aws_s3_bucket.old aws_s3_bucket.new
```

### Dependency Issues

```hcl
# Explicit dependency
resource "aws_s3_bucket" "processed" {
  depends_on = [aws_s3_bucket.raw]
  # ...
}

# Implicit dependency (preferred)
resource "aws_s3_bucket_policy" "policy" {
  bucket = aws_s3_bucket.raw.id  # Creates implicit dependency
  # ...
}
```

### Validation and Testing

```bash
# Validate syntax
terraform validate

# Plan with detailed output
terraform plan -detailed-exitcode

# Apply with auto-approve (use carefully)
terraform apply -auto-approve

# Target specific resources
terraform plan -target=aws_s3_bucket.data_lake

# Use workspace for environments
terraform workspace new staging
terraform workspace select staging
```
