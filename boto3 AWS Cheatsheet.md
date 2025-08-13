# Comprehensive Boto3 Cheat Sheet for AWS

## Installation & Setup

```bash
pip install boto3
```

### Authentication Methods

```python
import boto3

# Method 1: Default credentials (recommended)
client = boto3.client('s3')

# Method 2: Explicit credentials
client = boto3.client(
    's3',
    aws_access_key_id='YOUR_ACCESS_KEY',
    aws_secret_access_key='YOUR_SECRET_KEY',
    region_name='us-east-1'
)

# Method 3: Using profiles
session = boto3.Session(profile_name='dev')
client = session.client('s3')

# Method 4: IAM roles (for EC2 instances)
client = boto3.client('s3')  # Automatically uses instance role
```

## Core Concepts

### Client vs Resource

```python
# Client: Low-level service access
s3_client = boto3.client('s3')

# Resource: Higher-level object-oriented interface
s3_resource = boto3.resource('s3')
```

### Sessions

```python
# Custom session with specific region/profile
session = boto3.Session(
    profile_name='production',
    region_name='us-west-2'
)
ec2 = session.resource('ec2')
```

## S3 (Simple Storage Service)

### Bucket Operations

```python
s3 = boto3.client('s3')

# List buckets
response = s3.list_buckets()
for bucket in response['Buckets']:
    print(bucket['Name'])

# Create bucket
s3.create_bucket(Bucket='my-bucket')

# Delete bucket
s3.delete_bucket(Bucket='my-bucket')

# Check if bucket exists
try:
    s3.head_bucket(Bucket='my-bucket')
    print("Bucket exists")
except ClientError:
    print("Bucket doesn't exist")
```

### Object Operations

```python
# Upload file
s3.upload_file('local_file.txt', 'my-bucket', 'remote_file.txt')

# Download file
s3.download_file('my-bucket', 'remote_file.txt', 'local_file.txt')

# Upload with metadata
s3.put_object(
    Bucket='my-bucket',
    Key='file.txt',
    Body=b'Hello World',
    Metadata={'author': 'john', 'version': '1.0'}
)

# List objects
response = s3.list_objects_v2(Bucket='my-bucket', Prefix='folder/')
for obj in response.get('Contents', []):
    print(obj['Key'])

# Delete object
s3.delete_object(Bucket='my-bucket', Key='file.txt')

# Generate presigned URL
url = s3.generate_presigned_url(
    'get_object',
    Params={'Bucket': 'my-bucket', 'Key': 'file.txt'},
    ExpiresIn=3600
)
```

### S3 Resource Interface

```python
s3 = boto3.resource('s3')

# Access bucket
bucket = s3.Bucket('my-bucket')

# Upload file
bucket.upload_file('local.txt', 'remote.txt')

# Iterate through objects
for obj in bucket.objects.filter(Prefix='photos/'):
    print(obj.key)
```

## EC2 (Elastic Compute Cloud)

### Instance Management

```python
ec2 = boto3.client('ec2')

# Launch instance
response = ec2.run_instances(
    ImageId='ami-12345678',
    MinCount=1,
    MaxCount=1,
    InstanceType='t2.micro',
    KeyName='my-key-pair',
    SecurityGroupIds=['sg-12345678'],
    SubnetId='subnet-12345678'
)

# List instances
response = ec2.describe_instances()
for reservation in response['Reservations']:
    for instance in reservation['Instances']:
        print(f"Instance ID: {instance['InstanceId']}")
        print(f"State: {instance['State']['Name']}")

# Start/Stop instances
ec2.start_instances(InstanceIds=['i-12345678'])
ec2.stop_instances(InstanceIds=['i-12345678'])

# Terminate instances
ec2.terminate_instances(InstanceIds=['i-12345678'])

# Create security group
sg = ec2.create_security_group(
    GroupName='my-sg',
    Description='My security group'
)

# Add rules to security group
ec2.authorize_security_group_ingress(
    GroupId=sg['GroupId'],
    IpPermissions=[
        {
            'IpProtocol': 'tcp',
            'FromPort': 80,
            'ToPort': 80,
            'IpRanges': [{'CidrIp': '0.0.0.0/0'}]
        }
    ]
)
```

### Resource Interface

```python
ec2 = boto3.resource('ec2')

# Get instance
instance = ec2.Instance('i-12345678')
print(instance.state)
print(instance.public_ip_address)

# Start instance
instance.start()
instance.wait_until_running()

# Create key pair
key_pair = ec2.create_key_pair(KeyName='my-key')
with open('my-key.pem', 'w') as f:
    f.write(key_pair.key_material)
```

## Lambda

### Function Management

```python
lambda_client = boto3.client('lambda')

# List functions
response = lambda_client.list_functions()
for func in response['Functions']:
    print(func['FunctionName'])

# Invoke function
response = lambda_client.invoke(
    FunctionName='my-function',
    Payload=json.dumps({'key': 'value'})
)
result = json.loads(response['Payload'].read())

# Create function
with open('function.zip', 'rb') as f:
    zip_content = f.read()

lambda_client.create_function(
    FunctionName='my-function',
    Runtime='python3.9',
    Role='arn:aws:iam::123456789012:role/lambda-role',
    Handler='lambda_function.lambda_handler',
    Code={'ZipFile': zip_content}
)

# Update function code
lambda_client.update_function_code(
    FunctionName='my-function',
    ZipFile=zip_content
)
```

## DynamoDB

### Table Operations

```python
dynamodb = boto3.client('dynamodb')

# Create table
dynamodb.create_table(
    TableName='Users',
    KeySchema=[
        {'AttributeName': 'user_id', 'KeyType': 'HASH'}
    ],
    AttributeDefinitions=[
        {'AttributeName': 'user_id', 'AttributeType': 'S'}
    ],
    BillingMode='PAY_PER_REQUEST'
)

# List tables
response = dynamodb.list_tables()
print(response['TableNames'])

# Delete table
dynamodb.delete_table(TableName='Users')
```

### Item Operations

```python
# Put item
dynamodb.put_item(
    TableName='Users',
    Item={
        'user_id': {'S': '123'},
        'name': {'S': 'John Doe'},
        'age': {'N': '30'}
    }
)

# Get item
response = dynamodb.get_item(
    TableName='Users',
    Key={'user_id': {'S': '123'}}
)
item = response.get('Item')

# Query items
response = dynamodb.query(
    TableName='Users',
    KeyConditionExpression='user_id = :uid',
    ExpressionAttributeValues={':uid': {'S': '123'}}
)

# Scan table
response = dynamodb.scan(TableName='Users')
for item in response['Items']:
    print(item)
```

### Resource Interface (Higher Level)

```python
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('Users')

# Put item
table.put_item(Item={'user_id': '123', 'name': 'John', 'age': 30})

# Get item
response = table.get_item(Key={'user_id': '123'})
item = response.get('Item')

# Query
response = table.query(KeyConditionExpression=Key('user_id').eq('123'))

# Scan
response = table.scan()
```

## IAM (Identity and Access Management)

```python
iam = boto3.client('iam')

# List users
response = iam.list_users()
for user in response['Users']:
    print(user['UserName'])

# Create user
iam.create_user(UserName='new-user')

# Create access key
response = iam.create_access_key(UserName='new-user')
print(response['AccessKey']['AccessKeyId'])

# Attach policy to user
iam.attach_user_policy(
    UserName='new-user',
    PolicyArn='arn:aws:iam::aws:policy/ReadOnlyAccess'
)

# Create role
trust_policy = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {"Service": "lambda.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }
    ]
}

iam.create_role(
    RoleName='lambda-role',
    AssumeRolePolicyDocument=json.dumps(trust_policy)
)
```

## CloudFormation

```python
cf = boto3.client('cloudformation')

# Create stack
with open('template.yaml', 'r') as f:
    template = f.read()

cf.create_stack(
    StackName='my-stack',
    TemplateBody=template,
    Parameters=[
        {'ParameterKey': 'Environment', 'ParameterValue': 'dev'}
    ]
)

# List stacks
response = cf.list_stacks()
for stack in response['StackSummaries']:
    print(f"{stack['StackName']}: {stack['StackStatus']}")

# Update stack
cf.update_stack(
    StackName='my-stack',
    TemplateBody=template
)

# Delete stack
cf.delete_stack(StackName='my-stack')

# Describe stack resources
response = cf.describe_stack_resources(StackName='my-stack')
```

## CloudWatch

### Logs

```python
logs = boto3.client('logs')

# Create log group
logs.create_log_group(logGroupName='/aws/lambda/my-function')

# List log groups
response = logs.describe_log_groups()
for group in response['logGroups']:
    print(group['logGroupName'])

# Put log events
logs.put_log_events(
    logGroupName='/aws/lambda/my-function',
    logStreamName='stream-name',
    logEvents=[
        {
            'timestamp': int(time.time() * 1000),
            'message': 'Hello from Lambda'
        }
    ]
)
```

### Metrics

```python
cloudwatch = boto3.client('cloudwatch')

# Put custom metric
cloudwatch.put_metric_data(
    Namespace='MyApp',
    MetricData=[
        {
            'MetricName': 'RequestCount',
            'Value': 1,
            'Unit': 'Count',
            'Dimensions': [
                {'Name': 'Environment', 'Value': 'prod'}
            ]
        }
    ]
)

# Get metric statistics
response = cloudwatch.get_metric_statistics(
    Namespace='AWS/EC2',
    MetricName='CPUUtilization',
    Dimensions=[{'Name': 'InstanceId', 'Value': 'i-12345678'}],
    StartTime=datetime.utcnow() - timedelta(hours=1),
    EndTime=datetime.utcnow(),
    Period=300,
    Statistics=['Average']
)
```

## SQS (Simple Queue Service)

```python
sqs = boto3.client('sqs')

# Create queue
response = sqs.create_queue(QueueName='my-queue')
queue_url = response['QueueUrl']

# Send message
sqs.send_message(
    QueueUrl=queue_url,
    MessageBody='Hello World',
    MessageAttributes={
        'Author': {'StringValue': 'John', 'DataType': 'String'}
    }
)

# Receive messages
response = sqs.receive_message(
    QueueUrl=queue_url,
    MaxNumberOfMessages=10,
    WaitTimeSeconds=20
)

for message in response.get('Messages', []):
    print(message['Body'])
    # Delete message after processing
    sqs.delete_message(
        QueueUrl=queue_url,
        ReceiptHandle=message['ReceiptHandle']
    )

# Get queue attributes
response = sqs.get_queue_attributes(
    QueueUrl=queue_url,
    AttributeNames=['All']
)
```

## SNS (Simple Notification Service)

```python
sns = boto3.client('sns')

# Create topic
response = sns.create_topic(Name='my-topic')
topic_arn = response['TopicArn']

# Subscribe to topic
sns.subscribe(
    TopicArn=topic_arn,
    Protocol='email',
    Endpoint='user@example.com'
)

# Publish message
sns.publish(
    TopicArn=topic_arn,
    Message='Hello from SNS',
    Subject='Test Message'
)

# List subscriptions
response = sns.list_subscriptions_by_topic(TopicArn=topic_arn)
```

## RDS (Relational Database Service)

```python
rds = boto3.client('rds')

# Create DB instance
rds.create_db_instance(
    DBInstanceIdentifier='mydb',
    DBInstanceClass='db.t3.micro',
    Engine='mysql',
    MasterUsername='admin',
    MasterUserPassword='password',
    AllocatedStorage=20
)

# List DB instances
response = rds.describe_db_instances()
for db in response['DBInstances']:
    print(f"DB: {db['DBInstanceIdentifier']}")
    print(f"Status: {db['DBInstanceStatus']}")

# Create snapshot
rds.create_db_snapshot(
    DBSnapshotIdentifier='mydb-snapshot',
    DBInstanceIdentifier='mydb'
)

# Delete DB instance
rds.delete_db_instance(
    DBInstanceIdentifier='mydb',
    SkipFinalSnapshot=True
)
```

## Error Handling

```python
from botocore.exceptions import ClientError, NoCredentialsError

try:
    s3 = boto3.client('s3')
    s3.head_bucket(Bucket='non-existent-bucket')
except ClientError as e:
    error_code = e.response['Error']['Code']
    if error_code == 'NoSuchBucket':
        print("Bucket doesn't exist")
    elif error_code == 'AccessDenied':
        print("Access denied")
    else:
        print(f"Error: {error_code}")
except NoCredentialsError:
    print("No AWS credentials found")
```

## Pagination

```python
# Using paginator
s3 = boto3.client('s3')
paginator = s3.get_paginator('list_objects_v2')

for page in paginator.paginate(Bucket='my-bucket'):
    for obj in page.get('Contents', []):
        print(obj['Key'])

# Manual pagination
response = s3.list_objects_v2(Bucket='my-bucket')
while True:
    for obj in response.get('Contents', []):
        print(obj['Key'])

    if not response.get('IsTruncated'):
        break

    response = s3.list_objects_v2(
        Bucket='my-bucket',
        ContinuationToken=response['NextContinuationToken']
    )
```

## Configuration & Best Practices

### Retry Configuration

```python
from botocore.config import Config

config = Config(
    retries={'max_attempts': 10, 'mode': 'adaptive'},
    max_pool_connections=50
)
s3 = boto3.client('s3', config=config)
```

### Environment Variables

```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1
export AWS_PROFILE=default
```

### Common Patterns

```python
# Resource cleanup with context manager
import contextlib

@contextlib.contextmanager
def temporary_bucket(bucket_name):
    s3 = boto3.client('s3')
    s3.create_bucket(Bucket=bucket_name)
    try:
        yield bucket_name
    finally:
        # Delete all objects first
        response = s3.list_objects_v2(Bucket=bucket_name)
        for obj in response.get('Contents', []):
            s3.delete_object(Bucket=bucket_name, Key=obj['Key'])
        s3.delete_bucket(Bucket=bucket_name)

# Usage
with temporary_bucket('temp-bucket-123') as bucket:
    # Use bucket
    pass
```

## Useful CLI Equivalents

| CLI Command                            | Boto3 Equivalent                                   |
| -------------------------------------- | -------------------------------------------------- |
| `aws s3 ls`                            | `s3.list_buckets()`                                |
| `aws s3 cp file.txt s3://bucket/`      | `s3.upload_file('file.txt', 'bucket', 'file.txt')` |
| `aws ec2 describe-instances`           | `ec2.describe_instances()`                         |
| `aws lambda list-functions`            | `lambda_client.list_functions()`                   |
| `aws dynamodb scan --table-name Users` | `table.scan()`                                     |

## Quick Reference: Common Operations

```python
# Get AWS account ID
sts = boto3.client('sts')
account_id = sts.get_caller_identity()['Account']

# Get current region
session = boto3.Session()
region = session.region_name

# List all available services
available_services = boto3.Session().get_available_services()

# Get service endpoints
s3 = boto3.client('s3')
print(s3._endpoint.host)
```
