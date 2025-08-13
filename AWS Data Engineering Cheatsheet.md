# AWS Data Engineering Cheat Sheet

## Table of Contents

1. [S3 (Simple Storage Service)](#s3-simple-storage-service)
2. [Lambda](#lambda)
3. [Glue](#glue)
4. [Athena](#athena)
5. [RDS (Relational Database Service)](#rds-relational-database-service)
6. [DynamoDB](#dynamodb)
7. [Kinesis](#kinesis)
8. [Apache Airflow on AWS](#apache-airflow-on-aws)
9. [Common Patterns & Best Practices](#common-patterns--best-practices)

---

## S3 (Simple Storage Service)

### Basic Operations

```python
import boto3

# Initialize S3 client
s3 = boto3.client('s3')

# Upload file
s3.upload_file('local_file.csv', 'bucket-name', 'path/to/file.csv')

# Download file
s3.download_file('bucket-name', 'path/to/file.csv', 'local_file.csv')

# List objects
response = s3.list_objects_v2(Bucket='bucket-name', Prefix='data/')
for obj in response.get('Contents', []):
    print(obj['Key'])

# Delete object
s3.delete_object(Bucket='bucket-name', Key='path/to/file.csv')
```

### Working with Pandas and S3

```python
import pandas as pd
import boto3
from io import StringIO

# Read CSV from S3
s3 = boto3.client('s3')
obj = s3.get_object(Bucket='bucket-name', Key='data.csv')
df = pd.read_csv(obj['Body'])

# Write CSV to S3
csv_buffer = StringIO()
df.to_csv(csv_buffer, index=False)
s3.put_object(Bucket='bucket-name', Key='output.csv', Body=csv_buffer.getvalue())

# Parquet operations
df.to_parquet('s3://bucket-name/data.parquet')
df = pd.read_parquet('s3://bucket-name/data.parquet')
```

### S3 Best Practices

- Use partitioning: `s3://bucket/year=2024/month=01/day=15/data.parquet`
- Enable versioning for critical data
- Use lifecycle policies for cost optimization
- Compress files (gzip, snappy for parquet)

---

## Lambda

### Basic Lambda Function Structure

```python
import json
import boto3

def lambda_handler(event, context):
    """
    Basic Lambda function handler
    """
    try:
        # Process event data
        data = event.get('data', {})

        # Your processing logic here
        result = process_data(data)

        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Success',
                'result': result
            })
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e)
            })
        }

def process_data(data):
    # Your data processing logic
    return data
```

### S3 Trigger Lambda

```python
import json
import boto3
import pandas as pd
from io import StringIO

def lambda_handler(event, context):
    s3 = boto3.client('s3')

    # Get bucket and key from S3 event
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']

    try:
        # Read file from S3
        obj = s3.get_object(Bucket=bucket, Key=key)
        df = pd.read_csv(obj['Body'])

        # Process data
        df_processed = df.dropna()
        df_processed['processed_date'] = pd.Timestamp.now()

        # Write back to S3
        csv_buffer = StringIO()
        df_processed.to_csv(csv_buffer, index=False)

        output_key = f"processed/{key}"
        s3.put_object(
            Bucket=bucket,
            Key=output_key,
            Body=csv_buffer.getvalue()
        )

        return {
            'statusCode': 200,
            'body': json.dumps(f'Processed {key}')
        }
    except Exception as e:
        print(f"Error: {str(e)}")
        raise e
```

### Lambda Layers for Dependencies

```python
# requirements.txt for layer
pandas==1.5.3
boto3==1.26.137
numpy==1.24.3
```

---

## Glue

### Glue Job Script

```python
import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job

# Initialize
args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Read from Data Catalog
datasource = glueContext.create_dynamic_frame.from_catalog(
    database="my_database",
    table_name="my_table"
)

# Transform data
mapped = ApplyMapping.apply(
    frame=datasource,
    mappings=[
        ("old_column", "string", "new_column", "string"),
        ("timestamp", "string", "event_time", "timestamp")
    ]
)

# Filter data
filtered = Filter.apply(
    frame=mapped,
    f=lambda x: x["new_column"] is not None
)

# Write to S3
glueContext.write_dynamic_frame.from_options(
    frame=filtered,
    connection_type="s3",
    connection_options={
        "path": "s3://output-bucket/processed-data/",
        "partitionKeys": ["year", "month"]
    },
    format="parquet"
)

job.commit()
```

### Glue Catalog Operations

```python
import boto3

glue = boto3.client('glue')

# Create database
glue.create_database(
    DatabaseInput={
        'Name': 'my_data_lake',
        'Description': 'Data lake database'
    }
)

# Create table
table_input = {
    'Name': 'sales_data',
    'StorageDescriptor': {
        'Columns': [
            {'Name': 'order_id', 'Type': 'string'},
            {'Name': 'customer_id', 'Type': 'string'},
            {'Name': 'amount', 'Type': 'double'},
            {'Name': 'order_date', 'Type': 'timestamp'}
        ],
        'Location': 's3://my-bucket/sales-data/',
        'InputFormat': 'org.apache.hadoop.mapred.TextInputFormat',
        'OutputFormat': 'org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat',
        'SerdeInfo': {
            'SerializationLibrary': 'org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe'
        }
    },
    'PartitionKeys': [
        {'Name': 'year', 'Type': 'string'},
        {'Name': 'month', 'Type': 'string'}
    ]
}

glue.create_table(
    DatabaseName='my_data_lake',
    TableInput=table_input
)
```

---

## Athena

### Basic Athena Queries

```python
import boto3
import time
import pandas as pd

class AthenaClient:
    def __init__(self, region='us-east-1'):
        self.client = boto3.client('athena', region_name=region)
        self.s3_output = 's3://athena-query-results-bucket/'

    def execute_query(self, query, database='default'):
        """Execute Athena query and return results"""

        # Start query execution
        response = self.client.start_query_execution(
            QueryString=query,
            QueryExecutionContext={'Database': database},
            ResultConfiguration={'OutputLocation': self.s3_output}
        )

        query_id = response['QueryExecutionId']

        # Wait for completion
        while True:
            result = self.client.get_query_execution(QueryExecutionId=query_id)
            status = result['QueryExecution']['Status']['State']

            if status in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
                break
            time.sleep(1)

        if status == 'SUCCEEDED':
            return self.get_query_results(query_id)
        else:
            raise Exception(f"Query failed with status: {status}")

    def get_query_results(self, query_id):
        """Get query results as pandas DataFrame"""
        results = self.client.get_query_results(QueryExecutionId=query_id)

        # Extract column names
        columns = [col['Label'] for col in results['ResultSet']['ResultSetMetadata']['ColumnInfo']]

        # Extract data
        rows = []
        for row in results['ResultSet']['Rows'][1:]:  # Skip header
            rows.append([col.get('VarCharValue', '') for col in row['Data']])

        return pd.DataFrame(rows, columns=columns)

# Usage
athena = AthenaClient()

# Create table
create_table_query = """
CREATE EXTERNAL TABLE sales_data (
    order_id string,
    customer_id string,
    amount double,
    order_date timestamp
)
PARTITIONED BY (
    year string,
    month string
)
STORED AS PARQUET
LOCATION 's3://my-bucket/sales-data/'
"""

athena.execute_query(create_table_query)

# Query data
query = """
SELECT
    year,
    month,
    COUNT(*) as order_count,
    SUM(amount) as total_sales
FROM sales_data
WHERE year = '2024'
GROUP BY year, month
ORDER BY month
"""

df = athena.execute_query(query)
print(df)
```

---

## RDS (Relational Database Service)

### Connecting to RDS

```python
import boto3
import psycopg2
import pandas as pd
from sqlalchemy import create_engine

# PostgreSQL connection
def connect_postgresql(host, database, username, password, port=5432):
    conn_string = f"postgresql://{username}:{password}@{host}:{port}/{database}"
    engine = create_engine(conn_string)
    return engine

# MySQL connection
def connect_mysql(host, database, username, password, port=3306):
    conn_string = f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}"
    engine = create_engine(conn_string)
    return engine

# Usage with pandas
engine = connect_postgresql('rds-endpoint.region.rds.amazonaws.com',
                          'mydb', 'username', 'password')

# Read data
df = pd.read_sql("SELECT * FROM customers WHERE created_date >= '2024-01-01'", engine)

# Write data
df.to_sql('processed_customers', engine, if_exists='replace', index=False)
```

### RDS Data Pipeline

```python
import boto3
import pandas as pd
from sqlalchemy import create_engine

class RDSDataPipeline:
    def __init__(self, rds_config, s3_bucket):
        self.engine = create_engine(
            f"postgresql://{rds_config['user']}:{rds_config['password']}@"
            f"{rds_config['host']}:{rds_config['port']}/{rds_config['database']}"
        )
        self.s3_bucket = s3_bucket
        self.s3 = boto3.client('s3')

    def extract_from_rds(self, query):
        """Extract data from RDS"""
        return pd.read_sql(query, self.engine)

    def transform_data(self, df):
        """Transform data"""
        # Example transformations
        df['processed_date'] = pd.Timestamp.now()
        df = df.dropna()
        return df

    def load_to_s3(self, df, s3_key):
        """Load data to S3"""
        df.to_parquet(f's3://{self.s3_bucket}/{s3_key}')

    def run_pipeline(self):
        """Run complete ETL pipeline"""
        # Extract
        query = """
        SELECT customer_id, order_date, amount, product_id
        FROM orders
        WHERE order_date >= CURRENT_DATE - INTERVAL '7 days'
        """
        df = self.extract_from_rds(query)

        # Transform
        df_transformed = self.transform_data(df)

        # Load
        s3_key = f"processed/orders/{pd.Timestamp.now().strftime('%Y-%m-%d')}/data.parquet"
        self.load_to_s3(df_transformed, s3_key)

        return len(df_transformed)
```

---

## DynamoDB

### Basic DynamoDB Operations

```python
import boto3
from boto3.dynamodb.conditions import Key, Attr

# Initialize DynamoDB
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('UserData')

# Put item
table.put_item(
    Item={
        'user_id': '12345',
        'name': 'John Doe',
        'email': 'john@example.com',
        'created_at': '2024-01-15T10:30:00Z',
        'metadata': {
            'source': 'web',
            'campaign': 'summer2024'
        }
    }
)

# Get item
response = table.get_item(
    Key={'user_id': '12345'}
)
item = response.get('Item')

# Query with conditions
response = table.query(
    KeyConditionExpression=Key('user_id').eq('12345')
)

# Scan with filter
response = table.scan(
    FilterExpression=Attr('created_at').begins_with('2024-01')
)

# Batch operations
with table.batch_writer() as batch:
    for i in range(100):
        batch.put_item(
            Item={
                'user_id': f'user_{i}',
                'name': f'User {i}',
                'score': i * 10
            }
        )
```

### DynamoDB Data Processing

```python
import boto3
import pandas as pd
from decimal import Decimal

class DynamoDBProcessor:
    def __init__(self, table_name):
        self.dynamodb = boto3.resource('dynamodb')
        self.table = self.dynamodb.Table(table_name)

    def scan_to_dataframe(self, filter_expression=None):
        """Scan DynamoDB table and convert to DataFrame"""

        if filter_expression:
            response = self.table.scan(FilterExpression=filter_expression)
        else:
            response = self.table.scan()

        items = response['Items']

        # Handle pagination
        while 'LastEvaluatedKey' in response:
            if filter_expression:
                response = self.table.scan(
                    ExclusiveStartKey=response['LastEvaluatedKey'],
                    FilterExpression=filter_expression
                )
            else:
                response = self.table.scan(
                    ExclusiveStartKey=response['LastEvaluatedKey']
                )
            items.extend(response['Items'])

        # Convert Decimal to float for pandas compatibility
        for item in items:
            for key, value in item.items():
                if isinstance(value, Decimal):
                    item[key] = float(value)

        return pd.DataFrame(items)

    def dataframe_to_dynamodb(self, df, batch_size=25):
        """Write DataFrame to DynamoDB"""

        with self.table.batch_writer() as batch:
            for index, row in df.iterrows():
                # Convert DataFrame row to dict
                item = row.to_dict()

                # Convert NaN to None
                item = {k: (None if pd.isna(v) else v) for k, v in item.items()}

                batch.put_item(Item=item)

                if (index + 1) % batch_size == 0:
                    print(f"Processed {index + 1} items")

# Usage
processor = DynamoDBProcessor('UserEvents')

# Read data
df = processor.scan_to_dataframe(
    filter_expression=Attr('event_date').begins_with('2024-01')
)

# Process data
df_processed = df.groupby('user_id').agg({
    'event_count': 'sum',
    'last_activity': 'max'
}).reset_index()

# Write back
processor.dataframe_to_dynamodb(df_processed)
```

---

## Kinesis

### Kinesis Data Streams

```python
import boto3
import json
import time
from datetime import datetime

class KinesisProducer:
    def __init__(self, stream_name, region='us-east-1'):
        self.kinesis = boto3.client('kinesis', region_name=region)
        self.stream_name = stream_name

    def put_record(self, data, partition_key):
        """Put single record to Kinesis stream"""
        response = self.kinesis.put_record(
            StreamName=self.stream_name,
            Data=json.dumps(data),
            PartitionKey=partition_key
        )
        return response

    def put_records_batch(self, records):
        """Put multiple records to Kinesis stream"""
        kinesis_records = []
        for record in records:
            kinesis_records.append({
                'Data': json.dumps(record['data']),
                'PartitionKey': record['partition_key']
            })

        response = self.kinesis.put_records(
            Records=kinesis_records,
            StreamName=self.stream_name
        )
        return response

class KinesisConsumer:
    def __init__(self, stream_name, region='us-east-1'):
        self.kinesis = boto3.client('kinesis', region_name=region)
        self.stream_name = stream_name

    def consume_records(self, shard_id, limit=100):
        """Consume records from Kinesis stream"""

        # Get shard iterator
        response = self.kinesis.get_shard_iterator(
            StreamName=self.stream_name,
            ShardId=shard_id,
            ShardIteratorType='LATEST'
        )

        shard_iterator = response['ShardIterator']

        while shard_iterator:
            # Get records
            response = self.kinesis.get_records(
                ShardIterator=shard_iterator,
                Limit=limit
            )

            records = response['Records']
            shard_iterator = response.get('NextShardIterator')

            # Process records
            for record in records:
                data = json.loads(record['Data'])
                yield data

            time.sleep(1)  # Avoid throttling

# Usage
producer = KinesisProducer('user-events-stream')

# Send events
events = [
    {
        'data': {'user_id': '123', 'event': 'login', 'timestamp': datetime.now().isoformat()},
        'partition_key': '123'
    },
    {
        'data': {'user_id': '456', 'event': 'purchase', 'amount': 99.99, 'timestamp': datetime.now().isoformat()},
        'partition_key': '456'
    }
]

producer.put_records_batch(events)
```

### Kinesis Analytics

```python
# Kinesis Analytics SQL query example
kinesis_analytics_query = """
CREATE STREAM aggregated_stream (
    user_id VARCHAR(32),
    event_count INTEGER,
    window_start TIMESTAMP,
    window_end TIMESTAMP
);

CREATE PUMP aggregated_pump AS INSERT INTO aggregated_stream
SELECT STREAM
    user_id,
    COUNT(*) AS event_count,
    ROWTIME_RANGE_START AS window_start,
    ROWTIME_RANGE_END AS window_end
FROM SOURCE_SQL_STREAM_001
GROUP BY
    user_id,
    RANGE(ROWTIME RANGE INTERVAL '1' MINUTE);
"""
```

---

## Apache Airflow on AWS

### Basic DAG Structure

```python
from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.amazon.aws.operators.s3 import S3FileTransformOperator
from airflow.providers.amazon.aws.operators.glue import GlueJobOperator
from airflow.providers.amazon.aws.sensors.s3 import S3KeySensor
from airflow.operators.python import PythonOperator
import boto3

default_args = {
    'owner': 'data-engineering',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'data_pipeline_dag',
    default_args=default_args,
    description='Daily data processing pipeline',
    schedule_interval='@daily',
    catchup=False
)

# Wait for file to arrive
wait_for_file = S3KeySensor(
    task_id='wait_for_source_file',
    bucket_name='source-data-bucket',
    bucket_key='daily-data/{{ ds }}/data.csv',
    timeout=300,
    poke_interval=30,
    dag=dag
)

# Process data with Glue
process_data = GlueJobOperator(
    task_id='process_data_glue',
    job_name='daily-data-processing',
    script_location='s3://glue-scripts/process_daily_data.py',
    s3_bucket='glue-job-logs',
    job_arguments={
        '--input_path': 's3://source-data-bucket/daily-data/{{ ds }}/',
        '--output_path': 's3://processed-data-bucket/daily-data/{{ ds }}/'
    },
    dag=dag
)

def validate_processed_data(**context):
    """Validate processed data"""
    s3 = boto3.client('s3')

    # Check if output files exist
    date_str = context['ds']
    prefix = f'daily-data/{date_str}/'

    response = s3.list_objects_v2(
        Bucket='processed-data-bucket',
        Prefix=prefix
    )

    if 'Contents' not in response:
        raise ValueError(f"No processed files found for {date_str}")

    print(f"Found {len(response['Contents'])} processed files")
    return True

validate_data = PythonOperator(
    task_id='validate_processed_data',
    python_callable=validate_processed_data,
    dag=dag
)

# Set dependencies
wait_for_file >> process_data >> validate_data
```

### Advanced Airflow with AWS Services

```python
from airflow import DAG
from airflow.providers.amazon.aws.operators.lambda_function import LambdaInvokeFunctionOperator
from airflow.providers.amazon.aws.operators.athena import AthenaOperator
from airflow.providers.amazon.aws.operators.sns import SnsPublishOperator
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

def check_data_quality(**context):
    """Check data quality using Athena"""
    import boto3

    athena = boto3.client('athena')

    query = """
    SELECT
        COUNT(*) as total_records,
        COUNT(DISTINCT customer_id) as unique_customers,
        AVG(amount) as avg_amount
    FROM processed_sales
    WHERE partition_date = '{}'
    """.format(context['ds'])

    # Execute query and check results
    # Implementation details...

    return {'quality_check': 'passed'}

dag = DAG(
    'ml_data_pipeline',
    default_args=default_args,
    description='ML data pipeline with quality checks',
    schedule_interval='@daily'
)

# Process raw data
process_raw_data = LambdaInvokeFunctionOperator(
    task_id='process_raw_data',
    function_name='data-processing-lambda',
    payload=json.dumps({
        'date': '{{ ds }}',
        'source_bucket': 'raw-data-bucket',
        'target_bucket': 'processed-data-bucket'
    }),
    dag=dag
)

# Create Athena table partition
create_partition = AthenaOperator(
    task_id='create_athena_partition',
    query="""
    ALTER TABLE processed_sales
    ADD IF NOT EXISTS PARTITION (partition_date='{{ ds }}')
    LOCATION 's3://processed-data-bucket/sales/{{ ds }}/'
    """,
    database='data_lake',
    output_location='s3://athena-results/',
    dag=dag
)

# Data quality check
quality_check = PythonOperator(
    task_id='data_quality_check',
    python_callable=check_data_quality,
    dag=dag
)

# Send notification
send_notification = SnsPublishOperator(
    task_id='send_completion_notification',
    target_arn='arn:aws:sns:us-east-1:123456789012:data-pipeline-notifications',
    message='Data pipeline completed successfully for {{ ds }}',
    subject='Data Pipeline Success',
    dag=dag
)

# Dependencies
process_raw_data >> create_partition >> quality_check >> send_notification
```

---

## Common Patterns & Best Practices

### 1. ETL Pipeline Pattern

```python
class ETLPipeline:
    def __init__(self, config):
        self.s3 = boto3.client('s3')
        self.config = config

    def extract(self, source_path):
        """Extract data from various sources"""
        if source_path.startswith('s3://'):
            return pd.read_parquet(source_path)
        elif source_path.startswith('rds://'):
            # RDS extraction logic
            pass
        elif source_path.startswith('dynamodb://'):
            # DynamoDB extraction logic
            pass

    def transform(self, df):
        """Apply transformations"""
        # Data cleaning
        df = df.dropna()

        # Data validation
        df = df[df['amount'] > 0]

        # Feature engineering
        df['order_month'] = pd.to_datetime(df['order_date']).dt.month

        return df

    def load(self, df, target_path):
        """Load data to target"""
        if target_path.startswith('s3://'):
            df.to_parquet(target_path, partition_cols=['order_month'])
        elif target_path.startswith('rds://'):
            # RDS loading logic
            pass

    def run(self, source_path, target_path):
        """Run complete ETL pipeline"""
        df = self.extract(source_path)
        df_transformed = self.transform(df)
        self.load(df_transformed, target_path)

        return len(df_transformed)
```

### 2. Error Handling and Retry Logic

```python
import functools
import time
import logging

def retry_with_backoff(max_retries=3, backoff_factor=2):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logging.error(f"Final attempt failed: {e}")
                        raise

                    wait_time = backoff_factor ** attempt
                    logging.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s")
                    time.sleep(wait_time)
        return wrapper
    return decorator

@retry_with_backoff(max_retries=3)
def upload_to_s3(file_path, bucket, key):
    s3 = boto3.client('s3')
    s3.upload_file(file_path, bucket, key)
```

### 3. Configuration Management

```python
import json
import boto3

class ConfigManager:
    def __init__(self, parameter_store_prefix='/data-pipeline/'):
        self.ssm = boto3.client('ssm')
        self.prefix = parameter_store_prefix

    def get_config(self, config_name):
        """Get configuration from Parameter Store"""
        parameter_name = f"{self.prefix}{config_name}"

        response = self.ssm.get_parameter(
            Name=parameter_name,
            WithDecryption=True
        )

        return json.loads(response['Parameter']['Value'])

    def get_database_config(self):
        """Get database configuration"""
        return self.get_config('database')

    def get_s3_config(self):
        """Get S3 configuration"""
        return self.get_config('s3')

# Usage
config_manager = ConfigManager()
db_config = config_manager.get_database_config()
s3_config = config_manager.get_s3_config()
```

### 4. Monitoring and Logging

```python
import logging
import boto3
from datetime import datetime

class CloudWatchLogger:
    def __init__(self, log_group_name, log_stream_name):
        self.logs = boto3.client('logs')
        self.log_group = log_group_name
        self.log_stream = log_stream_name

        # Create log stream if it doesn't exist
        try:
            self.logs.create_log_stream(
                logGroupName=self.log_group,
                logStreamName=self.log_stream
            )
        except self.logs.exceptions.ResourceAlreadyExistsException:
            pass

    def log_metric(self, metric_name, value, unit='Count'):
        """Send custom metric to CloudWatch"""
        cloudwatch = boto3.client('cloudwatch')

        cloudwatch.put_metric_data(
            Namespace='DataPipeline',
            MetricData=[
                {
                    'MetricName': metric_name,
                    'Value': value,
                    'Unit': unit,
                    'Timestamp': datetime.utcnow()
                }
            ]
        )

    def log_pipeline_status(self, pipeline_name, status, records_processed=0):
        """Log pipeline execution status"""
        message = {
            'pipeline': pipeline_name,
            'status': status,
            'records_processed': records_processed,
            'timestamp': datetime.utcnow().isoformat()
        }

        logging.info(json.dumps(message))

        # Send metric to CloudWatch
        self.log_metric(f'{pipeline_name}_records_processed', records_processed)
        self.log_metric(f'{pipeline_name}_status', 1 if status == 'success' else 0)

# Usage
logger = CloudWatchLogger('data-pipeline-logs', 'pipeline-execution')
logger.log_pipeline_status('daily-etl', 'success', 10000)
```

### 5. Data Validation Framework

```python
import pandas as pd
from typing import List, Dict, Any

class DataValidator:
    def __init__(self):
        self.validation_results = []

    def validate_schema(self, df: pd.DataFrame, expected_columns: List[str]) -> bool:
        """Validate DataFrame schema"""
        missing_columns = set(expected_columns) - set(df.columns)
        if missing_columns:
            self.validation_results.append({
                'test': 'schema_validation',
                'status': 'failed',
                'message': f'Missing columns: {missing_columns}'
            })
            return False

        self.validation_results.append({
            'test': 'schema_validation',
            'status': 'passed',
            'message': 'All expected columns present'
        })
        return True

    def validate_data_quality(self, df: pd.DataFrame, rules: Dict[str, Any]) -> bool:
        """Validate data quality based on rules"""
        all_passed = True

        for column, rule in rules.items():
            if column not in df.columns:
                continue

            if 'null_threshold' in rule:
                null_percentage = df[column].isnull().sum() / len(df) * 100
                if null_percentage > rule['null_threshold']:
                    self.validation_results.append({
                        'test': f'{column}_null_check',
                        'status': 'failed',
                        'message': f'Null percentage {null_percentage:.2f}% exceeds threshold {rule["null_threshold"]}%'
                    })
                    all_passed = False

            if 'min_value' in rule:
                min_val = df[column].min()
                if min_val < rule['min_value']:
                    self.validation_results.append({
                        'test': f'{column}_min_value_check',
                        'status': 'failed',
                        'message': f'Minimum value {min_val} below threshold {rule["min_value"]}'
                    })
                    all_passed = False

            if 'unique_threshold' in rule:
                unique_percentage = df[column].nunique() / len(df) * 100
                if unique_percentage < rule['unique_threshold']:
                    self.validation_results.append({
                        'test': f'{column}_uniqueness_check',
                        'status': 'failed',
                        'message': f'Uniqueness {unique_percentage:.2f}% below threshold {rule["unique_threshold"]}%'
                    })
                    all_passed = False

        return all_passed

    def get_validation_report(self) -> Dict[str, Any]:
        """Get validation report"""
        total_tests = len(self.validation_results)
        passed_tests = sum(1 for result in self.validation_results if result['status'] == 'passed')

        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'success_rate': passed_tests / total_tests * 100 if total_tests > 0 else 0,
            'details': self.validation_results
        }

# Usage
validator = DataValidator()

# Define validation rules
validation_rules = {
    'customer_id': {'null_threshold': 0, 'unique_threshold': 95},
    'amount': {'null_threshold': 5, 'min_value': 0},
    'order_date': {'null_threshold': 0}
}

# Validate data
expected_columns = ['customer_id', 'amount', 'order_date', 'product_id']
schema_valid = validator.validate_schema(df, expected_columns)
quality_valid = validator.validate_data_quality(df, validation_rules)

# Get report
report = validator.get_validation_report()
print(json.dumps(report, indent=2))
```

### 6. Cost Optimization Tips

```python
# S3 Cost Optimization
class S3CostOptimizer:
    def __init__(self, bucket_name):
        self.s3 = boto3.client('s3')
        self.bucket = bucket_name

    def setup_lifecycle_policy(self):
        """Setup S3 lifecycle policy"""
        lifecycle_policy = {
            'Rules': [
                {
                    'ID': 'data-archival-policy',
                    'Status': 'Enabled',
                    'Filter': {'Prefix': 'raw-data/'},
                    'Transitions': [
                        {
                            'Days': 30,
                            'StorageClass': 'STANDARD_IA'
                        },
                        {
                            'Days': 90,
                            'StorageClass': 'GLACIER'
                        },
                        {
                            'Days': 365,
                            'StorageClass': 'DEEP_ARCHIVE'
                        }
                    ]
                },
                {
                    'ID': 'temp-data-cleanup',
                    'Status': 'Enabled',
                    'Filter': {'Prefix': 'temp/'},
                    'Expiration': {'Days': 7}
                }
            ]
        }

        self.s3.put_bucket_lifecycle_configuration(
            Bucket=self.bucket,
            LifecycleConfiguration=lifecycle_policy
        )

    def enable_intelligent_tiering(self):
        """Enable S3 Intelligent Tiering"""
        self.s3.put_bucket_intelligent_tiering_configuration(
            Bucket=self.bucket,
            Id='EntireBucket',
            IntelligentTieringConfiguration={
                'Id': 'EntireBucket',
                'Status': 'Enabled',
                'Filter': {},
                'OptionalFields': ['BucketKeyStatus']
            }
        )

# Lambda Cost Optimization
def optimize_lambda_memory():
    """Example of Lambda memory optimization"""
    import time
    import psutil

    start_time = time.time()
    start_memory = psutil.virtual_memory().used

    # Your processing code here
    result = process_data()

    end_time = time.time()
    end_memory = psutil.virtual_memory().used

    # Log metrics for optimization
    execution_time = end_time - start_time
    memory_used = end_memory - start_memory

    print(f"Execution time: {execution_time:.2f}s")
    print(f"Memory used: {memory_used / 1024 / 1024:.2f}MB")

    return result
```

### 7. Security Best Practices

```python
import boto3
from botocore.exceptions import ClientError

class SecurityManager:
    def __init__(self):
        self.sts = boto3.client('sts')
        self.iam = boto3.client('iam')

    def assume_role(self, role_arn, session_name):
        """Assume IAM role for cross-account access"""
        try:
            response = self.sts.assume_role(
                RoleArn=role_arn,
                RoleSessionName=session_name,
                DurationSeconds=3600
            )

            credentials = response['Credentials']

            # Create session with assumed role credentials
            session = boto3.Session(
                aws_access_key_id=credentials['AccessKeyId'],
                aws_secret_access_key=credentials['SecretAccessKey'],
                aws_session_token=credentials['SessionToken']
            )

            return session

        except ClientError as e:
            print(f"Error assuming role: {e}")
            return None

    def encrypt_s3_object(self, bucket, key, data, kms_key_id=None):
        """Upload encrypted object to S3"""
        s3 = boto3.client('s3')

        put_object_args = {
            'Bucket': bucket,
            'Key': key,
            'Body': data,
            'ServerSideEncryption': 'AES256'
        }

        if kms_key_id:
            put_object_args.update({
                'ServerSideEncryption': 'aws:kms',
                'SSEKMSKeyId': kms_key_id
            })

        s3.put_object(**put_object_args)

# Environment-specific configurations
class EnvironmentConfig:
    def __init__(self, environment='dev'):
        self.environment = environment
        self.config = self._load_config()

    def _load_config(self):
        configs = {
            'dev': {
                's3_bucket': 'dev-data-bucket',
                'glue_role': 'arn:aws:iam::123456789012:role/GlueDevRole',
                'lambda_memory': 512,
                'athena_workgroup': 'dev-workgroup'
            },
            'prod': {
                's3_bucket': 'prod-data-bucket',
                'glue_role': 'arn:aws:iam::123456789012:role/GlueProdRole',
                'lambda_memory': 1024,
                'athena_workgroup': 'prod-workgroup'
            }
        }
        return configs.get(self.environment, configs['dev'])

    def get(self, key):
        return self.config.get(key)

# Usage
env_config = EnvironmentConfig('prod')
bucket = env_config.get('s3_bucket')
```

### 8. Performance Optimization

```python
import concurrent.futures
import multiprocessing
from functools import partial

class PerformanceOptimizer:
    @staticmethod
    def parallel_s3_upload(files_to_upload, bucket_name, max_workers=5):
        """Upload multiple files to S3 in parallel"""
        s3 = boto3.client('s3')

        def upload_file(file_info):
            local_path, s3_key = file_info
            try:
                s3.upload_file(local_path, bucket_name, s3_key)
                return f"Success: {s3_key}"
            except Exception as e:
                return f"Error uploading {s3_key}: {str(e)}"

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(upload_file, files_to_upload))

        return results

    @staticmethod
    def batch_dynamodb_operations(table_name, items, operation='put', batch_size=25):
        """Perform batch operations on DynamoDB"""
        dynamodb = boto3.resource('dynamodb')
        table = dynamodb.Table(table_name)

        def process_batch(batch_items):
            if operation == 'put':
                with table.batch_writer() as batch:
                    for item in batch_items:
                        batch.put_item(Item=item)
            elif operation == 'delete':
                with table.batch_writer() as batch:
                    for item in batch_items:
                        batch.delete_item(Key=item)

        # Split items into batches
        batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

        # Process batches in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            executor.map(process_batch, batches)

    @staticmethod
    def optimize_pandas_operations(df):
        """Optimize pandas operations for large datasets"""
        # Use categorical data for string columns with limited unique values
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() / len(df) < 0.5:
                df[col] = df[col].astype('category')

        # Downcast numeric types
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')

        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')

        return df

# Usage examples
optimizer = PerformanceOptimizer()

# Parallel S3 uploads
files = [('local1.csv', 'data/file1.csv'), ('local2.csv', 'data/file2.csv')]
results = optimizer.parallel_s3_upload(files, 'my-bucket')

# Batch DynamoDB operations
items = [{'id': str(i), 'value': i * 10} for i in range(1000)]
optimizer.batch_dynamodb_operations('my-table', items)

# Optimize DataFrame
df_optimized = optimizer.optimize_pandas_operations(df)
```

### 9. Testing Framework

```python
import unittest
from unittest.mock import patch, MagicMock
import boto3
from moto import mock_s3, mock_dynamodb, mock_lambda

class TestDataPipeline(unittest.TestCase):

    @mock_s3
    def test_s3_operations(self):
        """Test S3 operations with moto"""
        # Create mock S3 bucket
        s3 = boto3.client('s3', region_name='us-east-1')
        s3.create_bucket(Bucket='test-bucket')

        # Test upload
        s3.put_object(Bucket='test-bucket', Key='test.txt', Body=b'test data')

        # Test download
        response = s3.get_object(Bucket='test-bucket', Key='test.txt')
        data = response['Body'].read()

        self.assertEqual(data, b'test data')

    @mock_dynamodb
    def test_dynamodb_operations(self):
        """Test DynamoDB operations with moto"""
        dynamodb = boto3.resource('dynamodb', region_name='us-east-1')

        # Create table
        table = dynamodb.create_table(
            TableName='test-table',
            KeySchema=[{'AttributeName': 'id', 'KeyType': 'HASH'}],
            AttributeDefinitions=[{'AttributeName': 'id', 'AttributeType': 'S'}],
            BillingMode='PAY_PER_REQUEST'
        )

        # Test put item
        table.put_item(Item={'id': 'test-id', 'data': 'test-data'})

        # Test get item
        response = table.get_item(Key={'id': 'test-id'})
        item = response['Item']

        self.assertEqual(item['data'], 'test-data')

    def test_etl_pipeline(self):
        """Test ETL pipeline logic"""
        # Mock data
        test_data = pd.DataFrame({
            'customer_id': ['1', '2', '3'],
            'amount': [100, 200, 300],
            'order_date': ['2024-01-01', '2024-01-02', '2024-01-03']
        })

        # Test transformation
        pipeline = ETLPipeline({})
        result = pipeline.transform(test_data)

        self.assertEqual(len(result), 3)
        self.assertIn('order_month', result.columns)

if __name__ == '__main__':
    unittest.main()
```

### 10. Deployment and Infrastructure as Code

```python
# CloudFormation template example (YAML)
cloudformation_template = """
AWSTemplateFormatVersion: '2010-09-09'
Description: 'Data Engineering Pipeline Infrastructure'

Parameters:
  Environment:
    Type: String
    Default: dev
    AllowedValues: [dev, staging, prod]

Resources:
  # S3 Buckets
  DataLakeBucket:
    Type: AWS::S3::Bucket
    Properties:
      BucketName: !Sub 'data-lake-${Environment}-${AWS::AccountId}'
      VersioningConfiguration:
        Status: Enabled
      PublicAccessBlockConfiguration:
        BlockPublicAcls: true
        BlockPublicPolicy: true
        IgnorePublicAcls: true
        RestrictPublicBuckets: true

  # Glue Database
  GlueDatabase:
    Type: AWS::Glue::Database
    Properties:
      CatalogId: !Ref AWS::AccountId
      DatabaseInput:
        Name: !Sub '${Environment}_data_lake'
        Description: 'Data lake database'

  # Lambda Execution Role
  LambdaExecutionRole:
    Type: AWS::IAM::Role
    Properties:
      AssumeRolePolicyDocument:
        Version: '2012-10-17'
        Statement:
          - Effect: Allow
            Principal:
              Service: lambda.amazonaws.com
            Action: sts:AssumeRole
      ManagedPolicyArns:
        - arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
      Policies:
        - PolicyName: S3Access
          PolicyDocument:
            Version: '2012-10-17'
            Statement:
              - Effect: Allow
                Action:
                  - s3:GetObject
                  - s3:PutObject
                  - s3:DeleteObject
                Resource: !Sub '${DataLakeBucket}/*'

Outputs:
  DataLakeBucketName:
    Description: 'Data Lake S3 Bucket Name'
    Value: !Ref DataLakeBucket
    Export:
      Name: !Sub '${Environment}-DataLakeBucket'
"""

# Terraform example
terraform_config = """
# terraform/main.tf
provider "aws" {
  region = var.aws_region
}

# S3 Bucket for Data Lake
resource "aws_s3_bucket" "data_lake" {
  bucket = "${var.environment}-data-lake-${random_string.bucket_suffix.result}"
}

resource "aws_s3_bucket_versioning" "data_lake_versioning" {
  bucket = aws_s3_bucket.data_lake.id
  versioning_configuration {
    status = "Enabled"
  }
}

# DynamoDB Table
resource "aws_dynamodb_table" "metadata_table" {
  name           = "${var.environment}-pipeline-metadata"
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "pipeline_id"

  attribute {
    name = "pipeline_id"
    type = "S"
  }

  tags = {
    Environment = var.environment
    Purpose     = "DataPipeline"
  }
}

# Lambda Function
resource "aws_lambda_function" "data_processor" {
  filename         = "data_processor.zip"
  function_name    = "${var.environment}-data-processor"
  role            = aws_iam_role.lambda_role.arn
  handler         = "lambda_function.lambda_handler"
  runtime         = "python3.9"
  timeout         = 300
  memory_size     = 512

  environment {
    variables = {
      ENVIRONMENT = var.environment
      S3_BUCKET   = aws_s3_bucket.data_lake.bucket
    }
  }
}
"""
```

## Quick Reference Commands

### AWS CLI Commands

```bash
# S3 operations
aws s3 cp file.csv s3://bucket/path/
aws s3 sync ./local-folder s3://bucket/folder/
aws s3 ls s3://bucket/path/ --recursive

# Glue operations
aws glue start-job-run --job-name my-glue-job
aws glue get-job-run --job-name my-glue-job --run-id jr_xxx

# Lambda operations
aws lambda invoke --function-name my-function output.json
aws lambda update-function-code --function-name my-function --zip-file fileb://function.zip

# Athena operations
aws athena start-query-execution --query-string "SELECT * FROM my_table" --result-configuration OutputLocation=s3://results/

# DynamoDB operations
aws dynamodb scan --table-name my-table
aws dynamodb put-item --table-name my-table --item '{"id":{"S":"123"},"name":{"S":"John"}}'
```

### Environment Variables

```bash
# Common environment variables for AWS services
export AWS_DEFAULT_REGION=us-east-1
export AWS_PROFILE=data-engineering
export PYTHONPATH=/opt/python/lib/python3.9/site-packages

# For Glue jobs
export GLUE_VERSION=3.0
export PYTHON_VERSION=3

# For Lambda
export LAMBDA_RUNTIME=python3.9
export LAMBDA_TIMEOUT=300
```

### Performance Tips

- Use columnar formats (Parquet, ORC) for analytical workloads
- Partition data by frequently queried columns (date, region, etc.)
- Use compression (gzip for text, snappy for Parquet)
- Implement connection pooling for database connections
- Use batch operations instead of individual API calls
- Enable S3 Transfer Acceleration for large uploads
- Use Glue bookmarks to process only new data
- Optimize Lambda memory allocation based on CPU requirements

This cheat sheet provides a comprehensive reference for AWS data engineering tasks. Keep it handy for quick lookups and modify the examples based on your specific use cases!
