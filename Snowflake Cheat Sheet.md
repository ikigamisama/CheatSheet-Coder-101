# Snowflake SQL Comprehensive Cheat Sheet

## Basic Query Structure

```sql
SELECT column1, column2
FROM database.schema.table
WHERE condition
GROUP BY column1
HAVING condition
ORDER BY column1 DESC
LIMIT 100;
```

## Warehouses & Compute Management

```sql
-- Create warehouse
CREATE WAREHOUSE my_warehouse
WITH WAREHOUSE_SIZE = 'MEDIUM'
     AUTO_SUSPEND = 300          -- seconds
     AUTO_RESUME = TRUE
     MIN_CLUSTER_COUNT = 1
     MAX_CLUSTER_COUNT = 3
     SCALING_POLICY = 'STANDARD'
     COMMENT = 'General purpose warehouse';

-- Warehouse sizes and their approximate credits per hour
-- X-SMALL: 1 credit/hour
-- SMALL: 2 credits/hour
-- MEDIUM: 4 credits/hour
-- LARGE: 8 credits/hour
-- X-LARGE: 16 credits/hour
-- 2X-LARGE: 32 credits/hour
-- 3X-LARGE: 64 credits/hour
-- 4X-LARGE: 128 credits/hour

-- Use warehouse
USE WAREHOUSE my_warehouse;

-- Show warehouses
SHOW WAREHOUSES;

-- Alter warehouse
ALTER WAREHOUSE my_warehouse SET
    WAREHOUSE_SIZE = 'LARGE'
    AUTO_SUSPEND = 600;

-- Suspend/Resume warehouse
ALTER WAREHOUSE my_warehouse SUSPEND;
ALTER WAREHOUSE my_warehouse RESUME;

-- Drop warehouse
DROP WAREHOUSE my_warehouse;

-- Grant warehouse usage
GRANT USAGE ON WAREHOUSE my_warehouse TO ROLE my_role;
GRANT OPERATE ON WAREHOUSE my_warehouse TO ROLE admin_role;

-- Multi-cluster warehouse (Enterprise+)
CREATE WAREHOUSE analytics_warehouse
WITH WAREHOUSE_SIZE = 'LARGE'
     MIN_CLUSTER_COUNT = 1
     MAX_CLUSTER_COUNT = 10
     SCALING_POLICY = 'ECONOMY'    -- or 'STANDARD'
     AUTO_SUSPEND = 300
     AUTO_RESUME = TRUE;

-- Warehouse monitoring
SELECT
    warehouse_name,
    start_time,
    end_time,
    credits_used,
    credits_used_compute,
    credits_used_cloud_services
FROM SNOWFLAKE.ACCOUNT_USAGE.WAREHOUSE_METERING_HISTORY
WHERE start_time >= DATEADD(day, -7, CURRENT_TIMESTAMP())
ORDER BY start_time DESC;

-- Current warehouse info
SELECT CURRENT_WAREHOUSE();
SELECT SYSTEM$GET_WAREHOUSE_STATE('MY_WAREHOUSE');

-- Warehouse load monitoring
SELECT
    warehouse_name,
    avg_running_queries,
    avg_queued_load,
    avg_queued_provision_load,
    avg_blocked_queries
FROM SNOWFLAKE.ACCOUNT_USAGE.WAREHOUSE_LOAD_HISTORY
WHERE start_time >= DATEADD(hour, -1, CURRENT_TIMESTAMP());
```

## Database & Schema Operations

```sql
-- Create database
CREATE DATABASE my_database;

-- Use database
USE DATABASE my_database;

-- Create schema
CREATE SCHEMA my_schema;

-- Use schema
USE SCHEMA my_schema;

-- Show databases
SHOW DATABASES;

-- Show schemas
SHOW SCHEMAS IN DATABASE my_database;

-- Show tables
SHOW TABLES IN SCHEMA my_schema;
```

## Table Operations

### Create Table

```sql
-- Basic table
CREATE TABLE employees (
    id INTEGER,
    name VARCHAR(100),
    salary NUMBER(10,2),
    hire_date DATE
);

-- Table with constraints
CREATE TABLE employees (
    id INTEGER AUTOINCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(255) UNIQUE,
    salary NUMBER(10,2) DEFAULT 0,
    hire_date DATE DEFAULT CURRENT_DATE()
);

-- Create table from query
CREATE TABLE new_table AS
SELECT * FROM existing_table WHERE condition;

-- Create temporary table
CREATE TEMPORARY TABLE temp_table AS
SELECT * FROM source_table;
```

### Alter Table

```sql
-- Add column
ALTER TABLE employees ADD COLUMN department VARCHAR(50);

-- Drop column
ALTER TABLE employees DROP COLUMN department;

-- Rename column
ALTER TABLE employees RENAME COLUMN name TO full_name;

-- Modify column
ALTER TABLE employees ALTER COLUMN salary SET DATA TYPE NUMBER(12,2);
```

## Data Types

```sql
-- Numeric
NUMBER(precision, scale)    -- General numeric
INTEGER, INT               -- Integer
FLOAT, DOUBLE             -- Floating point
DECIMAL(p,s)              -- Fixed decimal

-- String
VARCHAR(length)           -- Variable length string
CHAR(length)             -- Fixed length string
STRING                   -- Unlimited length string
TEXT                     -- Alias for STRING

-- Date/Time
DATE                     -- Date only
TIME                     -- Time only
TIMESTAMP_LTZ           -- Timestamp with local timezone
TIMESTAMP_NTZ           -- Timestamp without timezone
TIMESTAMP_TZ            -- Timestamp with timezone

-- Semi-structured
VARIANT                  -- JSON, XML, Avro, etc.
OBJECT                   -- JSON object
ARRAY                    -- JSON array

-- Binary
BINARY(length)           -- Fixed length binary
VARBINARY(length)        -- Variable length binary

-- Boolean
BOOLEAN                  -- TRUE/FALSE
```

## Window Functions

```sql
-- Row number
SELECT name, salary,
       ROW_NUMBER() OVER (ORDER BY salary DESC) as rank
FROM employees;

-- Rank functions
SELECT name, salary,
       RANK() OVER (ORDER BY salary DESC) as rank,
       DENSE_RANK() OVER (ORDER BY salary DESC) as dense_rank,
       PERCENT_RANK() OVER (ORDER BY salary DESC) as percent_rank
FROM employees;

-- Aggregate window functions
SELECT name, salary,
       SUM(salary) OVER (PARTITION BY department) as dept_total,
       AVG(salary) OVER (PARTITION BY department) as dept_avg,
       COUNT(*) OVER (PARTITION BY department) as dept_count
FROM employees;

-- Lag/Lead
SELECT name, salary,
       LAG(salary, 1) OVER (ORDER BY hire_date) as prev_salary,
       LEAD(salary, 1) OVER (ORDER BY hire_date) as next_salary
FROM employees;

-- First/Last value
SELECT name, salary,
       FIRST_VALUE(salary) OVER (PARTITION BY department ORDER BY salary DESC) as highest_in_dept,
       LAST_VALUE(salary) OVER (PARTITION BY department ORDER BY salary DESC) as lowest_in_dept
FROM employees;
```

## Date/Time Functions

```sql
-- Current date/time
SELECT CURRENT_DATE();
SELECT CURRENT_TIME();
SELECT CURRENT_TIMESTAMP();

-- Date arithmetic
SELECT DATEADD(day, 30, CURRENT_DATE());
SELECT DATEADD(month, -6, CURRENT_DATE());
SELECT DATEDIFF(day, '2023-01-01', CURRENT_DATE());

-- Date formatting
SELECT TO_CHAR(CURRENT_DATE(), 'YYYY-MM-DD');
SELECT TO_CHAR(CURRENT_TIMESTAMP(), 'YYYY-MM-DD HH24:MI:SS');

-- Date parsing
SELECT TO_DATE('2023-12-25', 'YYYY-MM-DD');
SELECT TO_TIMESTAMP('2023-12-25 15:30:00', 'YYYY-MM-DD HH24:MI:SS');

-- Extract date parts
SELECT EXTRACT(year FROM CURRENT_DATE());
SELECT EXTRACT(month FROM CURRENT_DATE());
SELECT EXTRACT(day FROM CURRENT_DATE());
SELECT DAYOFWEEK(CURRENT_DATE());
SELECT DAYOFYEAR(CURRENT_DATE());
```

## String Functions

```sql
-- Basic string operations
SELECT UPPER('hello');              -- HELLO
SELECT LOWER('HELLO');              -- hello
SELECT LENGTH('hello');             -- 5
SELECT SUBSTR('hello', 2, 3);       -- ell
SELECT LEFT('hello', 3);            -- hel
SELECT RIGHT('hello', 3);           -- llo

-- String manipulation
SELECT CONCAT('Hello', ' ', 'World');    -- Hello World
SELECT 'Hello' || ' ' || 'World';        -- Hello World
SELECT REPLACE('Hello World', 'World', 'Snowflake');
SELECT TRIM('  hello  ');                -- hello
SELECT LTRIM('  hello');                 -- hello
SELECT RTRIM('hello  ');                 -- hello

-- String search
SELECT POSITION('o' IN 'hello');         -- 5
SELECT CHARINDEX('o', 'hello');          -- 5
SELECT CONTAINS('hello world', 'world'); -- TRUE

-- Pattern matching
SELECT * FROM table WHERE column LIKE 'A%';      -- Starts with A
SELECT * FROM table WHERE column LIKE '%ing';    -- Ends with ing
SELECT * FROM table WHERE column LIKE '%test%';  -- Contains test
SELECT * FROM table WHERE column RLIKE '^[A-Z]'; -- Regex: starts with capital
```

## Aggregate Functions

```sql
-- Basic aggregates
SELECT COUNT(*) FROM employees;
SELECT COUNT(DISTINCT department) FROM employees;
SELECT SUM(salary) FROM employees;
SELECT AVG(salary) FROM employees;
SELECT MIN(salary) FROM employees;
SELECT MAX(salary) FROM employees;

-- Statistical functions
SELECT STDDEV(salary) FROM employees;
SELECT VARIANCE(salary) FROM employees;
SELECT MEDIAN(salary) FROM employees;
SELECT MODE(department) FROM employees;

-- Array aggregates
SELECT ARRAY_AGG(name) FROM employees;
SELECT LISTAGG(name, ', ') FROM employees;
```

## Conditional Logic

```sql
-- CASE statement
SELECT name, salary,
       CASE
           WHEN salary > 100000 THEN 'High'
           WHEN salary > 50000 THEN 'Medium'
           ELSE 'Low'
       END as salary_category
FROM employees;

-- IFF (inline if)
SELECT name, IFF(salary > 50000, 'High', 'Low') as category
FROM employees;

-- COALESCE (first non-null)
SELECT COALESCE(phone, email, 'No contact') as contact
FROM employees;

-- NULLIF
SELECT NULLIF(division, '') as clean_division
FROM employees;
```

## JSON Functions (VARIANT)

```sql
-- Parse JSON
SELECT PARSE_JSON('{"name": "John", "age": 30}') as json_data;

-- Extract from JSON
SELECT json_data:name::STRING as name,
       json_data:age::INTEGER as age
FROM (SELECT PARSE_JSON('{"name": "John", "age": 30}') as json_data);

-- Array operations
SELECT json_array[0]::STRING as first_item
FROM (SELECT PARSE_JSON('["apple", "banana", "cherry"]') as json_array);

-- Flatten JSON arrays
SELECT value::STRING as item
FROM TABLE(FLATTEN(input => PARSE_JSON('["apple", "banana", "cherry"]')));

-- Check if key exists
SELECT json_data:name IS NOT NULL as has_name
FROM json_table;
```

## CTEs (Common Table Expressions)

```sql
-- Basic CTE
WITH high_earners AS (
    SELECT * FROM employees WHERE salary > 75000
),
dept_stats AS (
    SELECT department, AVG(salary) as avg_salary
    FROM high_earners
    GROUP BY department
)
SELECT * FROM dept_stats ORDER BY avg_salary DESC;

-- Recursive CTE
WITH RECURSIVE employee_hierarchy AS (
    -- Anchor
    SELECT id, name, manager_id, 0 as level
    FROM employees
    WHERE manager_id IS NULL

    UNION ALL

    -- Recursive
    SELECT e.id, e.name, e.manager_id, eh.level + 1
    FROM employees e
    JOIN employee_hierarchy eh ON e.manager_id = eh.id
)
SELECT * FROM employee_hierarchy;
```

## Joins

```sql
-- Inner join
SELECT e.name, d.department_name
FROM employees e
INNER JOIN departments d ON e.dept_id = d.id;

-- Left join
SELECT e.name, d.department_name
FROM employees e
LEFT JOIN departments d ON e.dept_id = d.id;

-- Right join
SELECT e.name, d.department_name
FROM employees e
RIGHT JOIN departments d ON e.dept_id = d.id;

-- Full outer join
SELECT e.name, d.department_name
FROM employees e
FULL OUTER JOIN departments d ON e.dept_id = d.id;

-- Cross join
SELECT e.name, p.project_name
FROM employees e
CROSS JOIN projects p;

-- Self join
SELECT e1.name as employee, e2.name as manager
FROM employees e1
JOIN employees e2 ON e1.manager_id = e2.id;
```

## Set Operations

```sql
-- Union (removes duplicates)
SELECT name FROM employees_2023
UNION
SELECT name FROM employees_2024;

-- Union all (keeps duplicates)
SELECT name FROM employees_2023
UNION ALL
SELECT name FROM employees_2024;

-- Intersect
SELECT name FROM employees_2023
INTERSECT
SELECT name FROM employees_2024;

-- Except (difference)
SELECT name FROM employees_2023
EXCEPT
SELECT name FROM employees_2024;
```

## Data Loading & Unloading

### S3 Integration & External Stages

```sql
-- Create S3 external stage with credentials
CREATE STAGE my_s3_stage
URL = 's3://mybucket/data/'
CREDENTIALS = (
    AWS_KEY_ID = 'AKIAIOSFODNN7EXAMPLE'
    AWS_SECRET_KEY = 'wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY'
);

-- Create S3 external stage with IAM role
CREATE STAGE my_s3_stage_iam
URL = 's3://mybucket/data/'
CREDENTIALS = (AWS_ROLE = 'arn:aws:iam::123456789012:role/MySnowflakeRole');

-- Create S3 external stage with storage integration
CREATE STAGE my_s3_stage_integration
URL = 's3://mybucket/data/'
STORAGE_INTEGRATION = my_s3_integration;

-- List files in S3 stage
LIST @my_s3_stage;
LIST @my_s3_stage/subfolder/;

-- Show stage details
DESCRIBE STAGE my_s3_stage;
```

### Storage Integrations (Recommended for Production)

```sql
-- Create storage integration (requires ACCOUNTADMIN role)
CREATE STORAGE INTEGRATION s3_integration
  TYPE = EXTERNAL_STAGE
  STORAGE_PROVIDER = 'S3'
  ENABLED = TRUE
  STORAGE_AWS_ROLE_ARN = 'arn:aws:iam::123456789012:role/MySnowflakeRole'
  STORAGE_ALLOWED_LOCATIONS = ('s3://mybucket/data/', 's3://mybucket/backup/');

-- Show storage integration details (get external ID for AWS trust policy)
DESCRIBE STORAGE INTEGRATION s3_integration;

-- Grant usage on storage integration
GRANT USAGE ON INTEGRATION s3_integration TO ROLE my_role;
```

### Copy Into (Loading from S3)

```sql
-- Basic S3 load
COPY INTO my_table
FROM @my_s3_stage/data.csv
FILE_FORMAT = (TYPE = 'CSV' SKIP_HEADER = 1);

-- Load specific files from S3
COPY INTO my_table
FROM @my_s3_stage
FILES = ('file1.csv', 'file2.csv')
FILE_FORMAT = (TYPE = 'CSV' SKIP_HEADER = 1);

-- Load with pattern matching
COPY INTO my_table
FROM @my_s3_stage
PATTERN = '.*sales_[0-9]{4}\.csv'
FILE_FORMAT = (TYPE = 'CSV' SKIP_HEADER = 1);

-- Load JSON from S3
COPY INTO json_table
FROM @my_s3_stage/data.json
FILE_FORMAT = (TYPE = 'JSON');

-- Load Parquet from S3
COPY INTO parquet_table
FROM @my_s3_stage/data.parquet
FILE_FORMAT = (TYPE = 'PARQUET');

-- Load with error handling
COPY INTO my_table
FROM @my_s3_stage/data.csv
FILE_FORMAT = (TYPE = 'CSV' SKIP_HEADER = 1)
ON_ERROR = 'CONTINUE'
VALIDATION_MODE = 'RETURN_ERRORS';

-- Load with transformations
COPY INTO my_table (id, name, email, created_date)
FROM (
    SELECT
        $1::INTEGER,
        UPPER($2::STRING),
        LOWER($3::STRING),
        TO_DATE($4::STRING, 'MM/DD/YYYY')
    FROM @my_s3_stage/data.csv
)
FILE_FORMAT = (TYPE = 'CSV' SKIP_HEADER = 1);

-- Incremental loading with metadata
COPY INTO my_table
FROM @my_s3_stage/
FILE_FORMAT = (TYPE = 'CSV' SKIP_HEADER = 1)
FORCE = FALSE;  -- Skip files already loaded
```

### Copy Into (Unloading to S3)

```sql
-- Basic unload to S3
COPY INTO @my_s3_stage/export/data.csv
FROM (SELECT * FROM my_table)
FILE_FORMAT = (TYPE = 'CSV' HEADER = TRUE);

-- Unload with partitioning
COPY INTO @my_s3_stage/export/
FROM (SELECT * FROM sales WHERE date >= '2023-01-01')
PARTITION BY ('year=' || YEAR(date) || '/month=' || MONTH(date))
FILE_FORMAT = (TYPE = 'PARQUET')
HEADER = TRUE;

-- Unload compressed
COPY INTO @my_s3_stage/export/data.csv.gz
FROM (SELECT * FROM my_table)
FILE_FORMAT = (TYPE = 'CSV' COMPRESSION = 'GZIP' HEADER = TRUE);

-- Unload to multiple files (parallel)
COPY INTO @my_s3_stage/export/data_
FROM (SELECT * FROM large_table)
FILE_FORMAT = (TYPE = 'CSV' HEADER = TRUE)
MAX_FILE_SIZE = 100000000;  -- 100MB per file
```

### File Formats

```sql
-- Create CSV file format
CREATE FILE FORMAT csv_format
TYPE = 'CSV'
FIELD_DELIMITER = ','
RECORD_DELIMITER = '\n'
SKIP_HEADER = 1
DATE_FORMAT = 'YYYY-MM-DD'
TIMESTAMP_FORMAT = 'YYYY-MM-DD HH24:MI:SS'
FIELD_OPTIONALLY_ENCLOSED_BY = '"'
ESCAPE_UNENCLOSED_FIELD = '\\'
COMPRESSION = 'AUTO'
ERROR_ON_COLUMN_COUNT_MISMATCH = TRUE;

-- Create JSON file format
CREATE FILE FORMAT json_format
TYPE = 'JSON'
COMPRESSION = 'AUTO'
DATE_FORMAT = 'AUTO'
TIMESTAMP_FORMAT = 'AUTO';

-- Create Parquet file format
CREATE FILE FORMAT parquet_format
TYPE = 'PARQUET'
COMPRESSION = 'AUTO';

-- Use named file format
COPY INTO my_table
FROM @my_s3_stage/data.csv
FILE_FORMAT = (FORMAT_NAME = 'csv_format');
```

### S3 External Tables

```sql
-- Create external table pointing to S3
CREATE EXTERNAL TABLE s3_external_table (
    id INTEGER AS (value:id::INTEGER),
    name STRING AS (value:name::STRING),
    date DATE AS (TO_DATE(value:date::STRING, 'YYYY-MM-DD'))
)
LOCATION = @my_s3_stage/
FILE_FORMAT = json_format
AUTO_REFRESH = TRUE;

-- Query external table
SELECT * FROM s3_external_table WHERE date >= '2023-01-01';

-- Refresh external table metadata
ALTER EXTERNAL TABLE s3_external_table REFRESH;
```

### Monitoring S3 Operations

```sql
-- Check copy history
SELECT *
FROM TABLE(INFORMATION_SCHEMA.COPY_HISTORY(
    TABLE_NAME=>'MY_TABLE',
    START_TIME=> DATEADD(hours, -1, CURRENT_TIMESTAMP())
));

-- Check load errors
SELECT *
FROM TABLE(VALIDATE(MY_TABLE, JOB_ID=>'01234567-89ab-cdef-0123-456789abcdef'));

-- Monitor data loading jobs
SELECT
    query_id,
    query_text,
    start_time,
    end_time,
    rows_loaded,
    error_count
FROM TABLE(INFORMATION_SCHEMA.COPY_HISTORY(
    TABLE_NAME=>'MY_TABLE'
))
ORDER BY start_time DESC;
```

### S3 Best Practices

```sql
-- Use storage integrations instead of hardcoded credentials
-- Partition large datasets when unloading
-- Use appropriate file formats (Parquet for analytics, JSON for flexibility)
-- Enable auto-refresh for external tables when possible
-- Monitor file sizes (aim for 100-250MB compressed files)
-- Use FORCE=FALSE for incremental loading
-- Implement error handling and monitoring

-- Example: Robust S3 loading procedure
CREATE OR REPLACE PROCEDURE load_from_s3(stage_path STRING, table_name STRING)
RETURNS STRING
LANGUAGE SQL
AS
$
DECLARE
    result STRING;
    error_count INTEGER;
BEGIN
    -- Load data
    EXECUTE IMMEDIATE 'COPY INTO ' || table_name || ' FROM @my_s3_stage/' || stage_path ||
                     ' FILE_FORMAT = (TYPE = ''CSV'' SKIP_HEADER = 1) ON_ERROR = ''CONTINUE''';

    -- Check for errors
    SELECT COUNT(*) INTO error_count
    FROM TABLE(VALIDATE(IDENTIFIER(table_name), JOB_ID=>LAST_QUERY_ID()));

    IF (error_count > 0) THEN
        result := 'Load completed with ' || error_count || ' errors. Check VALIDATE() for details.';
    ELSE
        result := 'Load completed successfully.';
    END IF;

    RETURN result;
END;
$;
```

## Stored Procedures

```sql
-- Basic stored procedure
CREATE OR REPLACE PROCEDURE update_salaries(increase_percent FLOAT)
RETURNS STRING
LANGUAGE SQL
AS
$$
BEGIN
    UPDATE employees
    SET salary = salary * (1 + increase_percent / 100);

    RETURN 'Salaries updated successfully';
END;
$$;

-- Call procedure
CALL update_salaries(5.0);

-- JavaScript stored procedure
CREATE OR REPLACE PROCEDURE process_data(table_name STRING)
RETURNS STRING
LANGUAGE JAVASCRIPT
AS
$$
    var sql_command = "SELECT COUNT(*) FROM " + TABLE_NAME;
    var statement = snowflake.createStatement({sqlText: sql_command});
    var result = statement.execute();
    result.next();
    return "Row count: " + result.getColumnValue(1);
$$;
```

## Performance & Optimization

### Clustering

```sql
-- Create clustered table
CREATE TABLE sales (
    date DATE,
    product_id INTEGER,
    amount DECIMAL(10,2)
) CLUSTER BY (date);

-- Add clustering to existing table
ALTER TABLE sales CLUSTER BY (date, product_id);

-- Show clustering information
SELECT SYSTEM$CLUSTERING_INFORMATION('sales');
```

### Query Optimization

```sql
-- Use LIMIT for testing
SELECT * FROM large_table LIMIT 1000;

-- Use specific columns instead of *
SELECT id, name, salary FROM employees;

-- Use WHERE clauses to filter early
SELECT * FROM sales
WHERE date >= '2023-01-01'
  AND product_category = 'Electronics';

-- Use appropriate data types
-- Use NUMBER instead of VARCHAR for numeric data
-- Use DATE instead of STRING for dates
```

## Useful System Functions

```sql
-- Current user and role
SELECT CURRENT_USER();
SELECT CURRENT_ROLE();

-- Session information
SELECT CURRENT_SESSION();
SELECT CURRENT_WAREHOUSE();
SELECT CURRENT_DATABASE();
SELECT CURRENT_SCHEMA();

-- Query history
SELECT * FROM TABLE(INFORMATION_SCHEMA.QUERY_HISTORY())
WHERE start_time >= DATEADD(hour, -1, CURRENT_TIMESTAMP());

-- Account usage
SELECT * FROM SNOWFLAKE.ACCOUNT_USAGE.WAREHOUSE_METERING_HISTORY
WHERE start_time >= DATEADD(day, -7, CURRENT_TIMESTAMP());
```

## Common Patterns

### Deduplication

```sql
-- Remove duplicates keeping latest
DELETE FROM employees
WHERE id IN (
    SELECT id FROM (
        SELECT id,
               ROW_NUMBER() OVER (PARTITION BY email ORDER BY updated_at DESC) as rn
        FROM employees
    ) WHERE rn > 1
);
```

### Pivot/Unpivot

```sql
-- Pivot
SELECT *
FROM (
    SELECT year, quarter, sales
    FROM quarterly_sales
) PIVOT (
    SUM(sales) FOR quarter IN ('Q1', 'Q2', 'Q3', 'Q4')
);

-- Unpivot
SELECT *
FROM quarterly_sales_pivoted
UNPIVOT (
    sales FOR quarter IN (Q1, Q2, Q3, Q4)
);
```

### Upsert (Merge)

```sql
MERGE INTO target_table t
USING source_table s ON t.id = s.id
WHEN MATCHED THEN
    UPDATE SET t.name = s.name, t.updated_at = CURRENT_TIMESTAMP()
WHEN NOT MATCHED THEN
    INSERT (id, name, created_at) VALUES (s.id, s.name, CURRENT_TIMESTAMP());
```

## Error Handling

```sql
-- Try/Catch in stored procedures
CREATE OR REPLACE PROCEDURE safe_update()
RETURNS STRING
LANGUAGE SQL
AS
$$
BEGIN
    BEGIN
        UPDATE employees SET salary = salary * 1.1;
        RETURN 'Success';
    EXCEPTION
        WHEN OTHER THEN
            RETURN 'Error: ' || SQLERRM;
    END;
END;
$$;
```

## Useful Queries for Administration

```sql
-- Show table sizes
SELECT table_name, row_count, bytes
FROM INFORMATION_SCHEMA.TABLES
WHERE table_schema = 'PUBLIC'
ORDER BY bytes DESC;

-- Show column information
SELECT column_name, data_type, is_nullable
FROM INFORMATION_SCHEMA.COLUMNS
WHERE table_name = 'EMPLOYEES';

-- Show constraints
SELECT constraint_name, constraint_type, table_name
FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS
WHERE table_schema = 'PUBLIC';
```
