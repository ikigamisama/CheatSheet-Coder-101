# PostgreSQL Cheatsheet

## Connection & Authentication

```bash
# Connect to PostgreSQL
psql -U username -d database_name
psql -h hostname -p 5432 -U username -d database_name

# Connect as specific user to specific database
psql -U postgres -d mydb

# Connect with password prompt
psql -U username -W

# Exit PostgreSQL
\q
```

## psql Meta-Commands

```sql
-- Database operations
\l                          -- List all databases
\c database_name            -- Connect to database
\conninfo                   -- Show connection info

-- Table operations
\dt                         -- List tables
\dt+                        -- List tables with details
\d table_name               -- Describe table
\d+ table_name              -- Detailed table description

-- Schema operations
\dn                         -- List schemas
\dt schema_name.*           -- List tables in schema

-- User operations
\du                         -- List users/roles
\du+                        -- List users with details

-- Other useful commands
\timing on                  -- Enable query timing
\x                          -- Toggle expanded display
\i filename.sql             -- Execute SQL from file
\o filename.txt             -- Output to file
\h command                  -- Help for SQL command
\?                          -- List all meta-commands
```

## Database Operations

```sql
-- Show all databases
SELECT datname FROM pg_database;

-- Create database
CREATE DATABASE database_name;
CREATE DATABASE database_name WITH ENCODING 'UTF8';

-- Drop database
DROP DATABASE database_name;
DROP DATABASE IF EXISTS database_name;

-- Show current database
SELECT current_database();

-- Database size
SELECT pg_size_pretty(pg_database_size('database_name'));
```

## Schema Operations

```sql
-- Create schema
CREATE SCHEMA schema_name;

-- Drop schema
DROP SCHEMA schema_name;
DROP SCHEMA schema_name CASCADE;  -- Drop with all objects

-- Set search path
SET search_path TO schema_name, public;
```

## Table Operations

```sql
-- Create table
CREATE TABLE table_name (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE,
    age INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create table with constraints
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE,
    age INTEGER CHECK (age >= 0),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Drop table
DROP TABLE table_name;
DROP TABLE IF EXISTS table_name CASCADE;

-- Rename table
ALTER TABLE old_name RENAME TO new_name;

-- Truncate table
TRUNCATE TABLE table_name;
TRUNCATE TABLE table_name RESTART IDENTITY;  -- Reset auto-increment
```

## Column Operations

```sql
-- Add column
ALTER TABLE table_name ADD COLUMN column_name datatype;
ALTER TABLE table_name ADD COLUMN column_name VARCHAR(255) DEFAULT 'default_value';

-- Drop column
ALTER TABLE table_name DROP COLUMN column_name;

-- Rename column
ALTER TABLE table_name RENAME COLUMN old_name TO new_name;

-- Change column type
ALTER TABLE table_name ALTER COLUMN column_name TYPE new_datatype;
ALTER TABLE table_name ALTER COLUMN column_name TYPE VARCHAR(100) USING column_name::VARCHAR(100);

-- Set/drop default
ALTER TABLE table_name ALTER COLUMN column_name SET DEFAULT value;
ALTER TABLE table_name ALTER COLUMN column_name DROP DEFAULT;

-- Set/drop NOT NULL
ALTER TABLE table_name ALTER COLUMN column_name SET NOT NULL;
ALTER TABLE table_name ALTER COLUMN column_name DROP NOT NULL;
```

## Data Types

```sql
-- Numeric Types
SMALLINT, INTEGER, BIGINT
DECIMAL(precision, scale), NUMERIC(precision, scale)
REAL, DOUBLE PRECISION
SERIAL, BIGSERIAL                   -- Auto-incrementing

-- Character Types
CHAR(n)                            -- Fixed length
VARCHAR(n)                         -- Variable length
TEXT                               -- Unlimited length

-- Date/Time Types
DATE                               -- Date only
TIME                               -- Time only
TIMESTAMP                          -- Date and time
TIMESTAMPTZ                        -- Timestamp with timezone
INTERVAL                           -- Time interval

-- Boolean
BOOLEAN                            -- TRUE, FALSE, NULL

-- Binary Data
BYTEA                              -- Binary data

-- JSON Types
JSON                               -- JSON data
JSONB                              -- Binary JSON (faster)

-- Arrays
INTEGER[]                          -- Array of integers
TEXT[]                             -- Array of text

-- UUID
UUID                               -- Universally unique identifier

-- Network Types
INET                               -- IP address
CIDR                               -- Network address
MACADDR                            -- MAC address
```

## CRUD Operations

### INSERT

```sql
-- Insert single record
INSERT INTO table_name (column1, column2) VALUES (value1, value2);

-- Insert multiple records
INSERT INTO table_name (column1, column2) VALUES
    (value1, value2),
    (value3, value4);

-- Insert and return
INSERT INTO table_name (column1, column2) VALUES (value1, value2) RETURNING id;

-- Insert from SELECT
INSERT INTO table_name (column1, column2)
SELECT column1, column2 FROM other_table WHERE condition;

-- Upsert (INSERT ... ON CONFLICT)
INSERT INTO table_name (id, name) VALUES (1, 'John')
ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name;

INSERT INTO table_name (email, name) VALUES ('john@example.com', 'John')
ON CONFLICT (email) DO NOTHING;
```

### SELECT

```sql
-- Basic select
SELECT * FROM table_name;
SELECT column1, column2 FROM table_name;

-- With conditions
SELECT * FROM table_name WHERE condition;

-- Case-sensitive/insensitive matching
SELECT * FROM table_name WHERE column LIKE 'pattern%';
SELECT * FROM table_name WHERE column ILIKE 'pattern%';  -- Case insensitive

-- Regular expressions
SELECT * FROM table_name WHERE column ~ 'pattern';       -- Case sensitive
SELECT * FROM table_name WHERE column ~* 'pattern';      -- Case insensitive

-- Sorting
SELECT * FROM table_name ORDER BY column1 ASC, column2 DESC;

-- Limiting results
SELECT * FROM table_name LIMIT 10;
SELECT * FROM table_name LIMIT 10 OFFSET 20;

-- Window functions
SELECT column1, ROW_NUMBER() OVER (ORDER BY column1) FROM table_name;
SELECT column1, RANK() OVER (PARTITION BY column2 ORDER BY column1) FROM table_name;

-- Common Table Expressions (CTE)
WITH cte_name AS (
    SELECT column1, column2 FROM table_name WHERE condition
)
SELECT * FROM cte_name;
```

### UPDATE

```sql
-- Update records
UPDATE table_name SET column1 = value1 WHERE condition;

-- Update with RETURNING
UPDATE table_name SET column1 = value1 WHERE condition RETURNING *;

-- Update from another table
UPDATE table1 SET column1 = table2.column1
FROM table2
WHERE table1.id = table2.foreign_id;
```

### DELETE

```sql
-- Delete records
DELETE FROM table_name WHERE condition;

-- Delete with RETURNING
DELETE FROM table_name WHERE condition RETURNING *;

-- Delete using another table
DELETE FROM table1
USING table2
WHERE table1.id = table2.foreign_id AND condition;
```

## Joins

```sql
-- INNER JOIN
SELECT * FROM table1 t1
INNER JOIN table2 t2 ON t1.id = t2.foreign_id;

-- LEFT JOIN
SELECT * FROM table1 t1
LEFT JOIN table2 t2 ON t1.id = t2.foreign_id;

-- RIGHT JOIN
SELECT * FROM table1 t1
RIGHT JOIN table2 t2 ON t1.id = t2.foreign_id;

-- FULL OUTER JOIN
SELECT * FROM table1 t1
FULL OUTER JOIN table2 t2 ON t1.id = t2.foreign_id;

-- CROSS JOIN
SELECT * FROM table1 CROSS JOIN table2;
```

## Functions

### String Functions

```sql
CONCAT(str1, str2)              -- Concatenate strings
LENGTH(str)                     -- String length
UPPER(str), LOWER(str)          -- Case conversion
TRIM(str)                       -- Remove spaces
SUBSTRING(str FROM start FOR len) -- Extract substring
REPLACE(str, old, new)          -- Replace text
SPLIT_PART(str, delimiter, n)   -- Split string
POSITION(substring IN string)   -- Find position
```

### Date/Time Functions

```sql
NOW()                           -- Current timestamp
CURRENT_DATE                    -- Current date
CURRENT_TIME                    -- Current time
EXTRACT(field FROM timestamp)   -- Extract part (YEAR, MONTH, DAY, etc.)
DATE_TRUNC('field', timestamp)  -- Truncate to field
AGE(timestamp1, timestamp2)     -- Calculate age/difference
TO_CHAR(timestamp, format)      -- Format timestamp
TO_DATE(string, format)         -- Parse date from string
```

### Aggregate Functions

```sql
COUNT(*)                        -- Count rows
COUNT(column)                   -- Count non-null values
SUM(column)                     -- Sum values
AVG(column)                     -- Average
MIN(column), MAX(column)        -- Min/Max values
STRING_AGG(column, delimiter)   -- Concatenate with delimiter
ARRAY_AGG(column)               -- Aggregate into array
```

### Mathematical Functions

```sql
ABS(number)                     -- Absolute value
CEIL(number), CEILING(number)   -- Ceiling
FLOOR(number)                   -- Floor
ROUND(number, decimals)         -- Round
RANDOM()                        -- Random number 0-1
GREATEST(val1, val2, ...)       -- Maximum value
LEAST(val1, val2, ...)          -- Minimum value
```

## Indexes

```sql
-- Create index
CREATE INDEX index_name ON table_name (column1, column2);
CREATE UNIQUE INDEX index_name ON table_name (column1);

-- Partial index
CREATE INDEX index_name ON table_name (column1) WHERE condition;

-- Expression index
CREATE INDEX index_name ON table_name (LOWER(column1));

-- GIN index (for arrays, JSON)
CREATE INDEX index_name ON table_name USING GIN (json_column);

-- Show indexes
\di                             -- List indexes
SELECT * FROM pg_indexes WHERE tablename = 'table_name';

-- Drop index
DROP INDEX index_name;
```

## Views

```sql
-- Create view
CREATE VIEW view_name AS
SELECT column1, column2 FROM table_name WHERE condition;

-- Materialized view
CREATE MATERIALIZED VIEW mv_name AS
SELECT column1, column2 FROM table_name WHERE condition;

-- Refresh materialized view
REFRESH MATERIALIZED VIEW mv_name;

-- Drop view
DROP VIEW view_name;
DROP MATERIALIZED VIEW mv_name;
```

## Functions & Procedures

```sql
-- Create function
CREATE OR REPLACE FUNCTION function_name(param1 INTEGER, param2 TEXT)
RETURNS INTEGER AS $$
BEGIN
    -- function body
    RETURN param1 + LENGTH(param2);
END;
$$ LANGUAGE plpgsql;

-- Create procedure (PostgreSQL 11+)
CREATE OR REPLACE PROCEDURE procedure_name(param1 INTEGER)
LANGUAGE plpgsql AS $$
BEGIN
    -- procedure body
    UPDATE table_name SET column1 = param1;
    COMMIT;
END;
$$;

-- Call function
SELECT function_name(10, 'hello');

-- Call procedure
CALL procedure_name(5);

-- Drop function/procedure
DROP FUNCTION function_name(INTEGER, TEXT);
DROP PROCEDURE procedure_name(INTEGER);
```

## Triggers

```sql
-- Create trigger function
CREATE OR REPLACE FUNCTION trigger_function()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger
CREATE TRIGGER trigger_name
    BEFORE UPDATE ON table_name
    FOR EACH ROW
    EXECUTE FUNCTION trigger_function();

-- Drop trigger
DROP TRIGGER trigger_name ON table_name;
```

## Transactions

```sql
-- Start transaction
BEGIN;
START TRANSACTION;

-- Commit transaction
COMMIT;

-- Rollback transaction
ROLLBACK;

-- Savepoints
SAVEPOINT savepoint_name;
ROLLBACK TO savepoint_name;
RELEASE SAVEPOINT savepoint_name;
```

## User Management & Roles

```sql
-- Create role/user
CREATE ROLE role_name;
CREATE USER username WITH PASSWORD 'password';

-- Create role with options
CREATE ROLE role_name WITH LOGIN PASSWORD 'password' CREATEDB;

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE database_name TO username;
GRANT SELECT, INSERT ON table_name TO username;
GRANT USAGE ON SCHEMA schema_name TO username;

-- Grant role to user
GRANT role_name TO username;

-- Show roles
\du
SELECT rolname FROM pg_roles;

-- Drop role
DROP ROLE role_name;
```

## Backup & Restore

```bash
# Backup database
pg_dump -U username -h hostname database_name > backup.sql
pg_dump -U username -h hostname -Fc database_name > backup.dump  # Custom format

# Backup all databases
pg_dumpall -U username -h hostname > all_databases.sql

# Restore database
psql -U username -h hostname database_name < backup.sql
pg_restore -U username -h hostname -d database_name backup.dump

# Restore with options
pg_restore -U username -h hostname -d database_name -c -v backup.dump
```

## Array Operations

```sql
-- Create array column
CREATE TABLE test_table (
    id SERIAL,
    tags TEXT[]
);

-- Insert array data
INSERT INTO test_table (tags) VALUES (ARRAY['tag1', 'tag2', 'tag3']);
INSERT INTO test_table (tags) VALUES ('{"tag1", "tag2", "tag3"}');

-- Query arrays
SELECT * FROM test_table WHERE 'tag1' = ANY(tags);
SELECT * FROM test_table WHERE tags @> ARRAY['tag1'];        -- Contains
SELECT * FROM test_table WHERE tags && ARRAY['tag1', 'tag2']; -- Overlaps

-- Array functions
SELECT array_length(tags, 1) FROM test_table;               -- Array length
SELECT unnest(tags) FROM test_table;                        -- Expand array to rows
SELECT array_agg(column_name) FROM table_name;              -- Aggregate to array
```

## JSON/JSONB Operations

```sql
-- Create JSON column
CREATE TABLE test_table (
    id SERIAL,
    data JSONB
);

-- Insert JSON data
INSERT INTO test_table (data) VALUES ('{"name": "John", "age": 30}');

-- Query JSON
SELECT data->>'name' FROM test_table;                       -- Get as text
SELECT data->'age' FROM test_table;                         -- Get as JSON
SELECT * FROM test_table WHERE data->>'name' = 'John';
SELECT * FROM test_table WHERE data @> '{"age": 30}';       -- Contains

-- JSON path queries
SELECT * FROM test_table WHERE data #> '{address,city}' = '"New York"';

-- JSON functions
SELECT jsonb_object_keys(data) FROM test_table;             -- Get keys
SELECT jsonb_each(data) FROM test_table;                    -- Get key-value pairs
SELECT jsonb_set(data, '{age}', '31') FROM test_table;      -- Update value
```

## System Information

```sql
-- PostgreSQL version
SELECT version();

-- Current user
SELECT current_user;

-- Current database
SELECT current_database();

-- Show active connections
SELECT * FROM pg_stat_activity;

-- Database size
SELECT pg_size_pretty(pg_database_size(current_database()));

-- Table sizes
SELECT
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- Show configuration
SHOW ALL;
SHOW config_parameter;

-- Show installed extensions
SELECT * FROM pg_available_extensions;
```

## Common WHERE Clauses

```sql
-- Comparison operators
WHERE column = value
WHERE column != value (or <> value)
WHERE column > value

-- Pattern matching
WHERE column LIKE 'pattern%'       -- Case sensitive
WHERE column ILIKE 'pattern%'      -- Case insensitive
WHERE column ~ 'regex_pattern'     -- Regular expression
WHERE column ~* 'regex_pattern'    -- Case insensitive regex

-- Array operations
WHERE value = ANY(array_column)
WHERE array_column @> ARRAY[value]

-- JSON operations
WHERE json_column->>'key' = 'value'
WHERE json_column @> '{"key": "value"}'

-- Range operations
WHERE column BETWEEN value1 AND value2
WHERE column IN (value1, value2, value3)

-- NULL checks
WHERE column IS NULL
WHERE column IS NOT NULL
WHERE column IS DISTINCT FROM value  -- NULL-safe comparison
```

## Extensions

```sql
-- Enable extension
CREATE EXTENSION extension_name;

-- Common extensions
CREATE EXTENSION uuid-ossp;        -- UUID generation
CREATE EXTENSION pg_trgm;          -- Trigram matching
CREATE EXTENSION postgis;          -- Geographic objects
CREATE EXTENSION hstore;           -- Key-value pairs
CREATE EXTENSION pg_stat_statements; -- Query statistics

-- Use UUID
SELECT uuid_generate_v4();

-- Trigram similarity
SELECT similarity('hello', 'helo');
```
