# MySQL Cheatsheet

## Connection & Authentication

```sql
-- Connect to MySQL
mysql -u username -p
mysql -u username -p database_name
mysql -h hostname -u username -p database_name

-- Connect with specific port
mysql -h hostname -P 3306 -u username -p

-- Exit MySQL
EXIT; or QUIT; or \q
```

## Database Operations

```sql
-- Show all databases
SHOW DATABASES;

-- Create database
CREATE DATABASE database_name;
CREATE DATABASE IF NOT EXISTS database_name;

-- Use/Select database
USE database_name;

-- Drop database
DROP DATABASE database_name;
DROP DATABASE IF EXISTS database_name;

-- Show current database
SELECT DATABASE();
```

## Table Operations

```sql
-- Show all tables
SHOW TABLES;

-- Create table
CREATE TABLE table_name (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE,
    age INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Show table structure
DESCRIBE table_name;
SHOW COLUMNS FROM table_name;

-- Show create table statement
SHOW CREATE TABLE table_name;

-- Drop table
DROP TABLE table_name;
DROP TABLE IF EXISTS table_name;

-- Rename table
RENAME TABLE old_name TO new_name;
ALTER TABLE old_name RENAME TO new_name;

-- Truncate table (delete all data)
TRUNCATE TABLE table_name;
```

## Column Operations

```sql
-- Add column
ALTER TABLE table_name ADD COLUMN column_name datatype;
ALTER TABLE table_name ADD column_name VARCHAR(255) AFTER existing_column;

-- Modify column
ALTER TABLE table_name MODIFY COLUMN column_name new_datatype;
ALTER TABLE table_name CHANGE old_column_name new_column_name datatype;

-- Drop column
ALTER TABLE table_name DROP COLUMN column_name;

-- Add primary key
ALTER TABLE table_name ADD PRIMARY KEY (column_name);

-- Drop primary key
ALTER TABLE table_name DROP PRIMARY KEY;
```

## Data Types

```sql
-- Numeric
TINYINT, SMALLINT, MEDIUMINT, INT, BIGINT
DECIMAL(precision, scale), NUMERIC(precision, scale)
FLOAT, DOUBLE

-- String
CHAR(length)        -- Fixed length
VARCHAR(length)     -- Variable length
TEXT, MEDIUMTEXT, LONGTEXT
BINARY, VARBINARY
BLOB, MEDIUMBLOB, LONGBLOB

-- Date/Time
DATE                -- YYYY-MM-DD
TIME                -- HH:MM:SS
DATETIME            -- YYYY-MM-DD HH:MM:SS
TIMESTAMP           -- YYYY-MM-DD HH:MM:SS
YEAR                -- YYYY

-- JSON (MySQL 5.7+)
JSON
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

-- Insert with SELECT
INSERT INTO table_name (column1, column2)
SELECT column1, column2 FROM other_table WHERE condition;

-- Insert or update (upsert)
INSERT INTO table_name (id, name) VALUES (1, 'John')
ON DUPLICATE KEY UPDATE name = VALUES(name);
```

### SELECT

```sql
-- Basic select
SELECT * FROM table_name;
SELECT column1, column2 FROM table_name;

-- With conditions
SELECT * FROM table_name WHERE condition;
SELECT * FROM table_name WHERE column1 = 'value' AND column2 > 10;

-- Sorting
SELECT * FROM table_name ORDER BY column1 ASC, column2 DESC;

-- Limiting results
SELECT * FROM table_name LIMIT 10;
SELECT * FROM table_name LIMIT 10 OFFSET 20;

-- Grouping
SELECT column1, COUNT(*) FROM table_name GROUP BY column1;
SELECT column1, COUNT(*) FROM table_name GROUP BY column1 HAVING COUNT(*) > 5;

-- Distinct values
SELECT DISTINCT column1 FROM table_name;
```

### UPDATE

```sql
-- Update records
UPDATE table_name SET column1 = value1 WHERE condition;
UPDATE table_name SET column1 = value1, column2 = value2 WHERE condition;

-- Update with JOIN
UPDATE table1 t1
JOIN table2 t2 ON t1.id = t2.id
SET t1.column1 = t2.column1
WHERE condition;
```

### DELETE

```sql
-- Delete records
DELETE FROM table_name WHERE condition;

-- Delete all records (keep structure)
DELETE FROM table_name;

-- Delete with JOIN
DELETE t1 FROM table1 t1
JOIN table2 t2 ON t1.id = t2.id
WHERE condition;
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

-- FULL OUTER JOIN (not directly supported, use UNION)
SELECT * FROM table1 t1 LEFT JOIN table2 t2 ON t1.id = t2.foreign_id
UNION
SELECT * FROM table1 t1 RIGHT JOIN table2 t2 ON t1.id = t2.foreign_id;

-- CROSS JOIN
SELECT * FROM table1 CROSS JOIN table2;
```

## Functions

### String Functions

```sql
CONCAT(str1, str2)          -- Concatenate strings
LENGTH(str)                 -- String length
UPPER(str), LOWER(str)      -- Case conversion
TRIM(str)                   -- Remove spaces
SUBSTRING(str, start, len)  -- Extract substring
REPLACE(str, old, new)      -- Replace text
```

### Date Functions

```sql
NOW()                       -- Current datetime
CURDATE()                   -- Current date
CURTIME()                   -- Current time
DATE_FORMAT(date, format)   -- Format date
DATEDIFF(date1, date2)      -- Difference in days
DATE_ADD(date, INTERVAL 1 DAY)  -- Add interval
YEAR(date), MONTH(date), DAY(date)  -- Extract parts
```

### Aggregate Functions

```sql
COUNT(*)                    -- Count rows
COUNT(column)               -- Count non-null values
SUM(column)                 -- Sum values
AVG(column)                 -- Average
MIN(column), MAX(column)    -- Min/Max values
GROUP_CONCAT(column)        -- Concatenate group values
```

## Indexes

```sql
-- Create index
CREATE INDEX index_name ON table_name (column1, column2);
CREATE UNIQUE INDEX index_name ON table_name (column1);

-- Show indexes
SHOW INDEX FROM table_name;

-- Drop index
DROP INDEX index_name ON table_name;
ALTER TABLE table_name DROP INDEX index_name;
```

## Views

```sql
-- Create view
CREATE VIEW view_name AS
SELECT column1, column2 FROM table_name WHERE condition;

-- Show views
SHOW FULL TABLES WHERE Table_type = 'VIEW';

-- Drop view
DROP VIEW view_name;
```

## Stored Procedures & Functions

```sql
-- Create stored procedure
DELIMITER //
CREATE PROCEDURE procedure_name(IN param1 INT, OUT param2 VARCHAR(255))
BEGIN
    -- procedure body
    SELECT column1 INTO param2 FROM table_name WHERE id = param1;
END //
DELIMITER ;

-- Call procedure
CALL procedure_name(1, @result);
SELECT @result;

-- Create function
DELIMITER //
CREATE FUNCTION function_name(param1 INT) RETURNS VARCHAR(255)
READS SQL DATA
BEGIN
    DECLARE result VARCHAR(255);
    SELECT column1 INTO result FROM table_name WHERE id = param1;
    RETURN result;
END //
DELIMITER ;

-- Drop procedure/function
DROP PROCEDURE procedure_name;
DROP FUNCTION function_name;
```

## Transactions

```sql
-- Start transaction
START TRANSACTION;
BEGIN;

-- Commit transaction
COMMIT;

-- Rollback transaction
ROLLBACK;

-- Savepoints
SAVEPOINT savepoint_name;
ROLLBACK TO savepoint_name;
```

## User Management

```sql
-- Create user
CREATE USER 'username'@'localhost' IDENTIFIED BY 'password';
CREATE USER 'username'@'%' IDENTIFIED BY 'password';

-- Grant privileges
GRANT ALL PRIVILEGES ON database_name.* TO 'username'@'localhost';
GRANT SELECT, INSERT ON table_name TO 'username'@'localhost';

-- Show grants
SHOW GRANTS FOR 'username'@'localhost';

-- Revoke privileges
REVOKE ALL PRIVILEGES ON database_name.* FROM 'username'@'localhost';

-- Drop user
DROP USER 'username'@'localhost';

-- Change password
ALTER USER 'username'@'localhost' IDENTIFIED BY 'new_password';
```

## Backup & Restore

```bash
# Export database
mysqldump -u username -p database_name > backup.sql
mysqldump -u username -p --all-databases > all_databases.sql

# Import database
mysql -u username -p database_name < backup.sql
mysql -u username -p < all_databases.sql
```

## System Information

```sql
-- Show MySQL version
SELECT VERSION();

-- Show current user
SELECT USER();

-- Show processlist
SHOW PROCESSLIST;

-- Show variables
SHOW VARIABLES LIKE 'variable_name%';

-- Show status
SHOW STATUS LIKE 'status_name%';

-- Show engines
SHOW ENGINES;
```

## Common WHERE Clauses

```sql
-- Comparison operators
WHERE column = value
WHERE column != value (or <> value)
WHERE column > value
WHERE column >= value
WHERE column < value
WHERE column <= value

-- Pattern matching
WHERE column LIKE 'pattern%'      -- Starts with
WHERE column LIKE '%pattern'      -- Ends with
WHERE column LIKE '%pattern%'     -- Contains
WHERE column REGEXP 'pattern'     -- Regular expression

-- Range and lists
WHERE column BETWEEN value1 AND value2
WHERE column IN (value1, value2, value3)
WHERE column NOT IN (value1, value2)

-- NULL checks
WHERE column IS NULL
WHERE column IS NOT NULL

-- Logical operators
WHERE condition1 AND condition2
WHERE condition1 OR condition2
WHERE NOT condition
```

## JSON Functions (MySQL 5.7+)

```sql
-- JSON creation
JSON_OBJECT('key', 'value')
JSON_ARRAY(value1, value2)

-- JSON extraction
JSON_EXTRACT(json_column, '$.key')
json_column->'$.key'              -- Shorthand
json_column->>'$.key'             -- Unquoted

-- JSON modification
JSON_SET(json_column, '$.key', 'new_value')
JSON_INSERT(json_column, '$.new_key', 'value')
JSON_REPLACE(json_column, '$.key', 'new_value')
JSON_REMOVE(json_column, '$.key')
```
