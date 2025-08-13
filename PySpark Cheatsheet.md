# PySpark Cheatsheet

## 🚀 Setup & Initialization

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

# Create SparkSession
spark = SparkSession.builder \
    .appName("MyApp") \
    .config("spark.sql.adaptive.enabled", "true") \
    .getOrCreate()

# Stop SparkSession
spark.stop()
```

## 📊 Creating DataFrames

```python
# From list of tuples
data = [("Alice", 25), ("Bob", 30)]
df = spark.createDataFrame(data, ["name", "age"])

# From list of dictionaries
data = [{"name": "Alice", "age": 25}, {"name": "Bob", "age": 30}]
df = spark.createDataFrame(data)

# From pandas DataFrame
import pandas as pd
pandas_df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
df = spark.createDataFrame(pandas_df)

# Read from files
df = spark.read.csv("file.csv", header=True, inferSchema=True)
df = spark.read.json("file.json")
df = spark.read.parquet("file.parquet")
df = spark.read.option("multiline", "true").json("file.json")
```

## 🔍 Basic DataFrame Operations

```python
# Show data
df.show()                    # Show first 20 rows
df.show(5)                   # Show first 5 rows
df.show(truncate=False)      # Show without truncating

# Schema and info
df.printSchema()             # Print schema
df.columns                   # List column names
df.dtypes                    # Column names and types
df.count()                   # Number of rows
df.describe().show()         # Summary statistics

# First few rows
df.head()                    # First row as Row object
df.head(3)                   # First 3 rows as list
df.first()                   # First row
df.take(5)                   # First 5 rows as list
```

## 🎯 Column Selection & Manipulation

```python
# Select columns
df.select("name", "age")
df.select(col("name"), col("age"))
df.select("*")               # All columns
df.select(df.name, df.age)   # Using DataFrame notation

# Add/rename columns
df.withColumn("age_plus_1", col("age") + 1)
df.withColumnRenamed("old_name", "new_name")

# Drop columns
df.drop("column_name")
df.drop("col1", "col2")

# Column operations
df.select(
    col("name"),
    col("age").alias("person_age"),
    (col("age") * 2).alias("double_age"),
    upper(col("name")).alias("upper_name")
)
```

## 🔧 Data Types & Casting

```python
# Cast data types
df.withColumn("age", col("age").cast("string"))
df.withColumn("age", col("age").cast(IntegerType()))

# Define schema explicitly
schema = StructType([
    StructField("name", StringType(), True),
    StructField("age", IntegerType(), True),
    StructField("salary", DoubleType(), True)
])
```

## 🚰 Filtering & Conditional Operations

```python
# Filter rows
df.filter(col("age") > 25)
df.filter("age > 25")
df.where(col("age") > 25)

# Multiple conditions
df.filter((col("age") > 20) & (col("age") < 30))
df.filter((col("name") == "Alice") | (col("age") > 30))

# String operations
df.filter(col("name").startswith("A"))
df.filter(col("name").endswith("e"))
df.filter(col("name").contains("lic"))
df.filter(col("name").like("A%"))
df.filter(col("name").rlike(r"A.*"))

# Null handling
df.filter(col("name").isNull())
df.filter(col("name").isNotNull())
df.filter(~col("name").isNull())  # Alternative

# Case when
df.withColumn("age_group",
    when(col("age") < 18, "Minor")
    .when(col("age") < 65, "Adult")
    .otherwise("Senior")
)
```

## 📈 Aggregations & GroupBy

```python
# Basic aggregations
df.agg(
    count("*").alias("total_count"),
    avg("age").alias("avg_age"),
    max("age").alias("max_age"),
    min("age").alias("min_age"),
    sum("age").alias("sum_age")
).show()

# GroupBy operations
df.groupBy("department").agg(
    count("*").alias("count"),
    avg("salary").alias("avg_salary"),
    max("salary").alias("max_salary")
)

# Multiple grouping columns
df.groupBy("department", "gender").agg(avg("salary"))

# Built-in aggregation functions
df.groupBy("department").count()
df.groupBy("department").avg("salary")
df.groupBy("department").sum("salary")
df.groupBy("department").max("salary")
df.groupBy("department").min("salary")
```

## 🔗 Joins

```python
# Different join types
df1.join(df2, "common_column")                    # Inner join
df1.join(df2, "common_column", "inner")
df1.join(df2, "common_column", "left")            # Left join
df1.join(df2, "common_column", "right")           # Right join
df1.join(df2, "common_column", "outer")           # Full outer join
df1.join(df2, "common_column", "left_semi")       # Left semi join
df1.join(df2, "common_column", "left_anti")       # Left anti join

# Join on multiple columns
df1.join(df2, ["col1", "col2"])

# Join with different column names
df1.join(df2, df1.id == df2.user_id)

# Join with complex conditions
df1.join(df2, (df1.id == df2.user_id) & (df1.date >= df2.start_date))
```

## 🗂️ Sorting & Ordering

```python
# Sort by single column
df.orderBy("age")
df.orderBy(col("age").asc())
df.orderBy(col("age").desc())

# Sort by multiple columns
df.orderBy("department", col("age").desc())
df.orderBy(["department", "age"])

# Alternative syntax
df.sort("age")
df.sort(col("age").desc())
```

## 🎲 Sampling & Distinct

```python
# Remove duplicates
df.distinct()
df.dropDuplicates()
df.dropDuplicates(["name", "age"])  # Based on specific columns

# Sampling
df.sample(0.1)                      # 10% sample
df.sample(0.1, seed=42)             # With seed for reproducibility
df.limit(100)                       # First 100 rows
```

## 🧹 Data Cleaning

```python
# Handle null values
df.na.drop()                        # Drop rows with any null
df.na.drop(how="all")               # Drop rows where all values are null
df.na.drop(subset=["name", "age"])  # Drop based on specific columns

# Fill null values
df.na.fill(0)                       # Fill all nulls with 0
df.na.fill({"age": 0, "name": "Unknown"})  # Different values per column
df.fillna("Unknown", subset=["name"])

# Replace values
df.replace("old_value", "new_value")
df.replace(["val1", "val2"], ["new1", "new2"])
```

## 🔤 String Functions

```python
# String manipulations
df.select(
    upper(col("name")),
    lower(col("name")),
    length(col("name")),
    trim(col("name")),
    ltrim(col("name")),
    rtrim(col("name")),
    substring(col("name"), 1, 3),
    concat(col("first_name"), lit(" "), col("last_name")),
    split(col("name"), " "),
    regexp_replace(col("name"), "pattern", "replacement")
)
```

## 📅 Date & Time Functions

```python
# Date operations
df.select(
    current_date(),
    current_timestamp(),
    year(col("date_col")),
    month(col("date_col")),
    dayofmonth(col("date_col")),
    dayofweek(col("date_col")),
    date_format(col("date_col"), "yyyy-MM-dd"),
    to_date(col("string_date"), "yyyy-MM-dd"),
    datediff(col("end_date"), col("start_date"))
)
```

## 🪟 Window Functions

```python
from pyspark.sql.window import Window

# Define window
window = Window.partitionBy("department").orderBy("salary")

# Window functions
df.select(
    col("*"),
    row_number().over(window).alias("row_num"),
    rank().over(window).alias("rank"),
    dense_rank().over(window).alias("dense_rank"),
    lag(col("salary"), 1).over(window).alias("prev_salary"),
    lead(col("salary"), 1).over(window).alias("next_salary"),
    sum(col("salary")).over(window).alias("running_sum"),
    avg(col("salary")).over(window).alias("running_avg")
)

# Window with range
window_range = Window.partitionBy("department").orderBy("date").rangeBetween(-7, 0)
df.withColumn("weekly_avg", avg("sales").over(window_range))
```

## 💾 Writing Data

```python
# Write to different formats
df.write.mode("overwrite").csv("output.csv", header=True)
df.write.mode("append").json("output.json")
df.write.mode("overwrite").parquet("output.parquet")

# Write modes
df.write.mode("overwrite")   # Overwrite existing data
df.write.mode("append")      # Append to existing data
df.write.mode("ignore")      # Ignore if exists
df.write.mode("error")       # Error if exists (default)

# Partitioned writes
df.write.partitionBy("year", "month").parquet("partitioned_output")

# Write to database
df.write \
  .format("jdbc") \
  .option("url", "jdbc:postgresql://localhost/test") \
  .option("dbtable", "table_name") \
  .option("user", "username") \
  .option("password", "password") \
  .save()
```

## 🏃‍♂️ Performance & Optimization

```python
# Cache DataFrames
df.cache()
df.persist()
df.unpersist()

# Repartition
df.repartition(4)                    # 4 partitions
df.repartition("column")             # Partition by column
df.coalesce(2)                       # Reduce to 2 partitions

# Check partitions
df.rdd.getNumPartitions()

# Broadcast join (for small DataFrames)
from pyspark.sql.functions import broadcast
large_df.join(broadcast(small_df), "key")

# Explain query plan
df.explain()
df.explain(True)  # Extended explanation
```

## 🔧 Configuration & Tuning

```python
# Common Spark configurations
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

# Check current configuration
spark.conf.get("spark.sql.adaptive.enabled")
```

## 📝 UDFs (User Defined Functions)

```python
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

# Define UDF
def categorize_age(age):
    if age < 18:
        return "Minor"
    elif age < 65:
        return "Adult"
    else:
        return "Senior"

# Register UDF
categorize_udf = udf(categorize_age, StringType())

# Use UDF
df.withColumn("age_category", categorize_udf(col("age")))

# Or register for SQL use
spark.udf.register("categorize_age", categorize_age, StringType())
```

## 🔍 Common Patterns

```python
# Count nulls in each column
null_counts = df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns])

# Pivot table
df.groupBy("department").pivot("gender").agg(avg("salary"))

# Union DataFrames
df1.union(df2)
df1.unionByName(df2)  # Union by column names

# Collect results to driver
results = df.collect()  # Be careful with large datasets
results_list = [row.asDict() for row in df.collect()]

# Convert to Pandas (use with caution for large data)
pandas_df = df.toPandas()
```

## ⚡ Quick Tips

- Always use `show()` for debugging, avoid `collect()` on large datasets
- Use `limit()` when exploring data to avoid overwhelming output
- Cache DataFrames that are used multiple times
- Use column functions instead of UDFs when possible for better performance
- Prefer `parquet` format for better performance and compression
- Use `explain()` to understand query execution plans
- Partition data appropriately when writing large datasets
