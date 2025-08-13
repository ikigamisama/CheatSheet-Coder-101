# Comprehensive Polars Cheat Sheet

## Import and Setup

```python
import polars as pl
import numpy as np

# Display options
pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_cols(20)
pl.Config.set_tbl_width_chars(120)
```

## Creating DataFrames and Series

### From scratch

```python
# DataFrame from dictionary
df = pl.DataFrame({
    'A': [1, 2, 3],
    'B': ['x', 'y', 'z'],
    'C': [1.1, 2.2, 3.3]
})

# Series
s = pl.Series('name', [1, 2, 3, 4])

# From lists/arrays
df = pl.DataFrame({
    'col1': [1, 2, 3],
    'col2': ['a', 'b', 'c']
})

# Empty DataFrame with schema
df = pl.DataFrame(schema={'col1': pl.Int64, 'col2': pl.Utf8})
```

### Reading Data

```python
# CSV files
df = pl.read_csv('file.csv')
df = pl.read_csv('file.csv', separator=';', encoding='utf-8')

# Parquet files (recommended for Polars)
df = pl.read_parquet('file.parquet')

# JSON files
df = pl.read_json('file.json')
df = pl.read_ndjson('file.ndjson')  # Newline-delimited JSON

# Excel files
df = pl.read_excel('file.xlsx', sheet_name='Sheet1')

# Lazy reading (for large files)
lazy_df = pl.scan_csv('file.csv')
lazy_df = pl.scan_parquet('file.parquet')
```

## Basic Information and Inspection

```python
# Basic info
df.shape                # (rows, columns)
df.describe()           # Statistical summary
df.dtypes               # Data types
df.columns              # Column names
df.schema               # Column names with types
df.width                # Number of columns
df.height               # Number of rows

# First/last rows
df.head(n)              # First n rows (default 5)
df.tail(n)              # Last n rows
df.sample(n)            # Random n rows
df.glimpse()            # Compact overview

# Memory and performance
df.estimated_size()     # Estimated memory usage
```

## Data Types in Polars

```python
# Common data types
pl.Int64, pl.Int32, pl.Int16, pl.Int8
pl.UInt64, pl.UInt32, pl.UInt16, pl.UInt8
pl.Float64, pl.Float32
pl.Boolean
pl.Utf8                 # String type
pl.Date, pl.Datetime, pl.Time, pl.Duration
pl.List(pl.Int64)       # List type
pl.Struct               # Struct type
pl.Categorical          # Categorical type
```

## Column Selection and Expressions

### Basic selection

```python
df.select('column')                    # Single column
df.select(['col1', 'col2'])           # Multiple columns
df.select(pl.col('column'))           # Using expression
df.select(pl.all())                   # All columns

# Column patterns
df.select(pl.col('^col.*'))           # Regex pattern
df.select(pl.col('*').exclude('col1')) # All except col1
```

### Expression syntax (the Polars way)

```python
# Expressions are the core of Polars
df.select(
    pl.col('A').sum().alias('A_sum'),
    pl.col('B').mean().alias('B_mean'),
    (pl.col('C') * 2).alias('C_doubled')
)

# Chain operations
df.select(
    pl.col('column')
    .str.to_uppercase()
    .str.replace('old', 'new')
    .alias('processed_column')
)
```

## Filtering and Boolean Indexing

```python
# Basic filtering
df.filter(pl.col('column') > 5)
df.filter(pl.col('col').is_in(['A', 'B']))
df.filter(pl.col('col').is_null())
df.filter(pl.col('col').is_not_null())

# Multiple conditions
df.filter(
    (pl.col('col1') > 5) &
    (pl.col('col2') < 10)
)

df.filter(
    (pl.col('col1') > 5) |
    (pl.col('col2').str.contains('pattern'))
)

# Using query-like syntax
df.filter(pl.col('A').is_between(1, 10))
```

## Data Manipulation

### Adding and modifying columns

```python
# Add new columns
df.with_columns([
    pl.col('A').sum().alias('A_sum'),
    (pl.col('B') * 2).alias('B_doubled'),
    pl.lit('constant').alias('constant_col')
])

# Modify existing columns
df.with_columns(
    pl.col('column').cast(pl.Int64),
    pl.col('text_col').str.to_lowercase()
)

# Conditional logic
df.with_columns(
    pl.when(pl.col('A') > 5)
    .then(pl.lit('high'))
    .otherwise(pl.lit('low'))
    .alias('category')
)
```

### Removing columns

```python
df.drop('column')
df.drop(['col1', 'col2'])
df.select(pl.all().exclude('unwanted_col'))
```

### Renaming

```python
df.rename({'old_name': 'new_name'})
df.with_columns(pl.col('old_name').alias('new_name')).drop('old_name')
```

### Sorting

```python
df.sort('column')
df.sort(['col1', 'col2'], descending=[False, True])
df.sort(pl.col('column').desc())
```

## Data Cleaning

### Missing data

```python
# Check for missing values
df.null_count()
df.select(pl.all().is_null().sum())

# Handle missing values
df.drop_nulls()                        # Drop rows with any null
df.drop_nulls(subset=['col'])          # Drop where specific column is null
df.fill_null(0)                        # Fill nulls with 0
df.fill_null(strategy='forward')       # Forward fill
df.fill_null(strategy='backward')      # Backward fill
df.fill_null(pl.col('column').mean())  # Fill with mean

# Interpolate
df.with_columns(pl.col('column').interpolate())
```

### Duplicates

```python
df.is_duplicated()                     # Check for duplicates
df.unique()                            # Remove duplicates
df.unique(subset=['col1', 'col2'])     # Remove based on specific columns
```

### Data type conversion

```python
df.with_columns(
    pl.col('int_col').cast(pl.Float64),
    pl.col('str_col').cast(pl.Categorical),
    pl.col('date_str').str.strptime(pl.Date, '%Y-%m-%d')
)
```

## String Operations

```python
# Basic string operations
df.with_columns(
    pl.col('text').str.to_lowercase().alias('lower'),
    pl.col('text').str.to_uppercase().alias('upper'),
    pl.col('text').str.strip_chars().alias('stripped'),
    pl.col('text').str.len_chars().alias('length')
)

# String manipulation
df.with_columns(
    pl.col('text').str.replace('old', 'new').alias('replaced'),
    pl.col('text').str.contains('pattern').alias('contains'),
    pl.col('text').str.starts_with('prefix').alias('starts_with'),
    pl.col('text').str.ends_with('suffix').alias('ends_with')
)

# String splitting and extraction
df.with_columns(
    pl.col('text').str.split(',').alias('split_list'),
    pl.col('text').str.extract(r'(\d+)', 1).alias('extracted'),
    pl.col('text').str.slice(0, 5).alias('first_5_chars')
)
```

## GroupBy Operations

### Basic grouping

```python
df.group_by('column').agg(
    pl.col('value').sum().alias('sum'),
    pl.col('value').mean().alias('mean'),
    pl.col('value').count().alias('count')
)

# Multiple grouping columns
df.group_by(['col1', 'col2']).agg(
    pl.col('value').sum()
)

# Multiple aggregations per column
df.group_by('group').agg([
    pl.col('value').sum().alias('sum'),
    pl.col('value').mean().alias('mean'),
    pl.col('value').std().alias('std'),
    pl.col('value').min().alias('min'),
    pl.col('value').max().alias('max')
])
```

### Advanced grouping

```python
# Custom aggregations
df.group_by('group').agg(
    pl.col('value').map_elements(lambda x: x.max() - x.min()).alias('range')
)

# Group by expressions
df.group_by(pl.col('date').dt.month()).agg(
    pl.col('value').sum()
)

# Window functions (group_by + over)
df.with_columns(
    pl.col('value').sum().over('group').alias('group_sum'),
    pl.col('value').rank().over('group').alias('rank_in_group')
)
```

## Joins and Concatenation

### Joining DataFrames

```python
# Inner join
df1.join(df2, on='key')
df1.join(df2, on='key', how='inner')

# Different join types
df1.join(df2, on='key', how='left')    # left, right, outer, inner
df1.join(df2, on='key', how='outer')

# Join on different column names
df1.join(df2, left_on='key1', right_on='key2')

# Multiple keys
df1.join(df2, on=['key1', 'key2'])

# Suffix for duplicate column names
df1.join(df2, on='key', suffix='_right')
```

### Concatenation

```python
# Vertical concatenation (stacking)
pl.concat([df1, df2])
pl.concat([df1, df2], how='vertical')

# Horizontal concatenation
pl.concat([df1, df2], how='horizontal')

# Diagonal concatenation (fill missing columns)
pl.concat([df1, df2], how='diagonal')
```

## Pivot and Reshape Operations

### Pivot

```python
df.pivot(
    values='value',
    index='row_key',
    columns='col_key',
    aggregate_function='sum'
)
```

### Melt (unpivot)

```python
df.melt(
    id_vars=['id'],
    value_vars=['col1', 'col2'],
    variable_name='variable',
    value_name='value'
)
```

## Date and Time Operations

### Creating and parsing dates

```python
# Parse string to date
df.with_columns(
    pl.col('date_str').str.strptime(pl.Date, '%Y-%m-%d').alias('date')
)

# Date components
df.with_columns([
    pl.col('date').dt.year().alias('year'),
    pl.col('date').dt.month().alias('month'),
    pl.col('date').dt.day().alias('day'),
    pl.col('date').dt.weekday().alias('weekday')
])

# Date arithmetic
df.with_columns(
    (pl.col('date') + pl.duration(days=30)).alias('date_plus_30'),
    pl.col('date').dt.offset_by('1mo').alias('date_plus_1_month')
)
```

### Date ranges

```python
# Create date range
dates = pl.date_range(
    start=pl.date(2023, 1, 1),
    end=pl.date(2023, 12, 31),
    interval='1d'
)

# Create DataFrame with date range
df_dates = pl.DataFrame({'date': dates})
```

## Window Functions and Rolling Operations

```python
# Rolling operations
df.with_columns([
    pl.col('value').rolling_mean(window_size=7).alias('7day_avg'),
    pl.col('value').rolling_sum(window_size=30).alias('30day_sum'),
    pl.col('value').rolling_std(window_size=7).alias('7day_std')
])

# Window functions with partitioning
df.with_columns([
    pl.col('value').rank().over('group').alias('rank'),
    pl.col('value').sum().over('group').alias('group_total'),
    pl.col('value').shift(1).over('group').alias('previous_value')
])

# Cumulative operations
df.with_columns([
    pl.col('value').cumsum().alias('cumulative_sum'),
    pl.col('value').cummax().alias('cumulative_max'),
    pl.col('value').cumcount().alias('row_number')
])
```

## Lazy Evaluation

```python
# Create lazy frame
lazy_df = pl.scan_csv('large_file.csv')

# Chain operations (not executed yet)
result = (
    lazy_df
    .filter(pl.col('column') > 5)
    .group_by('group')
    .agg(pl.col('value').sum())
    .sort('value', descending=True)
)

# Execute the query
final_result = result.collect()

# Show query plan
print(result.explain())

# Streaming execution for very large datasets
result = result.collect(streaming=True)
```

## Advanced Operations

### Apply custom functions

```python
# Element-wise custom function
df.with_columns(
    pl.col('column').map_elements(lambda x: x ** 2).alias('squared')
)

# Apply to multiple columns
df.with_columns(
    pl.struct(['col1', 'col2'])
    .map_elements(lambda x: x['col1'] + x['col2'])
    .alias('sum')
)
```

### Conditional operations

```python
# When-then-otherwise
df.with_columns(
    pl.when(pl.col('A') > 5)
    .then(pl.lit('high'))
    .when(pl.col('A') > 2)
    .then(pl.lit('medium'))
    .otherwise(pl.lit('low'))
    .alias('category')
)

# Multiple conditions
df.with_columns(
    pl.when((pl.col('A') > 5) & (pl.col('B') < 10))
    .then(pl.lit('valid'))
    .otherwise(pl.lit('invalid'))
    .alias('status')
)
```

### List operations

```python
# Working with list columns
df.with_columns([
    pl.col('list_col').list.len().alias('list_length'),
    pl.col('list_col').list.first().alias('first_item'),
    pl.col('list_col').list.last().alias('last_item'),
    pl.col('list_col').list.sum().alias('list_sum')
])

# Explode lists to rows
df.explode('list_col')

# Create lists from groupby
df.group_by('group').agg(
    pl.col('value').list().alias('value_list')
)
```

## Input/Output Operations

### Saving data

```python
# Parquet (recommended)
df.write_parquet('file.parquet')

# CSV
df.write_csv('file.csv')

# JSON
df.write_json('file.json')
df.write_ndjson('file.ndjson')

# Excel
df.write_excel('file.xlsx')

# Database
df.write_database('table_name', connection_uri)
```

### Advanced I/O options

```python
# Lazy writing for large datasets
pl.scan_csv('input.csv').filter(
    pl.col('column') > 5
).sink_parquet('output.parquet')

# Streaming
pl.scan_csv('large_file.csv').collect(streaming=True)
```

## Performance Optimization

### Best practices

```python
# Use lazy evaluation for large datasets
lazy_df = pl.scan_csv('file.csv')
result = lazy_df.filter(...).group_by(...).collect()

# Prefer Parquet over CSV
df.write_parquet('file.parquet')  # Fast read/write
df_loaded = pl.read_parquet('file.parquet')

# Use appropriate data types
df.with_columns(
    pl.col('category').cast(pl.Categorical),  # Memory efficient
    pl.col('small_int').cast(pl.UInt8)        # Smaller integers
)

# Predicate pushdown in lazy queries
lazy_df.filter(pl.col('date') > '2023-01-01')  # Filter early

# Use streaming for memory efficiency
result = lazy_df.collect(streaming=True)
```

### Memory optimization

```python
# Check memory usage
df.estimated_size()

# Use categorical for repeated strings
df.with_columns(pl.col('category').cast(pl.Categorical))

# Scan with projection (read only needed columns)
pl.scan_csv('file.csv').select(['col1', 'col2']).collect()
```

## Common Patterns and Idioms

### Data validation

```python
# Check data quality
df.select([
    pl.col('*').null_count(),
    pl.col('*').n_unique(),
    pl.all().dtype
])

# Find outliers
df.filter(
    pl.col('value') > (
        pl.col('value').mean() + 3 * pl.col('value').std()
    )
)
```

### Complex transformations

```python
# Rank within groups
df.with_columns(
    pl.col('value')
    .rank(method='ordinal')
    .over('group')
    .alias('rank')
)

# Percentage of total
df.with_columns(
    (pl.col('value') / pl.col('value').sum() * 100).alias('percentage')
)

# Lead and lag
df.with_columns([
    pl.col('value').shift(1).alias('previous'),
    pl.col('value').shift(-1).alias('next')
])
```

### Time series operations

```python
# Resample time series
df.group_by_dynamic(
    'date',
    every='1mo',
    period='1mo'
).agg(pl.col('value').mean())

# Rolling with time-based windows
df.with_columns(
    pl.col('value')
    .rolling_mean_by('date', window_size='30d')
    .alias('30day_rolling_avg')
)
```

## Polars vs Pandas Comparison

```python
# Pandas -> Polars equivalent operations

# pandas: df.groupby('col').sum()
# polars: df.group_by('col').agg(pl.all().sum())

# pandas: df.apply(lambda x: x**2)
# polars: df.with_columns(pl.all().map_elements(lambda x: x**2))

# pandas: df.fillna(0)
# polars: df.fill_null(0)

# pandas: df.drop_duplicates()
# polars: df.unique()

# pandas: df.query('col > 5')
# polars: df.filter(pl.col('col') > 5)
```

## Configuration and Settings

```python
# Display settings
pl.Config.set_tbl_rows(50)
pl.Config.set_tbl_cols(10)
pl.Config.set_tbl_width_chars(100)
pl.Config.set_fmt_str_lengths(50)

# Performance settings
pl.Config.set_streaming_chunk_size(10000)

# Reset to defaults
pl.Config.restore_defaults()
```
