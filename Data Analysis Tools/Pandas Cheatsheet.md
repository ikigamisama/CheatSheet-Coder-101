# Comprehensive Pandas Cheat Sheet

## Import and Setup

```python
import pandas as pd
import numpy as np

# Display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', None)
```

## Creating DataFrames and Series

### From scratch

```python
# DataFrame from dictionary
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': ['x', 'y', 'z'],
    'C': [1.1, 2.2, 3.3]
})

# Series
s = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])

# From lists
df = pd.DataFrame(data, columns=['col1', 'col2'])
```

### Reading Data

```python
# CSV files
df = pd.read_csv('file.csv')
df = pd.read_csv('file.csv', sep=';', encoding='utf-8', index_col=0)

# Excel files
df = pd.read_excel('file.xlsx', sheet_name='Sheet1')

# JSON
df = pd.read_json('file.json')

# SQL
df = pd.read_sql('SELECT * FROM table', connection)

# Clipboard
df = pd.read_clipboard()
```

## Basic Information and Inspection

```python
# Basic info
df.shape                # (rows, columns)
df.info()              # Data types and null counts
df.describe()          # Statistical summary
df.dtypes              # Data types
df.columns             # Column names
df.index               # Index

# First/last rows
df.head(n)             # First n rows (default 5)
df.tail(n)             # Last n rows
df.sample(n)           # Random n rows

# Memory usage
df.memory_usage()
df.memory_usage(deep=True)
```

## Indexing and Selection

### Column selection

```python
df['column']           # Single column (Series)
df[['col1', 'col2']]   # Multiple columns
df.column              # Single column (dot notation)
```

### Row selection

```python
df.iloc[0]             # First row by position
df.iloc[0:3]           # First 3 rows
df.iloc[-1]            # Last row
df.loc['index_name']   # Row by index label
df.loc[0:2, 'A':'C']   # Rows 0-2, columns A-C
```

### Boolean indexing

```python
df[df['column'] > 5]
df[df['col'].isin(['A', 'B'])]
df[(df['col1'] > 5) & (df['col2'] < 10)]
df[df['col'].str.contains('pattern')]
df.query('column > 5 and other_col == "value"')
```

### Setting values

```python
df.loc[df['col'] > 5, 'new_col'] = 'high'
df.at[0, 'column'] = new_value
df.iat[0, 1] = new_value
```

## Data Cleaning and Preprocessing

### Missing data

```python
# Check for missing values
df.isnull().sum()
df.isna().any()
df.info()

# Handle missing values
df.dropna()                    # Drop rows with any NaN
df.dropna(axis=1)              # Drop columns with any NaN
df.dropna(subset=['col'])      # Drop rows where 'col' is NaN
df.fillna(0)                   # Fill NaN with 0
df.fillna(method='ffill')      # Forward fill
df.fillna(method='bfill')      # Backward fill
df.fillna(df.mean())           # Fill with mean
df.interpolate()               # Interpolate missing values
```

### Duplicates

```python
df.duplicated()                # Check for duplicates
df.drop_duplicates()           # Remove duplicates
df.drop_duplicates(subset=['col1', 'col2'])
df.drop_duplicates(keep='last')
```

### Data types

```python
df.astype({'col1': 'int', 'col2': 'str'})
pd.to_numeric(df['col'], errors='coerce')
pd.to_datetime(df['date_col'])
df['col'] = df['col'].astype('category')
```

## Data Manipulation

### Adding/removing columns

```python
df['new_col'] = values
df['new_col'] = df['col1'] + df['col2']
df.drop('column', axis=1)      # Drop column
df.drop(['col1', 'col2'], axis=1)
del df['column']
```

### Adding/removing rows

```python
df.drop(0)                     # Drop row by index
df.drop([0, 1, 2])            # Drop multiple rows
new_row = pd.DataFrame({'A': [1], 'B': [2]})
df = pd.concat([df, new_row], ignore_index=True)
```

### Renaming

```python
df.rename(columns={'old': 'new'})
df.rename(index={0: 'first'})
df.columns = ['new1', 'new2', 'new3']
```

### Sorting

```python
df.sort_values('column')
df.sort_values(['col1', 'col2'], ascending=[True, False])
df.sort_index()
df.nlargest(5, 'column')
df.nsmallest(5, 'column')
```

## String Operations

```python
df['col'].str.lower()
df['col'].str.upper()
df['col'].str.strip()
df['col'].str.replace('old', 'new')
df['col'].str.contains('pattern')
df['col'].str.startswith('prefix')
df['col'].str.endswith('suffix')
df['col'].str.split('delimiter')
df['col'].str.len()
df['col'].str.extract(r'(\d+)')     # Regex extraction
```

## GroupBy Operations

### Basic grouping

```python
df.groupby('column').sum()
df.groupby(['col1', 'col2']).mean()
df.groupby('col').agg({'col1': 'sum', 'col2': 'mean'})

# Multiple aggregations
df.groupby('col').agg({
    'col1': ['sum', 'mean'],
    'col2': ['count', 'std']
})
```

### Advanced groupby

```python
df.groupby('col').apply(lambda x: x.describe())
df.groupby('col').filter(lambda x: len(x) > 5)
df.groupby('col').transform('mean')    # Broadcast back to original size
```

## Pivot Tables and Reshaping

### Pivot tables

```python
pd.pivot_table(df, values='value', index='row', columns='col', aggfunc='mean')
df.pivot(index='row', columns='col', values='value')
```

### Melting and stacking

```python
pd.melt(df, id_vars=['id'], value_vars=['col1', 'col2'])
df.stack()                     # Pivot columns to rows
df.unstack()                   # Pivot rows to columns
```

### Transposing

```python
df.T                           # Transpose
df.transpose()
```

## Merging and Joining

### Concatenation

```python
pd.concat([df1, df2])                    # Vertical
pd.concat([df1, df2], axis=1)            # Horizontal
pd.concat([df1, df2], ignore_index=True) # Reset index
```

### Merging

```python
pd.merge(df1, df2, on='key')
pd.merge(df1, df2, left_on='key1', right_on='key2')
pd.merge(df1, df2, how='left')   # left, right, outer, inner
df1.merge(df2, on='key', suffixes=('_x', '_y'))
```

### Joining

```python
df1.join(df2, how='outer')
df1.set_index('key').join(df2.set_index('key'))
```

## Date and Time Operations

### Converting to datetime

```python
pd.to_datetime(df['date_col'])
pd.to_datetime(df['date_col'], format='%Y-%m-%d')
```

### Date operations

```python
df['date'].dt.year
df['date'].dt.month
df['date'].dt.day
df['date'].dt.dayofweek
df['date'].dt.strftime('%Y-%m')

# Date ranges
pd.date_range('2023-01-01', '2023-12-31', freq='D')
pd.date_range('2023-01-01', periods=365, freq='D')
```

### Resampling (time series)

```python
df.set_index('date').resample('M').mean()    # Monthly average
df.set_index('date').resample('D').sum()     # Daily sum
```

## Statistical Operations

### Descriptive statistics

```python
df.mean()
df.median()
df.std()
df.var()
df.min(), df.max()
df.quantile(0.25)
df.corr()                      # Correlation matrix
df.cov()                       # Covariance matrix
```

### Rolling operations

```python
df['col'].rolling(window=7).mean()    # 7-day moving average
df['col'].rolling(window=30).std()
df['col'].expanding().mean()          # Expanding mean
```

## Advanced Operations

### Apply functions

```python
df.apply(function)             # Apply to each column
df.apply(function, axis=1)     # Apply to each row
df['col'].apply(lambda x: x**2)
df.applymap(function)          # Apply to each element
```

### Mapping and replacing

```python
df['col'].map({'old': 'new'})
df['col'].replace({'old': 'new'})
df['col'].replace([1, 2], [10, 20])
```

### Binning and categorizing

```python
pd.cut(df['col'], bins=5)              # Equal-width bins
pd.qcut(df['col'], q=4)                # Quartiles
pd.get_dummies(df['categorical_col'])   # One-hot encoding
```

## Input/Output Operations

### Saving data

```python
df.to_csv('file.csv', index=False)
df.to_excel('file.xlsx', sheet_name='Sheet1')
df.to_json('file.json')
df.to_sql('table_name', connection)
df.to_pickle('file.pkl')
```

### Export options

```python
df.to_csv('file.csv', sep=';', encoding='utf-8', na_rep='NULL')
df.to_clipboard()              # Copy to clipboard
```

## Performance Tips

### Memory optimization

```python
# Use categorical for strings
df['col'] = df['col'].astype('category')

# Downcast numeric types
df['int_col'] = pd.to_numeric(df['int_col'], downcast='integer')

# Use chunks for large files
for chunk in pd.read_csv('large_file.csv', chunksize=10000):
    process(chunk)
```

### Efficient operations

```python
# Use vectorized operations instead of loops
df['new_col'] = df['col1'] * df['col2']  # Good
# df['new_col'] = df.apply(lambda x: x['col1'] * x['col2'], axis=1)  # Slower

# Use .loc for setting values
df.loc[condition, 'col'] = value

# Use .query() for complex boolean indexing
df.query('col1 > 5 and col2 < 10')
```

## Common Patterns and Idioms

### Working with MultiIndex

```python
df.set_index(['col1', 'col2'])
df.reset_index()
df.swaplevel()
df.xs('key', level='level_name')
```

### Conditional operations

```python
# If-else logic
df['new_col'] = np.where(df['col'] > 5, 'high', 'low')
df['new_col'] = df['col'].apply(lambda x: 'high' if x > 5 else 'low')

# Multiple conditions
conditions = [
    df['col'] < 5,
    df['col'] < 10,
    df['col'] >= 10
]
choices = ['low', 'medium', 'high']
df['category'] = np.select(conditions, choices, default='unknown')
```

### Window functions

```python
df['rank'] = df['col'].rank()
df['pct_rank'] = df['col'].rank(pct=True)
df['cumsum'] = df['col'].cumsum()
df['cummax'] = df['col'].cummax()
df['shift'] = df['col'].shift(1)     # Lag by 1
df['lead'] = df['col'].shift(-1)     # Lead by 1
```
