# Design Patterns for Data Engineering & Data Science

## 1. Factory Pattern

**Definition**: Creates objects without specifying their exact class, allowing runtime decisions about which class to instantiate.

**Purpose**: Provides flexibility in object creation, especially useful when dealing with different data sources or processing engines.

### Example 1: Data Source Factory

```python
from abc import ABC, abstractmethod
import pandas as pd
import sqlite3
import json

class DataSource(ABC):
    @abstractmethod
    def read_data(self, source: str) -> pd.DataFrame:
        pass

class CSVDataSource(DataSource):
    def read_data(self, source: str) -> pd.DataFrame:
        return pd.read_csv(source)

class SQLDataSource(DataSource):
    def read_data(self, source: str) -> pd.DataFrame:
        conn = sqlite3.connect(source)
        return pd.read_sql_query("SELECT * FROM table", conn)

class ParquetDataSource(DataSource):
    def read_data(self, source: str) -> pd.DataFrame:
        return pd.read_parquet(source)

class JSONDataSource(DataSource):
    def read_data(self, source: str) -> pd.DataFrame:
        return pd.read_json(source)

class DataSourceFactory:
    @staticmethod
    def create_datasource(file_type: str) -> DataSource:
        sources = {
            'csv': CSVDataSource,
            'sql': SQLDataSource,
            'parquet': ParquetDataSource,
            'json': JSONDataSource
        }
        if file_type.lower() in sources:
            return sources[file_type.lower()]()
        raise ValueError(f"Unsupported file type: {file_type}")

# Usage
factory = DataSourceFactory()
csv_reader = factory.create_datasource('csv')
data = csv_reader.read_data('sales_data.csv')
```

### Example 2: ML Model Factory

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

class ModelFactory:
    @staticmethod
    def create_model(model_type: str, **kwargs):
        models = {
            'rf': RandomForestClassifier,
            'gb': GradientBoostingClassifier,
            'lr': LogisticRegression,
            'svm': SVC,
            'nb': GaussianNB
        }
        if model_type.lower() in models:
            return models[model_type.lower()](**kwargs)
        raise ValueError(f"Unsupported model type: {model_type}")

# Usage
model_factory = ModelFactory()
rf_model = model_factory.create_model('rf', n_estimators=100)
gb_model = model_factory.create_model('gb', learning_rate=0.1)
```

### Example 3: Database Connection Factory

```python
import sqlite3
import psycopg2
from pymongo import MongoClient

class DatabaseFactory:
    @staticmethod
    def create_connection(db_type: str, **kwargs):
        if db_type.lower() == 'sqlite':
            return sqlite3.connect(kwargs.get('database', 'default.db'))
        elif db_type.lower() == 'postgresql':
            return psycopg2.connect(
                host=kwargs.get('host', 'localhost'),
                database=kwargs.get('database'),
                user=kwargs.get('user'),
                password=kwargs.get('password')
            )
        elif db_type.lower() == 'mongodb':
            return MongoClient(kwargs.get('uri', 'mongodb://localhost:27017/'))
        else:
            raise ValueError(f"Unsupported database type: {db_type}")

# Usage
sqlite_conn = DatabaseFactory.create_connection('sqlite', database='data.db')
postgres_conn = DatabaseFactory.create_connection('postgresql',
                                                  host='localhost',
                                                  database='analytics',
                                                  user='user',
                                                  password='pass')
```

### Example 4: Data Transformer Factory

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

class TransformerFactory:
    @staticmethod
    def create_transformer(transformer_type: str, **kwargs):
        transformers = {
            'standard': StandardScaler,
            'minmax': MinMaxScaler,
            'robust': RobustScaler,
            'pca': PCA,
            'kbest': SelectKBest
        }
        if transformer_type.lower() in transformers:
            return transformers[transformer_type.lower()](**kwargs)
        raise ValueError(f"Unsupported transformer type: {transformer_type}")

# Usage
scaler = TransformerFactory.create_transformer('standard')
pca = TransformerFactory.create_transformer('pca', n_components=2)
```

### Example 5: Visualization Factory

```python
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

class VisualizationFactory:
    @staticmethod
    def create_plot(plot_type: str, data, x, y=None, **kwargs):
        if plot_type.lower() == 'scatter':
            return plt.scatter(data[x], data[y], **kwargs)
        elif plot_type.lower() == 'histogram':
            return plt.hist(data[x], **kwargs)
        elif plot_type.lower() == 'boxplot':
            return sns.boxplot(data=data, x=x, y=y, **kwargs)
        elif plot_type.lower() == 'interactive_scatter':
            return px.scatter(data, x=x, y=y, **kwargs)
        else:
            raise ValueError(f"Unsupported plot type: {plot_type}")

# Usage
import pandas as pd
df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
plot = VisualizationFactory.create_plot('scatter', df, 'x', 'y')
```

### Example 6: Data Validator Factory

```python
from abc import ABC, abstractmethod
import pandas as pd

class DataValidator(ABC):
    @abstractmethod
    def validate(self, data: pd.DataFrame) -> bool:
        pass

class NullValidator(DataValidator):
    def validate(self, data: pd.DataFrame) -> bool:
        return not data.isnull().any().any()

class DuplicateValidator(DataValidator):
    def validate(self, data: pd.DataFrame) -> bool:
        return not data.duplicated().any()

class SchemaValidator(DataValidator):
    def __init__(self, expected_columns):
        self.expected_columns = expected_columns

    def validate(self, data: pd.DataFrame) -> bool:
        return set(data.columns) == set(self.expected_columns)

class ValidatorFactory:
    @staticmethod
    def create_validator(validator_type: str, **kwargs):
        validators = {
            'null': NullValidator,
            'duplicate': DuplicateValidator,
            'schema': lambda: SchemaValidator(kwargs.get('columns', []))
        }
        if validator_type.lower() in validators:
            return validators[validator_type.lower()]()
        raise ValueError(f"Unsupported validator type: {validator_type}")

# Usage
null_validator = ValidatorFactory.create_validator('null')
schema_validator = ValidatorFactory.create_validator('schema', columns=['A', 'B', 'C'])
```

### Example 7: Feature Engineering Factory

```python
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

class FeatureEngineerFactory:
    @staticmethod
    def create_feature_engineer(engineer_type: str, **kwargs):
        engineers = {
            'polynomial': PolynomialFeatures,
            'onehot': OneHotEncoder,
            'tfidf': TfidfVectorizer
        }
        if engineer_type.lower() in engineers:
            return engineers[engineer_type.lower()](**kwargs)
        raise ValueError(f"Unsupported feature engineer type: {engineer_type}")

# Usage
poly_features = FeatureEngineerFactory.create_feature_engineer('polynomial', degree=2)
onehot_encoder = FeatureEngineerFactory.create_feature_engineer('onehot')
```

---

## 2. Singleton Pattern

**Definition**: Ensures a class has only one instance and provides global access to it.

**Purpose**: Useful for shared resources like database connections, configuration settings, or logging systems.

### Example 1: Database Connection Manager

```python
import sqlite3
from threading import Lock

class DatabaseManager:
    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._connection = None
        return cls._instance

    def get_connection(self):
        if self._connection is None:
            self._connection = sqlite3.connect('data_warehouse.db')
        return self._connection

    def execute_query(self, query: str):
        conn = self.get_connection()
        return conn.execute(query).fetchall()

# Usage
db1 = DatabaseManager()
db2 = DatabaseManager()
print(db1 is db2)  # True
```

### Example 2: Configuration Manager

```python
import json
from typing import Dict, Any

class ConfigManager:
    _instance = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load_config(self, config_file: str):
        if self._config is None:
            with open(config_file, 'r') as f:
                self._config = json.load(f)

    def get_config(self, key: str, default=None):
        return self._config.get(key, default) if self._config else default

# Usage
config = ConfigManager()
config.load_config('pipeline_config.json')
batch_size = config.get_config('batch_size', 1000)
```

### Example 3: Logger Manager

```python
import logging
from threading import Lock

class LoggerManager:
    _instance = None
    _lock = Lock()
    _logger = None

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._setup_logger()
        return cls._instance

    def _setup_logger(self):
        self._logger = logging.getLogger('DataPipeline')
        self._logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)

    def get_logger(self):
        return self._logger

# Usage
logger_manager = LoggerManager()
logger = logger_manager.get_logger()
logger.info("Processing data batch")
```

### Example 4: Cache Manager

```python
import pickle
import os
from threading import Lock

class CacheManager:
    _instance = None
    _lock = Lock()
    _cache = {}

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def get(self, key: str):
        return self._cache.get(key)

    def set(self, key: str, value):
        self._cache[key] = value

    def save_to_disk(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump(self._cache, f)

    def load_from_disk(self, filepath: str):
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                self._cache = pickle.load(f)

# Usage
cache = CacheManager()
cache.set('processed_data', {'results': [1, 2, 3]})
data = cache.get('processed_data')
```

### Example 5: Model Registry

```python
import joblib
from threading import Lock

class ModelRegistry:
    _instance = None
    _lock = Lock()
    _models = {}

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def register_model(self, name: str, model):
        self._models[name] = model

    def get_model(self, name: str):
        return self._models.get(name)

    def list_models(self):
        return list(self._models.keys())

    def save_model(self, name: str, filepath: str):
        if name in self._models:
            joblib.dump(self._models[name], filepath)

# Usage
registry = ModelRegistry()
registry.register_model('rf_classifier', RandomForestClassifier())
model = registry.get_model('rf_classifier')
```

### Example 6: Resource Pool Manager

```python
import queue
from threading import Lock

class ResourcePoolManager:
    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._connections = queue.Queue()
                    cls._instance._max_connections = 10
        return cls._instance

    def get_connection(self):
        if not self._connections.empty():
            return self._connections.get()
        else:
            # Create new connection if pool not full
            return self._create_connection()

    def return_connection(self, connection):
        self._connections.put(connection)

    def _create_connection(self):
        # Simulate creating a database connection
        return f"Connection_{id(self)}"

# Usage
pool = ResourcePoolManager()
conn = pool.get_connection()
# Use connection
pool.return_connection(conn)
```

### Example 7: Metrics Collector

```python
import time
from collections import defaultdict
from threading import Lock

class MetricsCollector:
    _instance = None
    _lock = Lock()
    _metrics = defaultdict(list)

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def record_metric(self, name: str, value: float):
        self._metrics[name].append({
            'value': value,
            'timestamp': time.time()
        })

    def get_metric_stats(self, name: str):
        values = [m['value'] for m in self._metrics[name]]
        if not values:
            return None
        return {
            'count': len(values),
            'avg': sum(values) / len(values),
            'min': min(values),
            'max': max(values)
        }

# Usage
metrics = MetricsCollector()
metrics.record_metric('processing_time', 2.5)
metrics.record_metric('accuracy', 0.95)
stats = metrics.get_metric_stats('processing_time')
```

---

## 3. Decorator Pattern

**Definition**: Dynamically adds new functionality to objects without altering their structure.

**Purpose**: Enhances data processing functions with additional features like logging, validation, or caching.

### Example 1: Execution Time Logger

```python
import time
import logging
from functools import wraps

def log_execution_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.info(f"{func.__name__} executed in {end_time - start_time:.2f} seconds")
        return result
    return wrapper

@log_execution_time
def process_large_dataset(data):
    # Simulate processing
    time.sleep(1)
    return data.sum()

# Usage
import pandas as pd
df = pd.DataFrame({'A': range(1000)})
result = process_large_dataset(df)
```

### Example 2: Data Validation Decorator

```python
from functools import wraps
import pandas as pd

def validate_dataframe(func):
    @wraps(func)
    def wrapper(df, *args, **kwargs):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        if df.empty:
            raise ValueError("DataFrame is empty")
        if df.isnull().any().any():
            logging.warning("DataFrame contains null values")
        return func(df, *args, **kwargs)
    return wrapper

@validate_dataframe
def clean_data(df):
    return df.drop_duplicates().fillna(0)

# Usage
df = pd.DataFrame({'A': [1, 2, None, 4], 'B': [5, 6, 7, 8]})
cleaned = clean_data(df)
```

### Example 3: Result Caching Decorator

```python
from functools import wraps
import pickle
import os
import hashlib

def cache_results(cache_dir='cache'):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            cache_key = hashlib.md5(str(args).encode() + str(kwargs).encode()).hexdigest()
            cache_file = os.path.join(cache_dir, f"{func.__name__}_{cache_key}.pkl")

            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)

            result = func(*args, **kwargs)

            os.makedirs(cache_dir, exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)

            return result
        return wrapper
    return decorator

@cache_results()
def expensive_computation(data):
    import time
    time.sleep(2)
    return data.describe()

# Usage
df = pd.DataFrame({'A': range(100), 'B': range(100, 200)})
result = expensive_computation(df)  # Cached after first call
```

### Example 4: Error Handling Decorator

```python
from functools import wraps
import logging

def handle_errors(default_return=None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logging.error(f"Error in {func.__name__}: {str(e)}")
                return default_return
        return wrapper
    return decorator

@handle_errors(default_return=pd.DataFrame())
def risky_data_operation(df):
    # This might fail
    return df.divide_by_zero()

# Usage
df = pd.DataFrame({'A': [1, 2, 3]})
result = risky_data_operation(df)  # Returns empty DataFrame on error
```

### Example 5: Performance Monitoring Decorator

```python
import psutil
import time
from functools import wraps

def monitor_performance(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Monitor before execution
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        result = func(*args, **kwargs)

        # Monitor after execution
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        print(f"{func.__name__} Performance:")
        print(f"  Time: {end_time - start_time:.2f} seconds")
        print(f"  Memory usage: {end_memory - start_memory:.2f} MB")

        return result
    return wrapper

@monitor_performance
def memory_intensive_operation(size):
    import numpy as np
    data = np.random.randn(size, size)
    return data.sum()

# Usage
result = memory_intensive_operation(1000)
```

### Example 6: Retry Decorator

```python
import time
from functools import wraps

def retry(max_attempts=3, delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise e
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(delay)
        return wrapper
    return decorator

@retry(max_attempts=3, delay=2)
def unreliable_api_call():
    import random
    if random.random() < 0.7:  # 70% chance of failure
        raise Exception("API call failed")
    return {"data": "success"}

# Usage
result = unreliable_api_call()
```

### Example 7: Input/Output Logging Decorator

```python
from functools import wraps
import json

def log_inputs_outputs(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Log inputs
        print(f"Calling {func.__name__} with:")
        print(f"  Args: {args}")
        print(f"  Kwargs: {kwargs}")

        result = func(*args, **kwargs)

        # Log outputs
        print(f"  Result: {result}")

        return result
    return wrapper

@log_inputs_outputs
def calculate_statistics(data, percentiles=[25, 50, 75]):
    return {
        'mean': data.mean(),
        'std': data.std(),
        'percentiles': data.quantile([p/100 for p in percentiles])
    }

# Usage
import pandas as pd
df = pd.Series([1, 2, 3, 4, 5])
stats = calculate_statistics(df)
```

### Example 8: Data Type Conversion Decorator

```python
from functools import wraps
import pandas as pd

def ensure_dataframe(func):
    @wraps(func)
    def wrapper(data, *args, **kwargs):
        if not isinstance(data, pd.DataFrame):
            if isinstance(data, dict):
                data = pd.DataFrame(data)
            elif isinstance(data, list):
                data = pd.DataFrame(data)
            else:
                raise TypeError("Data must be convertible to DataFrame")
        return func(data, *args, **kwargs)
    return wrapper

@ensure_dataframe
def analyze_data(df):
    return df.describe()

# Usage
# Works with different input types
dict_data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
result = analyze_data(dict_data)
```

---

## 4. Strategy Pattern

**Definition**: Defines a family of algorithms, encapsulates each one, and makes them interchangeable.

**Purpose**: Allows switching between different algorithms or processing methods at runtime.

### Example 1: Data Scaling Strategies

```python
from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import pandas as pd

class ScalingStrategy(ABC):
    @abstractmethod
    def scale(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

class StandardScalingStrategy(ScalingStrategy):
    def scale(self, data: pd.DataFrame) -> pd.DataFrame:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        return pd.DataFrame(scaled_data, columns=data.columns)

class MinMaxScalingStrategy(ScalingStrategy):
    def scale(self, data: pd.DataFrame) -> pd.DataFrame:
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
        return pd.DataFrame(scaled_data, columns=data.columns)

class RobustScalingStrategy(ScalingStrategy):
    def scale(self, data: pd.DataFrame) -> pd.DataFrame:
        scaler = RobustScaler()
        scaled_data = scaler.fit_transform(data)
        return pd.DataFrame(scaled_data, columns=data.columns)

class DataScaler:
    def __init__(self, strategy: ScalingStrategy):
        self.strategy = strategy

    def set_strategy(self, strategy: ScalingStrategy):
        self.strategy = strategy

    def scale_data(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.strategy.scale(data)

# Usage
df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [10, 20, 30, 40, 50]})
scaler = DataScaler(StandardScalingStrategy())
scaled_data = scaler.scale_data(df)
```

### Example 2: Feature Selection Strategies

```python
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier

class FeatureSelectionStrategy(ABC):
    @abstractmethod
    def select_features(self, X, y):
        pass

class KBestStrategy(FeatureSelectionStrategy):
    def __init__(self, k=5):
        self.k = k

    def select_features(self, X, y):
        selector = SelectKBest(score_func=f_classif, k=self.k)
        X_selected = selector.fit_transform(X, y)
        return X_selected, selector.get_support()

class RFEStrategy(FeatureSelectionStrategy):
    def __init__(self, n_features=5):
        self.n_features = n_features

    def select_features(self, X, y):
        estimator = RandomForestClassifier()
        selector = RFE(estimator, n_features_to_select=self.n_features)
        X_selected = selector.fit_transform(X, y)
        return X_selected, selector.support_

class ImportanceBasedStrategy(FeatureSelectionStrategy):
    def __init__(self, threshold=0.1):
        self.threshold = threshold

    def select_features(self, X, y):
        rf = RandomForestClassifier()
        rf.fit(X, y)
        importances = rf.feature_importances_
        mask = importances > self.threshold
        return X[:, mask], mask

# Usage
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=100, n_features=10, n_informative=5)

selector = FeatureSelector(KBestStrategy(k=3))
X_selected, feature_mask = selector.select(X, y)
```

### Example 3: Missing Value Handling Strategies

```python
from sklearn.impute import SimpleImputer, KNNImputer
import pandas as pd

class MissingValueStrategy(ABC):
    @abstractmethod
    def handle_missing(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

class DropMissingStrategy(MissingValueStrategy):
    def handle_missing(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.dropna()

class MeanImputeStrategy(MissingValueStrategy):
    def handle_missing(self, data: pd.DataFrame) -> pd.DataFrame:
        imputer = SimpleImputer(strategy='mean')
        numeric_cols = data.select_dtypes(include=['number']).columns
        data[numeric_cols] = imputer.fit_transform(data[numeric_cols])
        return data

class KNNImputeStrategy(MissingValueStrategy):
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def handle_missing(self, data: pd.DataFrame) -> pd.DataFrame:
        imputer = KNNImputer(n_neighbors=self.n_neighbors)
        numeric_cols = data.select_dtypes(include=['number']).columns
        data[numeric_cols] = imputer.fit_transform(data[numeric_cols])
        return data

class MissingValueHandler:
    def __init__(self, strategy: MissingValueStrategy):
        self.strategy = strategy

    def handle(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.strategy.handle_missing(data)

# Usage
df = pd.DataFrame({'A': [1, 2, None, 4], 'B': [5, None, 7, 8]})
handler = MissingValueHandler(MeanImputeStrategy())
cleaned_data = handler.handle(df)
```

### Example 4: Model Evaluation Strategies

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score

class EvaluationStrategy(ABC):
    @abstractmethod
    def evaluate(self, model, X, y):
        pass

class AccuracyEvaluationStrategy(EvaluationStrategy):
    def evaluate(self, model, X, y):
        predictions = model.predict(X)
        return accuracy_score(y, predictions)

class PrecisionEvaluationStrategy(EvaluationStrategy):
    def evaluate(self, model, X, y):
        predictions = model.predict(X)
        return precision_score(y, predictions, average='weighted')

class CrossValidationStrategy(EvaluationStrategy):
    def __init__(self, cv=5):
        self.cv = cv

    def evaluate(self, model, X, y):
        scores = cross_val_score(model, X, y, cv=self.cv)
        return scores.mean()

class ModelEvaluator:
    def __init__(self, strategy: EvaluationStrategy):
        self.strategy = strategy

    def evaluate(self, model, X, y):
        return self.strategy.evaluate(model, X, y)

# Usage
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=100, n_features=10)
model = RandomForestClassifier()
model.fit(X, y)

evaluator = ModelEvaluator(AccuracyEvaluationStrategy())
score = evaluator.evaluate(model, X, y)
```

### Example 5: Data Sampling Strategies

```python
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

class SamplingStrategy(ABC):
    @abstractmethod
    def sample(self, X, y):
        pass

class RandomOversamplingStrategy(SamplingStrategy):
    def sample(self, X, y):
        X_resampled, y_resampled = resample(X, y, replace=True, random_state=42)
        return X_resampled, y_resampled

class SMOTEStrategy(SamplingStrategy):
    def sample(self, X, y):
        smote = SMOTE(random_state=42)
        return smote.fit_resample(X, y)

class UndersamplingStrategy(SamplingStrategy):
    def sample(self, X, y):
        undersampler = RandomUnderSampler(random_state=42)
        return undersampler.fit_resample(X, y)

class DataSampler:
    def __init__(self, strategy: SamplingStrategy):
        self.strategy = strategy

    def sample(self, X, y):
        return self.strategy.sample(X, y)

# Usage
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=100, n_features=10, n_classes=2, weights=[0.9, 0.1])

sampler = DataSampler(SMOTEStrategy())
X_balanced, y_balanced = sampler.sample(X, y)
```

### Example 6: Data Export Strategies

```python
class ExportStrategy(ABC):
    @abstractmethod
    def export(self, data: pd.DataFrame, filepath: str):
        pass

class CSVExportStrategy(ExportStrategy):
    def export(self, data: pd.DataFrame, filepath: str):
        data.to_csv(filepath, index=False)

class ParquetExportStrategy(ExportStrategy):
    def export(self, data: pd.DataFrame, filepath: str):
        data.to_parquet(filepath)

class JSONExportStrategy(ExportStrategy):
    def export(self, data: pd.DataFrame, filepath: str):
        data.to_json(filepath, orient='records')

class ExcelExportStrategy(ExportStrategy):
    def export(self, data: pd.DataFrame, filepath: str):
        data.to_excel(filepath, index=False)

class DataExporter:
    def __init__(self, strategy: ExportStrategy):
        self.strategy = strategy

    def export(self, data: pd.DataFrame, filepath: str):
        return self.strategy.export(data, filepath)

# Usage
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
exporter = DataExporter(ParquetExportStrategy())
exporter.export(df, 'output.parquet')
```

### Example 7: Outlier Detection Strategies

```python
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
import numpy as np

class OutlierDetectionStrategy(ABC):
    @abstractmethod
    def detect_outliers(self, data: pd.DataFrame) -> np.ndarray:
        pass

class IQROutlierStrategy(OutlierDetectionStrategy):
    def detect_outliers(self, data: pd.DataFrame) -> np.ndarray:
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        outlier_mask = ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)
        return outlier_mask.values

class IsolationForestStrategy(OutlierDetectionStrategy):
    def detect_outliers(self, data: pd.DataFrame) -> np.ndarray:
        clf = IsolationForest(contamination=0.1, random_state=42)
        outlier_labels = clf.fit_predict(data)
        return outlier_labels == -1

class EllipticEnvelopeStrategy(OutlierDetectionStrategy):
    def detect_outliers(self, data: pd.DataFrame) -> np.ndarray:
        clf = EllipticEnvelope(contamination=0.1, random_state=42)
        outlier_labels = clf.fit_predict(data)
        return outlier_labels == -1

class OutlierDetector:
    def __init__(self, strategy: OutlierDetectionStrategy):
        self.strategy = strategy

    def detect(self, data: pd.DataFrame) -> np.ndarray:
        return self.strategy.detect_outliers(data)

# Usage
df = pd.DataFrame({'A': [1, 2, 3, 100, 5], 'B': [2, 4, 6, 8, 200]})
detector = OutlierDetector(IQROutlierStrategy())
outliers = detector.detect(df)
```

### Example 8: Text Processing Strategies

```python
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
import pandas as pd

class TextProcessingStrategy(ABC):
    @abstractmethod
    def process_text(self, texts: list) -> pd.DataFrame:
        pass

class TfidfStrategy(TextProcessingStrategy):
    def __init__(self, max_features=1000):
        self.max_features = max_features

    def process_text(self, texts: list) -> pd.DataFrame:
        vectorizer = TfidfVectorizer(max_features=self.max_features)
        features = vectorizer.fit_transform(texts)
        return pd.DataFrame(features.toarray(), columns=vectorizer.get_feature_names_out())

class CountVectorizerStrategy(TextProcessingStrategy):
    def __init__(self, max_features=1000):
        self.max_features = max_features

    def process_text(self, texts: list) -> pd.DataFrame:
        vectorizer = CountVectorizer(max_features=self.max_features)
        features = vectorizer.fit_transform(texts)
        return pd.DataFrame(features.toarray(), columns=vectorizer.get_feature_names_out())

class HashingStrategy(TextProcessingStrategy):
    def __init__(self, n_features=1000):
        self.n_features = n_features

    def process_text(self, texts: list) -> pd.DataFrame:
        vectorizer = HashingVectorizer(n_features=self.n_features)
        features = vectorizer.fit_transform(texts)
        return pd.DataFrame(features.toarray())

class TextProcessor:
    def __init__(self, strategy: TextProcessingStrategy):
        self.strategy = strategy

    def process(self, texts: list) -> pd.DataFrame:
        return self.strategy.process_text(texts)

# Usage
texts = ["This is a sample text", "Another text document", "Third text example"]
processor = TextProcessor(TfidfStrategy(max_features=50))
features = processor.process(texts)
```

---

## 5. Pipeline Pattern

**Definition**: Processes data through a series of connected stages, where each stage transforms the data.

**Purpose**: Creates modular, reusable data processing workflows that are easy to maintain and extend.

### Example 1: Data Preprocessing Pipeline

```python
from abc import ABC, abstractmethod
import pandas as pd
from sklearn.preprocessing import StandardScaler

class PipelineStage(ABC):
    @abstractmethod
    def process(self, data):
        pass

class DataCleaningStage(PipelineStage):
    def process(self, data):
        data = data.drop_duplicates()
        data = data.fillna(data.mean())
        return data

class FeatureEngineeringStage(PipelineStage):
    def process(self, data):
        if 'age' in data.columns and 'income' in data.columns:
            data['age_income_ratio'] = data['age'] / data['income']
        return data

class DataScalingStage(PipelineStage):
    def process(self, data):
        numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
        scaler = StandardScaler()
        data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
        return data

class DataPipeline:
    def __init__(self):
        self.stages = []

    def add_stage(self, stage: PipelineStage):
        self.stages.append(stage)
        return self

    def execute(self, data):
        result = data
        for stage in self.stages:
            result = stage.process(result)
        return result

# Usage
pipeline = DataPipeline()
pipeline.add_stage(DataCleaningStage()) \
        .add_stage(FeatureEngineeringStage()) \
        .add_stage(DataScalingStage())

df = pd.DataFrame({
    'age': [25, 30, None, 35, 40],
    'income': [50000, 60000, 70000, 80000, 90000],
    'score': [85, 90, 78, 92, 88]
})

processed_data = pipeline.execute(df)
```

### Example 2: ML Model Pipeline

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

class TrainTestSplitStage(PipelineStage):
    def __init__(self, test_size=0.2):
        self.test_size = test_size

    def process(self, data):
        X = data.drop('target', axis=1)
        y = data['target']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=42
        )
        return {
            'X_train': X_train, 'X_test': X_test,
            'y_train': y_train, 'y_test': y_test
        }

class ModelTrainingStage(PipelineStage):
    def __init__(self, model=None):
        self.model = model or RandomForestClassifier()

    def process(self, data):
        self.model.fit(data['X_train'], data['y_train'])
        data['model'] = self.model
        return data

class ModelEvaluationStage(PipelineStage):
    def process(self, data):
        predictions = data['model'].predict(data['X_test'])
        accuracy = accuracy_score(data['y_test'], predictions)
        data['accuracy'] = accuracy
        data['predictions'] = predictions
        return data

class ModelSavingStage(PipelineStage):
    def __init__(self, model_path='model.pkl'):
        self.model_path = model_path

    def process(self, data):
        joblib.dump(data['model'], self.model_path)
        data['model_path'] = self.model_path
        return data

# Usage
ml_pipeline = DataPipeline()
ml_pipeline.add_stage(TrainTestSplitStage(test_size=0.3)) \
           .add_stage(ModelTrainingStage()) \
           .add_stage(ModelEvaluationStage()) \
           .add_stage(ModelSavingStage('rf_model.pkl'))

df_ml = pd.DataFrame({
    'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'feature2': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
    'target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
})

results = ml_pipeline.execute(df_ml)
```

### Example 3: ETL Pipeline

```python
class DataExtractionStage(PipelineStage):
    def __init__(self, source_path):
        self.source_path = source_path

    def process(self, data):
        if self.source_path.endswith('.csv'):
            return pd.read_csv(self.source_path)
        elif self.source_path.endswith('.json'):
            return pd.read_json(self.source_path)
        else:
            raise ValueError(f"Unsupported file format: {self.source_path}")

class DataTransformationStage(PipelineStage):
    def process(self, data):
        # Apply transformations
        data['processed_date'] = pd.to_datetime(data['date'])
        data['year'] = data['processed_date'].dt.year
        data['month'] = data['processed_date'].dt.month
        return data

class DataLoadingStage(PipelineStage):
    def __init__(self, output_path):
        self.output_path = output_path

    def process(self, data):
        data.to_csv(self.output_path, index=False)
        return data

# Usage
etl_pipeline = DataPipeline()
etl_pipeline.add_stage(DataExtractionStage('raw_data.csv')) \
            .add_stage(DataTransformationStage()) \
            .add_stage(DataLoadingStage('processed_data.csv'))

result = etl_pipeline.execute(None)
```

### Example 4: Text Processing Pipeline

```python
import re
from sklearn.feature_extraction.text import TfidfVectorizer

class TextCleaningStage(PipelineStage):
    def process(self, data):
        data['cleaned_text'] = data['text'].apply(self._clean_text)
        return data

    def _clean_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text

class TextTokenizationStage(PipelineStage):
    def process(self, data):
        data['tokens'] = data['cleaned_text'].apply(lambda x: x.split())
        return data

class TextVectorizationStage(PipelineStage):
    def process(self, data):
        vectorizer = TfidfVectorizer()
        features = vectorizer.fit_transform(data['cleaned_text'])
        feature_names = vectorizer.get_feature_names_out()

        feature_df = pd.DataFrame(features.toarray(), columns=feature_names)
        return pd.concat([data, feature_df], axis=1)

# Usage
text_pipeline = DataPipeline()
text_pipeline.add_stage(TextCleaningStage()) \
             .add_stage(TextTokenizationStage()) \
             .add_stage(TextVectorizationStage())

text_df = pd.DataFrame({
    'text': ['Hello World!', 'Data Science is Amazing!', 'Python is Great!!!']
})

processed_text = text_pipeline.execute(text_df)
```

### Example 5: Feature Engineering Pipeline

```python
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder

class CategoricalEncodingStage(PipelineStage):
    def process(self, data):
        categorical_cols = data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            encoder = LabelEncoder()
            data[f'{col}_encoded'] = encoder.fit_transform(data[col])
        return data

class PolynomialFeatureStage(PipelineStage):
    def __init__(self, degree=2):
        self.degree = degree

    def process(self, data):
        numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
        poly = PolynomialFeatures(degree=self.degree, include_bias=False)
        poly_features = poly.fit_transform(data[numerical_cols])
        poly_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out())
        return pd.concat([data, poly_df], axis=1)

class FeatureSelectionStage(PipelineStage):
    def __init__(self, threshold=0.1):
        self.threshold = threshold

    def process(self, data):
        correlation_matrix = data.corr().abs()
        upper_triangle = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        high_corr_features = [col for col in upper_triangle.columns if any(upper_triangle[col] > self.threshold)]
        return data.drop(columns=high_corr_features)

# Usage
feature_pipeline = DataPipeline()
feature_pipeline.add_stage(CategoricalEncodingStage()) \
                 .add_stage(PolynomialFeatureStage(degree=2)) \
                 .add_stage(FeatureSelectionStage(threshold=0.8))

df = pd.DataFrame({
    'category': ['A', 'B', 'C', 'A', 'B'],
    'value1': [1, 2, 3, 4, 5],
    'value2': [2, 4, 6, 8, 10]
})

engineered_features = feature_pipeline.execute(df)
```

### Example 6: Data Validation Pipeline

```python
class SchemaValidationStage(PipelineStage):
    def __init__(self, expected_columns):
        self.expected_columns = expected_columns

    def process(self, data):
        missing_cols = set(self.expected_columns) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        return data

class DataTypeValidationStage(PipelineStage):
    def __init__(self, dtype_map):
        self.dtype_map = dtype_map

    def process(self, data):
        for col, expected_dtype in self.dtype_map.items():
            if col in data.columns:
                try:
                    data[col] = data[col].astype(expected_dtype)
                except:
                    raise ValueError(f"Cannot convert {col} to {expected_dtype}")
        return data

class RangeValidationStage(PipelineStage):
    def __init__(self, range_map):
        self.range_map = range_map

    def process(self, data):
        for col, (min_val, max_val) in self.range_map.items():
            if col in data.columns:
                out_of_range = (data[col] < min_val) | (data[col] > max_val)
                if out_of_range.any():
                    raise ValueError(f"Values in {col} are out of range [{min_val}, {max_val}]")
        return data

# Usage
validation_pipeline = DataPipeline()
validation_pipeline.add_stage(SchemaValidationStage(['age', 'income'])) \
                   .add_stage(DataTypeValidationStage({'age': 'int', 'income': 'float'})) \
                   .add_stage(RangeValidationStage({'age': (0, 120), 'income': (0, 1000000)}))

df = pd.DataFrame({
    'age': [25, 30, 35],
    'income': [50000.0, 60000.0, 70000.0]
})

validated_data = validation_pipeline.execute(df)
```

### Example 7: Time Series Processing Pipeline

```python
class TimeSeriesResamplingStage(PipelineStage):
    def __init__(self, freq='D'):
        self.freq = freq

    def process(self, data):
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data = data.set_index('timestamp')
        return data.resample(self.freq).mean()

class MovingAverageStage(PipelineStage):
    def __init__(self, window=7):
        self.window = window

    def process(self, data):
        for col in data.columns:
            data[f'{col}_ma'] = data[col].rolling(window=self.window).mean()
        return data

class SeasonalDecomposeStage(PipelineStage):
    def process(self, data):
        from statsmodels.tsa.seasonal import seasonal_decompose
        for col in data.columns:
            if not col.endswith('_ma'):
                decomposition = seasonal_decompose(data[col].dropna(), model='additive')
                data[f'{col}_trend'] = decomposition.trend
                data[f'{col}_seasonal'] = decomposition.seasonal
                data[f'{col}_residual'] = decomposition.resid
        return data

# Usage
ts_pipeline = DataPipeline()
ts_pipeline.add_stage(TimeSeriesResamplingStage(freq='D')) \
           .add_stage(MovingAverageStage(window=7)) \
           .add_stage(SeasonalDecomposeStage())

# Sample time series data
dates = pd.date_range('2023-01-01', periods=100, freq='D')
ts_df = pd.DataFrame({
    'timestamp': dates,
    'value': np.random.randn(100).cumsum()
})

processed_ts = ts_pipeline.execute(ts_df)
```

### Example 8: Model Comparison Pipeline

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

class ModelTrainingStage(PipelineStage):
    def __init__(self, models):
        self.models = models

    def process(self, data):
        trained_models = {}
        for name, model in self.models.items():
            model.fit(data['X_train'], data['y_train'])
            trained_models[name] = model
        data['trained_models'] = trained_models
        return data

class ModelEvaluationStage(PipelineStage):
    def process(self, data):
        results = {}
        for name, model in data['trained_models'].items():
            predictions = model.predict(data['X_test'])
            results[name] = {
                'accuracy': accuracy_score(data['y_test'], predictions),
                'precision': precision_score(data['y_test'], predictions, average='weighted'),
                'recall': recall_score(data['y_test'], predictions, average='weighted')
            }
        data['evaluation_results'] = results
        return data

class BestModelSelectionStage(PipelineStage):
    def __init__(self, metric='accuracy'):
        self.metric = metric

    def process(self, data):
        best_model_name = max(
            data['evaluation_results'].keys(),
            key=lambda x: data['evaluation_results'][x][self.metric]
        )
        data['best_model'] = data['trained_models'][best_model_name]
        data['best_model_name'] = best_model_name
        return data

# Usage
models = {
    'rf': RandomForestClassifier(n_estimators=100),
    'gb': GradientBoostingClassifier(n_estimators=100),
    'lr': LogisticRegression(max_iter=1000)
}

model_comparison_pipeline = DataPipeline()
model_comparison_pipeline.add_stage(TrainTestSplitStage()) \
                         .add_stage(ModelTrainingStage(models)) \
                         .add_stage(ModelEvaluationStage()) \
                         .add_stage(BestModelSelectionStage(metric='accuracy'))

# Sample data
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2)
df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
df['target'] = y

results = model_comparison_pipeline.execute(df)
print(f"Best model: {results['best_model_name']}")
print(f"Results: {results['evaluation_results']}")
```

---

## Summary

These design patterns provide comprehensive solutions for data engineering and data science workflows:

- **Factory Pattern** (8 examples): Creates appropriate objects based on type - data sources, models, databases, transformers, visualizations, validators, feature engineers
- **Singleton Pattern** (7 examples): Manages shared resources - database connections, configuration, logging, caching, model registry, resource pools, metrics
- **Decorator Pattern** (8 examples): Adds functionality - execution timing, validation, caching, error handling, performance monitoring, retry logic, I/O logging, type conversion
- **Strategy Pattern** (8 examples): Interchangeable algorithms - data scaling, feature selection, missing values, model evaluation, data sampling, export formats, outlier detection, text processing
- **Pipeline Pattern** (8 examples): Sequential processing - data preprocessing, ML modeling, ETL, text processing, feature engineering, data validation, time series, model comparison

Each pattern includes multiple real-world examples that solve common challenges in data science projects while maintaining clean, maintainable, and reusable code architecture.
