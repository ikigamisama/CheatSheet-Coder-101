# Design Patterns Cheat Sheet for Data Analytics, Engineering & Science

## I. CREATIONAL DESIGN PATTERNS

_Patterns that deal with object creation mechanisms_

### 1. Singleton

**Purpose:** Ensures a class has only one instance and provides a global point of access to it.

**When to Use:** Database connections, configuration managers, logging systems.

**Data Context:** Use when you need a single shared resource across your data pipeline.

**Examples:**

```python
# Example 1: Database Connection Manager
class DatabaseConnection:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.connection = cls._create_connection()
        return cls._instance

    @staticmethod
    def _create_connection():
        # Establish database connection
        return "Connected to PostgreSQL"

# Usage
db1 = DatabaseConnection()
db2 = DatabaseConnection()
# db1 and db2 are the same instance
```

```python
# Example 2: Configuration Manager for ML Pipeline
class MLConfig:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.config = {
                'model_path': '/models/',
                'batch_size': 32,
                'learning_rate': 0.001
            }
        return cls._instance

config = MLConfig()
```

```python
# Example 3: Logger for Data Pipeline
class PipelineLogger:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.logs = []
        return cls._instance

    def log(self, message):
        self.logs.append(message)

logger = PipelineLogger()
logger.log("ETL started")
```

---

### 2. Factory Method

**Purpose:** Defines an interface for creating objects, but lets subclasses decide which class to instantiate.

**When to Use:** When you need to create different types of data processors, connectors, or models based on conditions.

**Data Context:** Creating different data readers (CSV, JSON, Parquet) or model types dynamically.

**Examples:**

```python
# Example 1: Data Reader Factory
class DataReader:
    def read(self, filepath):
        pass

class CSVReader(DataReader):
    def read(self, filepath):
        return f"Reading CSV from {filepath}"

class JSONReader(DataReader):
    def read(self, filepath):
        return f"Reading JSON from {filepath}"

class ParquetReader(DataReader):
    def read(self, filepath):
        return f"Reading Parquet from {filepath}"

def data_reader_factory(file_type):
    readers = {
        'csv': CSVReader(),
        'json': JSONReader(),
        'parquet': ParquetReader()
    }
    return readers.get(file_type)

# Usage
reader = data_reader_factory('csv')
data = reader.read('sales.csv')
```

```python
# Example 2: Model Factory for Different Algorithms
class Model:
    def train(self, data):
        pass

class LinearRegressionModel(Model):
    def train(self, data):
        return "Training Linear Regression"

class RandomForestModel(Model):
    def train(self, data):
        return "Training Random Forest"

def model_factory(model_type):
    models = {
        'linear': LinearRegressionModel(),
        'rf': RandomForestModel()
    }
    return models.get(model_type)
```

```python
# Example 3: Database Connector Factory
class DBConnector:
    def connect(self):
        pass

class PostgreSQLConnector(DBConnector):
    def connect(self):
        return "Connected to PostgreSQL"

class MongoDBConnector(DBConnector):
    def connect(self):
        return "Connected to MongoDB"

def db_connector_factory(db_type):
    connectors = {
        'postgres': PostgreSQLConnector(),
        'mongo': MongoDBConnector()
    }
    return connectors.get(db_type)
```

---

### 3. Abstract Factory

**Purpose:** Provides an interface for creating families of related objects without specifying their concrete classes.

**When to Use:** When you need to create multiple related objects that work together (e.g., cloud provider services).

**Data Context:** Creating complete data pipeline components for different cloud environments.

**Examples:**

```python
# Example 1: Cloud Data Platform Factory
class DataWarehouse:
    def query(self, sql):
        pass

class StorageService:
    def upload(self, data):
        pass

class AWSWarehouse(DataWarehouse):
    def query(self, sql):
        return f"Querying Redshift: {sql}"

class AWSStorage(StorageService):
    def upload(self, data):
        return f"Uploading to S3: {data}"

class GCPWarehouse(DataWarehouse):
    def query(self, sql):
        return f"Querying BigQuery: {sql}"

class GCPStorage(StorageService):
    def upload(self, data):
        return f"Uploading to GCS: {data}"

class CloudFactory:
    def create_warehouse(self):
        pass
    def create_storage(self):
        pass

class AWSFactory(CloudFactory):
    def create_warehouse(self):
        return AWSWarehouse()
    def create_storage(self):
        return AWSStorage()

class GCPFactory(CloudFactory):
    def create_warehouse(self):
        return GCPWarehouse()
    def create_storage(self):
        return GCPStorage()

# Usage
factory = AWSFactory()
warehouse = factory.create_warehouse()
storage = factory.create_storage()
```

```python
# Example 2: ML Framework Factory
class Trainer:
    def fit(self, model, data):
        pass

class Predictor:
    def predict(self, model, data):
        pass

class TensorFlowTrainer(Trainer):
    def fit(self, model, data):
        return "Training with TensorFlow"

class TensorFlowPredictor(Predictor):
    def predict(self, model, data):
        return "Predicting with TensorFlow"

class PyTorchTrainer(Trainer):
    def fit(self, model, data):
        return "Training with PyTorch"

class PyTorchPredictor(Predictor):
    def predict(self, model, data):
        return "Predicting with PyTorch"

class MLFrameworkFactory:
    def create_trainer(self):
        pass
    def create_predictor(self):
        pass

class TensorFlowFactory(MLFrameworkFactory):
    def create_trainer(self):
        return TensorFlowTrainer()
    def create_predictor(self):
        return TensorFlowPredictor()
```

```python
# Example 3: Data Processing Environment Factory
class DataProcessor:
    def process(self, data):
        pass

class DataValidator:
    def validate(self, data):
        pass

class BatchProcessor(DataProcessor):
    def process(self, data):
        return "Batch processing with Spark"

class BatchValidator(DataValidator):
    def validate(self, data):
        return "Batch validation"

class StreamProcessor(DataProcessor):
    def process(self, data):
        return "Stream processing with Kafka"

class StreamValidator(DataValidator):
    def validate(self, data):
        return "Stream validation"

class ProcessingFactory:
    def create_processor(self):
        pass
    def create_validator(self):
        pass

class BatchFactory(ProcessingFactory):
    def create_processor(self):
        return BatchProcessor()
    def create_validator(self):
        return BatchValidator()
```

---

### 4. Builder

**Purpose:** Separates the construction of a complex object from its representation, allowing the same construction process to create different representations.

**When to Use:** When creating complex objects with many optional parameters (e.g., ML pipelines, queries).

**Data Context:** Building data processing pipelines, complex SQL queries, or ML model configurations.

**Examples:**

```python
# Example 1: ML Pipeline Builder
class MLPipeline:
    def __init__(self):
        self.preprocessor = None
        self.feature_selector = None
        self.model = None
        self.postprocessor = None

    def __str__(self):
        return f"Pipeline: {self.preprocessor} -> {self.feature_selector} -> {self.model} -> {self.postprocessor}"

class MLPipelineBuilder:
    def __init__(self):
        self.pipeline = MLPipeline()

    def add_preprocessor(self, preprocessor):
        self.pipeline.preprocessor = preprocessor
        return self

    def add_feature_selector(self, selector):
        self.pipeline.feature_selector = selector
        return self

    def add_model(self, model):
        self.pipeline.model = model
        return self

    def add_postprocessor(self, postprocessor):
        self.pipeline.postprocessor = postprocessor
        return self

    def build(self):
        return self.pipeline

# Usage
pipeline = (MLPipelineBuilder()
            .add_preprocessor("StandardScaler")
            .add_feature_selector("SelectKBest")
            .add_model("RandomForest")
            .add_postprocessor("Calibration")
            .build())
```

```python
# Example 2: SQL Query Builder
class SQLQuery:
    def __init__(self):
        self.select_clause = []
        self.from_clause = None
        self.where_clause = []
        self.group_by_clause = []
        self.order_by_clause = []

    def build_query(self):
        query = f"SELECT {', '.join(self.select_clause)}"
        query += f" FROM {self.from_clause}"
        if self.where_clause:
            query += f" WHERE {' AND '.join(self.where_clause)}"
        if self.group_by_clause:
            query += f" GROUP BY {', '.join(self.group_by_clause)}"
        if self.order_by_clause:
            query += f" ORDER BY {', '.join(self.order_by_clause)}"
        return query

class QueryBuilder:
    def __init__(self):
        self.query = SQLQuery()

    def select(self, *columns):
        self.query.select_clause.extend(columns)
        return self

    def from_table(self, table):
        self.query.from_clause = table
        return self

    def where(self, condition):
        self.query.where_clause.append(condition)
        return self

    def group_by(self, *columns):
        self.query.group_by_clause.extend(columns)
        return self

    def order_by(self, *columns):
        self.query.order_by_clause.extend(columns)
        return self

    def build(self):
        return self.query.build_query()

# Usage
query = (QueryBuilder()
         .select("customer_id", "SUM(amount)")
         .from_table("sales")
         .where("date >= '2024-01-01'")
         .group_by("customer_id")
         .order_by("SUM(amount) DESC")
         .build())
```

```python
# Example 3: Data Transformation Pipeline Builder
class DataPipeline:
    def __init__(self):
        self.transformations = []
        self.source = None
        self.destination = None

    def execute(self):
        return f"Pipeline: {self.source} -> {self.transformations} -> {self.destination}"

class DataPipelineBuilder:
    def __init__(self):
        self.pipeline = DataPipeline()

    def from_source(self, source):
        self.pipeline.source = source
        return self

    def add_transformation(self, transformation):
        self.pipeline.transformations.append(transformation)
        return self

    def to_destination(self, destination):
        self.pipeline.destination = destination
        return self

    def build(self):
        return self.pipeline

# Usage
pipeline = (DataPipelineBuilder()
            .from_source("PostgreSQL")
            .add_transformation("Clean nulls")
            .add_transformation("Normalize")
            .add_transformation("Aggregate")
            .to_destination("Redshift")
            .build())
```

---

### 5. Prototype

**Purpose:** Creates new objects by copying an existing object (prototype).

**When to Use:** When object creation is expensive or complex, and you need similar objects.

**Data Context:** Cloning trained models, duplicating data processing configurations, or creating similar datasets.

**Examples:**

```python
# Example 1: Model Configuration Cloning
import copy

class ModelConfig:
    def __init__(self, learning_rate, batch_size, epochs, optimizer):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optimizer

    def clone(self):
        return copy.deepcopy(self)

    def __str__(self):
        return f"LR: {self.learning_rate}, BS: {self.batch_size}, Epochs: {self.epochs}, Opt: {self.optimizer}"

# Usage
base_config = ModelConfig(0.001, 32, 100, "Adam")
experiment1 = base_config.clone()
experiment1.learning_rate = 0.01

experiment2 = base_config.clone()
experiment2.batch_size = 64
```

```python
# Example 2: Dataset Template Cloning
class DatasetConfig:
    def __init__(self, source, filters, transformations):
        self.source = source
        self.filters = filters.copy()
        self.transformations = transformations.copy()

    def clone(self):
        return DatasetConfig(
            self.source,
            self.filters.copy(),
            self.transformations.copy()
        )

    def add_filter(self, filter_condition):
        self.filters.append(filter_condition)

# Usage
base_dataset = DatasetConfig(
    source="sales_db",
    filters=["date >= '2024-01-01'"],
    transformations=["remove_nulls", "normalize"]
)

january_data = base_dataset.clone()
january_data.add_filter("month = 1")

february_data = base_dataset.clone()
february_data.add_filter("month = 2")
```

```python
# Example 3: ETL Job Template
class ETLJob:
    def __init__(self, name, extract_query, transform_steps, load_target):
        self.name = name
        self.extract_query = extract_query
        self.transform_steps = transform_steps.copy()
        self.load_target = load_target

    def clone(self):
        return ETLJob(
            self.name + "_copy",
            self.extract_query,
            self.transform_steps.copy(),
            self.load_target
        )

# Usage
daily_job = ETLJob(
    name="daily_sales",
    extract_query="SELECT * FROM sales WHERE date = CURRENT_DATE",
    transform_steps=["clean", "aggregate", "enrich"],
    load_target="warehouse.sales_daily"
)

weekly_job = daily_job.clone()
weekly_job.name = "weekly_sales"
weekly_job.extract_query = "SELECT * FROM sales WHERE date >= CURRENT_DATE - 7"
```

---

## II. STRUCTURAL DESIGN PATTERNS

_Patterns that deal with object composition and relationships_

### 1. Adapter

**Purpose:** Allows incompatible interfaces to work together by wrapping one interface to match another.

**When to Use:** When integrating legacy systems, third-party libraries, or different data formats.

**Data Context:** Adapting different data sources to a common interface or converting between data formats.

**Examples:**

```python
# Example 1: Data Source Adapter
class ModernDataAPI:
    def get_data(self):
        return {"data": [1, 2, 3, 4, 5], "format": "json"}

class LegacyDataSystem:
    def fetch_records(self):
        return "1,2,3,4,5"  # Returns CSV string

class LegacyDataAdapter:
    def __init__(self, legacy_system):
        self.legacy_system = legacy_system

    def get_data(self):
        csv_data = self.legacy_system.fetch_records()
        data_list = [int(x) for x in csv_data.split(',')]
        return {"data": data_list, "format": "json"}

# Usage
legacy = LegacyDataSystem()
adapter = LegacyDataAdapter(legacy)
data = adapter.get_data()  # Now compatible with ModernDataAPI interface
```

```python
# Example 2: ML Framework Adapter
class ScikitLearnModel:
    def fit(self, X, y):
        return "Scikit-learn training"

    def predict(self, X):
        return "Scikit-learn predictions"

class TensorFlowModel:
    def train(self, features, labels, epochs):
        return "TensorFlow training"

    def inference(self, features):
        return "TensorFlow predictions"

class TensorFlowAdapter:
    def __init__(self, tf_model):
        self.tf_model = tf_model

    def fit(self, X, y):
        return self.tf_model.train(X, y, epochs=10)

    def predict(self, X):
        return self.tf_model.inference(X)

# Usage - now both models have the same interface
sklearn_model = ScikitLearnModel()
tf_model = TensorFlowModel()
tf_adapted = TensorFlowAdapter(tf_model)

# Both can be used identically
sklearn_model.fit(X_train, y_train)
tf_adapted.fit(X_train, y_train)
```

```python
# Example 3: Database Result Adapter
class PostgreSQLResult:
    def __init__(self, rows):
        self.rows = rows

    def fetch_all(self):
        return self.rows

class MongoDBResult:
    def __init__(self, cursor):
        self.cursor = cursor

    def get_documents(self):
        return list(self.cursor)

class MongoDBResultAdapter:
    def __init__(self, mongo_result):
        self.mongo_result = mongo_result

    def fetch_all(self):
        return self.mongo_result.get_documents()

# Usage - unified interface for different databases
pg_result = PostgreSQLResult([{"id": 1}, {"id": 2}])
mongo_result = MongoDBResult([{"_id": 1}, {"_id": 2}])
mongo_adapted = MongoDBResultAdapter(mongo_result)

# Both can be accessed the same way
data1 = pg_result.fetch_all()
data2 = mongo_adapted.fetch_all()
```

---

### 2. Bridge

**Purpose:** Separates abstraction from implementation so they can vary independently.

**When to Use:** When you want to avoid permanent binding between abstraction and implementation.

**Data Context:** Separating data processing logic from storage mechanisms or visualization from data sources.

**Examples:**

```python
# Example 1: Data Visualization Bridge
class DataSource:
    def get_data(self):
        pass

class SQLDataSource(DataSource):
    def get_data(self):
        return [10, 20, 30, 40, 50]

class APIDataSource(DataSource):
    def get_data(self):
        return [15, 25, 35, 45, 55]

class Visualization:
    def __init__(self, data_source):
        self.data_source = data_source

    def render(self):
        pass

class BarChart(Visualization):
    def render(self):
        data = self.data_source.get_data()
        return f"Bar chart with data: {data}"

class LineChart(Visualization):
    def render(self):
        data = self.data_source.get_data()
        return f"Line chart with data: {data}"

# Usage - any visualization can work with any data source
sql_source = SQLDataSource()
api_source = APIDataSource()

bar_from_sql = BarChart(sql_source)
line_from_api = LineChart(api_source)
bar_from_api = BarChart(api_source)
```

```python
# Example 2: Report Generation Bridge
class DataFetcher:
    def fetch(self):
        pass

class DatabaseFetcher(DataFetcher):
    def fetch(self):
        return {"sales": 10000, "customers": 500}

class APIFetcher(DataFetcher):
    def fetch(self):
        return {"sales": 12000, "customers": 600}

class Report:
    def __init__(self, fetcher):
        self.fetcher = fetcher

    def generate(self):
        pass

class PDFReport(Report):
    def generate(self):
        data = self.fetcher.fetch()
        return f"PDF Report: {data}"

class ExcelReport(Report):
    def generate(self):
        data = self.fetcher.fetch()
        return f"Excel Report: {data}"

# Usage
db_fetcher = DatabaseFetcher()
api_fetcher = APIFetcher()

pdf_from_db = PDFReport(db_fetcher)
excel_from_api = ExcelReport(api_fetcher)
```

```python
# Example 3: ML Model Training Bridge
class DataLoader:
    def load(self):
        pass

class CSVLoader(DataLoader):
    def load(self):
        return "CSV data loaded"

class ParquetLoader(DataLoader):
    def load(self):
        return "Parquet data loaded"

class ModelTrainer:
    def __init__(self, data_loader):
        self.data_loader = data_loader

    def train(self):
        pass

class SupervisedTrainer(ModelTrainer):
    def train(self):
        data = self.data_loader.load()
        return f"Supervised training with {data}"

class UnsupervisedTrainer(ModelTrainer):
    def train(self):
        data = self.data_loader.load()
        return f"Unsupervised training with {data}"

# Usage
csv_loader = CSVLoader()
parquet_loader = ParquetLoader()

supervised_csv = SupervisedTrainer(csv_loader)
unsupervised_parquet = UnsupervisedTrainer(parquet_loader)
```

---

### 3. Composite

**Purpose:** Composes objects into tree structures to represent part-whole hierarchies, allowing individual objects and compositions to be treated uniformly.

**When to Use:** When you need to represent hierarchical data structures.

**Data Context:** Building nested data pipelines, hierarchical data structures, or complex feature engineering pipelines.

**Examples:**

```python
# Example 1: Data Transformation Pipeline
class DataTransformation:
    def apply(self, data):
        pass

class SimpleTransformation(DataTransformation):
    def __init__(self, name, operation):
        self.name = name
        self.operation = operation

    def apply(self, data):
        return f"Applying {self.name} on {data}"

class CompositeTransformation(DataTransformation):
    def __init__(self, name):
        self.name = name
        self.transformations = []

    def add(self, transformation):
        self.transformations.append(transformation)

    def apply(self, data):
        result = data
        for transform in self.transformations:
            result = transform.apply(result)
        return result

# Usage
# Individual transformations
remove_nulls = SimpleTransformation("RemoveNulls", lambda x: x)
normalize = SimpleTransformation("Normalize", lambda x: x)
scale = SimpleTransformation("Scale", lambda x: x)

# Composite transformation
preprocessing = CompositeTransformation("Preprocessing")
preprocessing.add(remove_nulls)
preprocessing.add(normalize)

feature_engineering = CompositeTransformation("FeatureEngineering")
feature_engineering.add(preprocessing)
feature_engineering.add(scale)

# Apply entire pipeline
result = feature_engineering.apply("raw_data")
```

```python
# Example 2: Organizational Data Hierarchy
class DataComponent:
    def get_size(self):
        pass

class DataFile(DataComponent):
    def __init__(self, name, size):
        self.name = name
        self.size = size

    def get_size(self):
        return self.size

class DataFolder(DataComponent):
    def __init__(self, name):
        self.name = name
        self.children = []

    def add(self, component):
        self.children.append(component)

    def get_size(self):
        return sum(child.get_size() for child in self.children)

# Usage
# Individual files
sales_csv = DataFile("sales.csv", 100)
customers_csv = DataFile("customers.csv", 50)
products_json = DataFile("products.json", 30)

# Folders
data_folder = DataFolder("data")
data_folder.add(sales_csv)
data_folder.add(customers_csv)

raw_folder = DataFolder("raw")
raw_folder.add(data_folder)
raw_folder.add(products_json)

total_size = raw_folder.get_size()  # 180
```

```python
# Example 3: Feature Engineering Tree
class Feature:
    def compute(self, data):
        pass

class BaseFeature(Feature):
    def __init__(self, name, column):
        self.name = name
        self.column = column

    def compute(self, data):
        return f"Computing {self.name} from {self.column}"

class CompositeFeature(Feature):
    def __init__(self, name):
        self.name = name
        self.features = []

    def add(self, feature):
        self.features.append(feature)

    def compute(self, data):
        results = [f.compute(data) for f in self.features]
        return f"Composite {self.name}: {results}"

# Usage
age = BaseFeature("age", "birth_date")
income = BaseFeature("income", "salary")
age_squared = BaseFeature("age_squared", "age")

demographic_features = CompositeFeature("Demographics")
demographic_features.add(age)
demographic_features.add(income)
demographic_features.add(age_squared)

result = demographic_features.compute("customer_data")
```

---

### 4. Decorator

**Purpose:** Adds new functionality to objects dynamically without altering their structure.

**When to Use:** When you want to add responsibilities to objects without subclassing.

**Data Context:** Adding logging, caching, validation, or monitoring to data operations.

**Examples:**

```python
# Example 1: Data Pipeline with Logging
class DataPipeline:
    def process(self, data):
        return f"Processing {data}"

class LoggingDecorator:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def process(self, data):
        print(f"[LOG] Starting processing: {data}")
        result = self.pipeline.process(data)
        print(f"[LOG] Completed processing")
        return result

class CachingDecorator:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.cache = {}

    def process(self, data):
        if data in self.cache:
            print(f"[CACHE] Returning cached result for {data}")
            return self.cache[data]
        result = self.pipeline.process(data)
        self.cache[data] = result
        return result

# Usage
pipeline = DataPipeline()
logged_pipeline = LoggingDecorator(pipeline)
cached_logged_pipeline = CachingDecorator(logged_pipeline)

result = cached_logged_pipeline.process("sales_data")
```

```python
# Example 2: Model Prediction with Monitoring
class Model:
    def predict(self, features):
        return [0.8, 0.2]

class PerformanceMonitorDecorator:
    def __init__(self, model):
        self.model = model

    def predict(self, features):
        import time
        start = time.time()
        result = self.model.predict(features)
        elapsed = time.time() - start
        print(f"[PERF] Prediction took {elapsed:.4f} seconds")
        return result

class ValidationDecorator:
    def __init__(self, model):
        self.model = model

    def predict(self, features):
        if features is None or len(features) == 0:
            raise ValueError("Features cannot be empty")
        return self.model.predict(features)

# Usage
model = Model()
monitored_model = PerformanceMonitorDecorator(model)
validated_monitored_model = ValidationDecorator(monitored_model)

predictions = validated_monitored_model.predict([1, 2, 3])
```

```python
# Example 3: Data Query with Auditing
class DataQuery:
    def execute(self, query):
        return f"Executing: {query}"

class AuditDecorator:
    def __init__(self, query_executor):
        self.query_executor = query_executor
        self.audit_log = []

    def execute(self, query):
        self.audit_log.append(f"Query executed: {query}")
        return self.query_executor.execute(query)

class RetryDecorator:
    def __init__(self, query_executor, max_retries=3):
        self.query_executor = query_executor
        self.max_retries = max_retries

    def execute(self, query):
        for attempt in range(self.max_retries):
            try:
                return self.query_executor.execute(query)
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
        raise Exception("Max retries exceeded")

# Usage
query = DataQuery()
audited_query = AuditDecorator(query)
retry_audited_query = RetryDecorator(audited_query)

result = retry_audited_query.execute("SELECT * FROM sales")
```

---

### 5. Facade

**Purpose:** Provides a simplified interface to a complex subsystem.

**When to Use:** When you want to hide complexity and provide a simple API.

**Data Context:** Simplifying complex data operations, ML workflows, or multi-step ETL processes.

**Examples:**

```python
# Example 1: ML Training Facade
class DataLoader:
    def load_data(self, path):
        return f"Data loaded from {path}"

class DataPreprocessor:
    def preprocess(self, data):
        return f"Preprocessed {data}"

class FeatureEngineering:
    def create_features(self, data):
        return f"Features created from {data}"

class ModelTrainer:
    def train(self, features, labels):
        return "Model trained"

class ModelEvaluator:
    def evaluate(self, model, test_data):
        return "Accuracy: 0.95"

class MLPipelineFacade:
    def __init__(self):
        self.loader = DataLoader()
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineering()
        self.trainer = ModelTrainer()
        self.evaluator = ModelEvaluator()

    def train_model(self, data_path):
        # Simplified interface for complex workflow
        data = self.loader.load_data(data_path)
        clean_data = self.preprocessor.preprocess(data)
```
