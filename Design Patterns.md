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
        features = self.feature_engineer.create_features(clean_data)
        model = self.trainer.train(features, "labels")
        evaluation = self.evaluator.evaluate(model, "test_data")
        return evaluation

# Usage - Simple interface hides complexity
pipeline = MLPipelineFacade()
result = pipeline.train_model("data/training.csv")
```

```python
# Example 2: ETL Pipeline Facade
class DataExtractor:
    def extract_from_db(self, query):
        return f"Extracted data with {query}"

class DataTransformer:
    def clean_data(self, data):
        return f"Cleaned {data}"

    def transform_data(self, data):
        return f"Transformed {data}"

class DataValidator:
    def validate(self, data):
        return f"Validated {data}"

class DataLoader:
    def load_to_warehouse(self, data, table):
        return f"Loaded {data} to {table}"

class ETLFacade:
    def __init__(self):
        self.extractor = DataExtractor()
        self.transformer = DataTransformer()
        self.validator = DataValidator()
        self.loader = DataLoader()

    def run_etl(self, source_query, target_table):
        # Complex ETL process simplified
        raw_data = self.extractor.extract_from_db(source_query)
        clean_data = self.transformer.clean_data(raw_data)
        transformed_data = self.transformer.transform_data(clean_data)
        validated_data = self.validator.validate(transformed_data)
        result = self.loader.load_to_warehouse(validated_data, target_table)
        return f"ETL completed: {result}"

# Usage
etl = ETLFacade()
etl.run_etl("SELECT * FROM sales", "warehouse.sales")
```

```python
# Example 3: Data Analytics Facade
class DataRetriever:
    def get_data(self, source):
        return [100, 200, 300, 400, 500]

class StatisticalAnalyzer:
    def calculate_mean(self, data):
        return sum(data) / len(data)

    def calculate_median(self, data):
        sorted_data = sorted(data)
        return sorted_data[len(data) // 2]

    def calculate_std(self, data):
        return 50.0

class Visualizer:
    def create_chart(self, data, chart_type):
        return f"{chart_type} created with {data}"

class ReportGenerator:
    def generate(self, stats, chart):
        return f"Report: {stats}, Chart: {chart}"

class AnalyticsFacade:
    def __init__(self):
        self.retriever = DataRetriever()
        self.analyzer = StatisticalAnalyzer()
        self.visualizer = Visualizer()
        self.reporter = ReportGenerator()

    def create_analysis_report(self, data_source):
        # Simplified analytics workflow
        data = self.retriever.get_data(data_source)
        mean = self.analyzer.calculate_mean(data)
        median = self.analyzer.calculate_median(data)
        std = self.analyzer.calculate_std(data)
        stats = {"mean": mean, "median": median, "std": std}
        chart = self.visualizer.create_chart(data, "histogram")
        report = self.reporter.generate(stats, chart)
        return report

# Usage
analytics = AnalyticsFacade()
report = analytics.create_analysis_report("sales_database")
```

---

### 6. Flyweight

**Purpose:** Reduces memory usage by sharing common data between multiple objects.

**When to Use:** When you need to create a large number of similar objects.

**Data Context:** Handling large datasets with repeated values, caching common features, or managing metadata.

**Examples:**

```python
# Example 1: Feature Value Flyweight
class FeatureValue:
    def __init__(self, value, data_type):
        self.value = value
        self.data_type = data_type

    def __str__(self):
        return f"{self.value} ({self.data_type})"

class FeatureFactory:
    _features = {}

    @classmethod
    def get_feature(cls, value, data_type):
        key = (value, data_type)
        if key not in cls._features:
            cls._features[key] = FeatureValue(value, data_type)
        return cls._features[key]

    @classmethod
    def get_total_features(cls):
        return len(cls._features)

# Usage - Millions of records but only unique values stored
factory = FeatureFactory()

# Even with millions of records, only 3 objects created
records = []
for i in range(1000000):
    gender = factory.get_feature("Male", "categorical")
    status = factory.get_feature("Active", "categorical")
    country = factory.get_feature("USA", "categorical")
    records.append((gender, status, country))

print(f"Total unique features created: {factory.get_total_features()}")  # Only 3
```

```python
# Example 2: ML Model Metadata Flyweight
class ModelMetadata:
    def __init__(self, model_type, version, framework):
        self.model_type = model_type
        self.version = version
        self.framework = framework
        print(f"Created metadata: {model_type}-{version}-{framework}")

class ModelMetadataFactory:
    _metadata = {}

    @classmethod
    def get_metadata(cls, model_type, version, framework):
        key = (model_type, version, framework)
        if key not in cls._metadata:
            cls._metadata[key] = ModelMetadata(model_type, version, framework)
        return cls._metadata[key]

# Usage - Multiple model instances share metadata
factory = ModelMetadataFactory()

# Creating 1000 models but only 2 unique metadata objects
models = []
for i in range(500):
    meta1 = factory.get_metadata("RandomForest", "1.0", "sklearn")
    meta2 = factory.get_metadata("XGBoost", "2.0", "xgboost")
    models.append({"id": i, "metadata": meta1})
    models.append({"id": i+500, "metadata": meta2})

# Only 2 metadata objects created despite 1000 models
```

```python
# Example 3: Data Type Schema Flyweight
class ColumnSchema:
    def __init__(self, data_type, nullable, default):
        self.data_type = data_type
        self.nullable = nullable
        self.default = default

class SchemaFactory:
    _schemas = {}

    @classmethod
    def get_schema(cls, data_type, nullable=True, default=None):
        key = (data_type, nullable, default)
        if key not in cls._schemas:
            cls._schemas[key] = ColumnSchema(data_type, nullable, default)
        return cls._schemas[key]

# Usage - Many columns share same schema definitions
factory = SchemaFactory()

# Define table with 1000 columns
columns = {}
for i in range(333):
    columns[f"col_int_{i}"] = factory.get_schema("INTEGER", True, 0)
    columns[f"col_str_{i}"] = factory.get_schema("VARCHAR", True, "")
    columns[f"col_float_{i}"] = factory.get_schema("FLOAT", True, 0.0)

# Only 3 schema objects created for 999 columns
print(f"Columns: {len(columns)}, Unique schemas: {len(factory._schemas)}")
```

---

### 7. Proxy

**Purpose:** Provides a surrogate or placeholder for another object to control access to it.

**When to Use:** When you need lazy loading, access control, logging, or caching.

**Data Context:** Lazy loading large datasets, controlling database access, or caching expensive computations.

**Examples:**

```python
# Example 1: Lazy Loading Data Proxy
class ExpensiveDataset:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None
        print(f"Loading large dataset from {filepath}...")
        # Simulate expensive loading
        self.data = [i for i in range(1000000)]
        print("Dataset loaded!")

    def get_data(self):
        return self.data

class DatasetProxy:
    def __init__(self, filepath):
        self.filepath = filepath
        self._dataset = None

    def get_data(self):
        # Lazy loading - only load when actually needed
        if self._dataset is None:
            print("First access - loading dataset...")
            self._dataset = ExpensiveDataset(self.filepath)
        return self._dataset.get_data()

# Usage
proxy = DatasetProxy("large_file.csv")
print("Proxy created, but data not loaded yet")
# ... do other work ...
data = proxy.get_data()  # Data loaded only when accessed
```

```python
# Example 2: Access Control Proxy for Database
class Database:
    def query(self, sql):
        return f"Executing: {sql}"

    def delete(self, table):
        return f"Deleting from {table}"

class DatabaseProxy:
    def __init__(self, database, user_role):
        self.database = database
        self.user_role = user_role

    def query(self, sql):
        # All users can query
        print(f"[ACCESS] {self.user_role} querying database")
        return self.database.query(sql)

    def delete(self, table):
        # Only admins can delete
        if self.user_role != "admin":
            return "[ACCESS DENIED] Only admins can delete data"
        print(f"[ACCESS] {self.user_role} deleting from database")
        return self.database.delete(table)

# Usage
db = Database()
analyst_db = DatabaseProxy(db, "analyst")
admin_db = DatabaseProxy(db, "admin")

analyst_db.query("SELECT * FROM sales")  # Allowed
analyst_db.delete("sales")  # Denied

admin_db.delete("sales")  # Allowed
```

```python
# Example 3: Caching Proxy for ML Model
class MLModel:
    def predict(self, features):
        print("[MODEL] Running expensive prediction...")
        # Simulate expensive computation
        return sum(features) * 0.1

class CachingModelProxy:
    def __init__(self, model):
        self.model = model
        self.cache = {}

    def predict(self, features):
        # Convert to tuple for hashing
        cache_key = tuple(features)

        if cache_key in self.cache:
            print("[CACHE HIT] Returning cached prediction")
            return self.cache[cache_key]

        print("[CACHE MISS] Computing new prediction")
        result = self.model.predict(features)
        self.cache[cache_key] = result
        return result

# Usage
model = MLModel()
proxy = CachingModelProxy(model)

# First call - computes
result1 = proxy.predict([1, 2, 3])

# Second call with same features - cached
result2 = proxy.predict([1, 2, 3])

# Different features - computes
result3 = proxy.predict([4, 5, 6])
```

---

## III. BEHAVIORAL DESIGN PATTERNS

_Patterns that deal with object communication and responsibility distribution_

### 1. Chain of Responsibility

**Purpose:** Passes requests along a chain of handlers where each handler decides either to process the request or pass it to the next handler.

**When to Use:** When you have multiple handlers that can process a request and the handler isn't known in advance.

**Data Context:** Data validation pipelines, error handling, data quality checks, or multi-stage data processing.

**Examples:**

```python
# Example 1: Data Validation Chain
class DataValidator:
    def __init__(self):
        self.next_validator = None

    def set_next(self, validator):
        self.next_validator = validator
        return validator

    def validate(self, data):
        pass

class NullValidator(DataValidator):
    def validate(self, data):
        if data is None or "" in str(data):
            return {"valid": False, "error": "Null values found"}
        if self.next_validator:
            return self.next_validator.validate(data)
        return {"valid": True}

class RangeValidator(DataValidator):
    def validate(self, data):
        if isinstance(data, (int, float)) and (data < 0 or data > 100):
            return {"valid": False, "error": "Value out of range"}
        if self.next_validator:
            return self.next_validator.validate(data)
        return {"valid": True}

class TypeValidator(DataValidator):
    def validate(self, data):
        if not isinstance(data, (int, float)):
            return {"valid": False, "error": "Invalid type"}
        if self.next_validator:
            return self.next_validator.validate(data)
        return {"valid": True}

# Usage
null_check = NullValidator()
type_check = TypeValidator()
range_check = RangeValidator()

# Build chain
null_check.set_next(type_check).set_next(range_check)

# Validate data
print(null_check.validate(50))     # Valid
print(null_check.validate(150))    # Out of range
print(null_check.validate("abc"))  # Invalid type
```

```python
# Example 2: Log Level Handler Chain
class LogHandler:
    def __init__(self, level):
        self.level = level
        self.next_handler = None

    def set_next(self, handler):
        self.next_handler = handler
        return handler

    def handle(self, level, message):
        if level >= self.level:
            self.write(message)
        if self.next_handler:
            self.next_handler.handle(level, message)

    def write(self, message):
        pass

class ConsoleHandler(LogHandler):
    def write(self, message):
        print(f"[CONSOLE] {message}")

class FileHandler(LogHandler):
    def write(self, message):
        print(f"[FILE] Writing to log.txt: {message}")

class EmailHandler(LogHandler):
    def write(self, message):
        print(f"[EMAIL] Sending alert: {message}")

# Usage
INFO = 1
WARNING = 2
ERROR = 3

console = ConsoleHandler(INFO)
file = FileHandler(WARNING)
email = EmailHandler(ERROR)

# Build chain
console.set_next(file).set_next(email)

# Test logging
console.handle(INFO, "Pipeline started")
console.handle(WARNING, "High memory usage")
console.handle(ERROR, "Pipeline failed")
```

```python
# Example 3: Data Processing Pipeline Chain
class DataProcessor:
    def __init__(self):
        self.next_processor = None

    def set_next(self, processor):
        self.next_processor = processor
        return processor

    def process(self, data):
        pass

class DataCleaner(DataProcessor):
    def process(self, data):
        cleaned = [x for x in data if x is not None]
        print(f"[CLEANER] Removed nulls: {len(data) - len(cleaned)} rows")
        if self.next_processor:
            return self.next_processor.process(cleaned)
        return cleaned

class DataNormalizer(DataProcessor):
    def process(self, data):
        normalized = [x / max(data) if max(data) > 0 else 0 for x in data]
        print(f"[NORMALIZER] Normalized data")
        if self.next_processor:
            return self.next_processor.process(normalized)
        return normalized

class DataAggregator(DataProcessor):
    def process(self, data):
        aggregated = {"mean": sum(data)/len(data), "count": len(data)}
        print(f"[AGGREGATOR] Aggregated: {aggregated}")
        if self.next_processor:
            return self.next_processor.process(aggregated)
        return aggregated

# Usage
cleaner = DataCleaner()
normalizer = DataNormalizer()
aggregator = DataAggregator()

# Build processing chain
cleaner.set_next(normalizer).set_next(aggregator)

# Process data through chain
raw_data = [10, None, 20, 30, None, 40]
result = cleaner.process(raw_data)
```

---

### 2. Command

**Purpose:** Encapsulates a request as an object, allowing you to parameterize clients with different requests, queue requests, and support undoable operations.

**When to Use:** When you need to queue operations, implement undo/redo, or log operations.

**Data Context:** ETL job scheduling, data transformation history, model training experiments, or transaction logging.

**Examples:**

```python
# Example 1: Data Transformation Commands with Undo
class Command:
    def execute(self):
        pass

    def undo(self):
        pass

class TransformCommand(Command):
    def __init__(self, data, transformation):
        self.data = data
        self.transformation = transformation
        self.previous_state = None

    def execute(self):
        self.previous_state = self.data.copy()
        if self.transformation == "normalize":
            max_val = max(self.data)
            self.data[:] = [x / max_val for x in self.data]
        elif self.transformation == "square":
            self.data[:] = [x ** 2 for x in self.data]
        return self.data

    def undo(self):
        if self.previous_state:
            self.data[:] = self.previous_state
        return self.data

class TransformationManager:
    def __init__(self):
        self.history = []

    def execute_command(self, command):
        result = command.execute()
        self.history.append(command)
        return result

    def undo_last(self):
        if self.history:
            command = self.history.pop()
            return command.undo()
        return None

# Usage
data = [10, 20, 30, 40]
manager = TransformationManager()

print(f"Original: {data}")
manager.execute_command(TransformCommand(data, "normalize"))
print(f"After normalize: {data}")

manager.execute_command(TransformCommand(data, "square"))
print(f"After square: {data}")

manager.undo_last()
print(f"After undo: {data}")

manager.undo_last()
print(f"After second undo: {data}")
```

```python
# Example 2: ETL Job Commands
class ETLCommand(Command):
    def execute(self):
        pass

    def get_status(self):
        pass

class ExtractCommand(ETLCommand):
    def __init__(self, source):
        self.source = source
        self.status = "pending"

    def execute(self):
        self.status = "running"
        print(f"Extracting data from {self.source}")
        self.status = "completed"
        return f"Data from {self.source}"

    def get_status(self):
        return self.status

class TransformCommand(ETLCommand):
    def __init__(self, data, operation):
        self.data = data
        self.operation = operation
        self.status = "pending"

    def execute(self):
        self.status = "running"
        print(f"Transforming data: {self.operation}")
        self.status = "completed"
        return f"Transformed {self.data}"

    def get_status(self):
        return self.status

class LoadCommand(ETLCommand):
    def __init__(self, data, destination):
        self.data = data
        self.destination = destination
        self.status = "pending"

    def execute(self):
        self.status = "running"
        print(f"Loading {self.data} to {self.destination}")
        self.status = "completed"
        return f"Loaded to {self.destination}"

    def get_status(self):
        return self.status

class ETLJobQueue:
    def __init__(self):
        self.commands = []

    def add_command(self, command):
        self.commands.append(command)

    def execute_all(self):
        results = []
        for cmd in self.commands:
            result = cmd.execute()
            results.append(result)
        return results

# Usage
queue = ETLJobQueue()
queue.add_command(ExtractCommand("database"))
queue.add_command(TransformCommand("raw_data", "clean_nulls"))
queue.add_command(LoadCommand("clean_data", "warehouse"))

results = queue.execute_all()
```

```python
# Example 3: ML Experiment Commands
class ExperimentCommand(Command):
    def __init__(self, name):
        self.name = name
        self.result = None

    def execute(self):
        pass

    def get_result(self):
        return self.result

class TrainModelCommand(ExperimentCommand):
    def __init__(self, model_type, hyperparams):
        super().__init__(f"Train_{model_type}")
        self.model_type = model_type
        self.hyperparams = hyperparams

    def execute(self):
        print(f"Training {self.model_type} with {self.hyperparams}")
        self.result = {"accuracy": 0.85, "loss": 0.15}
        return self.result

class EvaluateModelCommand(ExperimentCommand):
    def __init__(self, model):
        super().__init__(f"Evaluate_{model}")
        self.model = model

    def execute(self):
        print(f"Evaluating {self.model}")
        self.result = {"precision": 0.87, "recall": 0.83}
        return self.result

class ExperimentRunner:
    def __init__(self):
        self.experiments = []
        self.results = []

    def add_experiment(self, command):
        self.experiments.append(command)

    def run_all_experiments(self):
        for exp in self.experiments:
            result = exp.execute()
            self.results.append({"name": exp.name, "result": result})
        return self.results

# Usage
runner = ExperimentRunner()
runner.add_experiment(TrainModelCommand("RandomForest", {"n_estimators": 100}))
runner.add_experiment(TrainModelCommand("XGBoost", {"max_depth": 5}))
runner.add_experiment(EvaluateModelCommand("RandomForest"))

all_results = runner.run_all_experiments()
```

---

### 3. Interpreter

**Purpose:** Defines a representation for a grammar and an interpreter to interpret sentences in that language.

**When to Use:** When you need to interpret or evaluate expressions, rules, or domain-specific languages.

**Data Context:** Query builders, business rule engines, data validation rules, or custom formula evaluators.

**Examples:**

```python
# Example 1: Data Filter Expression Interpreter
class Expression:
    def interpret(self, context):
        pass

class ColumnExpression(Expression):
    def __init__(self, column_name):
        self.column_name = column_name

    def interpret(self, context):
        return context.get(self.column_name)

class ValueExpression(Expression):
    def __init__(self, value):
        self.value = value

    def interpret(self, context):
        return self.value

class GreaterThanExpression(Expression):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def interpret(self, context):
        return self.left.interpret(context) > self.right.interpret(context)

class AndExpression(Expression):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def interpret(self, context):
        return self.left.interpret(context) and self.right.interpret(context)

# Usage - Build filter: age > 18 AND salary > 50000
age_check = GreaterThanExpression(
    ColumnExpression("age"),
    ValueExpression(18)
)

salary_check = GreaterThanExpression(
    ColumnExpression("salary"),
    ValueExpression(50000)
)

filter_rule = AndExpression(age_check, salary_check)

# Test with data
data1 = {"age": 25, "salary": 60000}
data2 = {"age": 16, "salary": 70000}

print(filter_rule.interpret(data1))  # True
print(filter_rule.interpret(data2))  # False
```

```python
# Example 2: Custom Metric Calculator
class MetricExpression:
    def calculate(self, data):
        pass

class SumMetric(MetricExpression):
    def __init__(self, column):
        self.column = column

    def calculate(self, data):
        return sum(row[self.column] for row in data)

class AverageMetric(MetricExpression):
    def __init__(self, column):
        self.column = column

    def calculate(self, data):
        values = [row[self.column] for row in data]
        return sum(values) / len(values) if values else 0

class DivideMetric(MetricExpression):
    def __init__(self, numerator, denominator):
        self.numerator = numerator
        self.denominator = denominator

    def calculate(self, data):
        num = self.numerator.calculate(data)
        denom = self.denominator.calculate(data)
        return num / denom if denom != 0 else 0

# Usage - Calculate average order value: total_sales / num_orders
sales_data = [
    {"sales": 100, "orders": 1},
    {"sales": 200, "orders": 2},
    {"sales": 300, "orders": 3}
]

total_sales = SumMetric("sales")
total_orders = SumMetric("orders")
avg_order_value = DivideMetric(total_sales, total_orders)

print(f"Average Order Value: {avg_order_value.calculate(sales_data)}")
```

```python
# Example 3: Feature Engineering Expression Language
class FeatureExpression:
    def compute(self, row):
        pass

class FieldExpression(FeatureExpression):
    def __init__(self, field_name):
        self.field_name = field_name

    def compute(self, row):
        return row.get(self.field_name, 0)

class AddExpression(FeatureExpression):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def compute(self, row):
        return self.left.compute(row) + self.right.compute(row)

class MultiplyExpression(FeatureExpression):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def compute(self, row):
        return self.left.compute(row) * self.right.compute(row)

class ConstantExpression(FeatureExpression):
    def __init__(self, value):
        self.value = value

    def compute(self, row):
        return self.value

# Usage - Create feature: (age * 2) + income
age = FieldExpression("age")
income = FieldExpression("income")
two = ConstantExpression(2)

feature = AddExpression(
    MultiplyExpression(age, two),
    income
)

customer = {"age": 30, "income": 50000}
print(f"Feature value: {feature.compute(customer)}")  # (30 * 2) + 50000 = 50060
```

---

### 4. Iterator

**Purpose:** Provides a way to access elements of a collection sequentially without exposing its underlying representation.

**When to Use:** When you need to traverse different data structures in a uniform way.

**Data Context:** Iterating through datasets, batching data for ML training, or streaming large files.

**Examples:**

```python
# Example 1: Data Batch Iterator for ML Training
class DataBatchIterator:
    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.data):
            raise StopIteration

        batch = self.data[self.index:self.index + self.batch_size]
        self.index += self.batch_size
        return batch

# Usage
training_data = list(range(100))  # 100 samples
batch_iterator = DataBatchIterator(training_data, batch_size=10)

for batch_num, batch in enumerate(batch_iterator, 1):
    print(f"Batch {batch_num}: {len(batch)} samples, First: {batch[0]}, Last: {batch[-1]}")
```

```python
# Example 2: Database Result Iterator
class DatabaseResultIterator:
    def __init__(self, query_result):
        self.results = query_result
        self.current = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current >= len(self.results):
            raise StopIteration

        row = self.results[self.current]
        self.current += 1
        return row

class DatabaseQuery:
    def __init__(self, table):
        self.table = table
        # Simulate query results
        self.results = [
            {"id": 1, "name": "Alice", "sales": 1000},
            {"id": 2, "name": "Bob", "sales": 1500},
            {"id": 3, "name": "Charlie", "sales": 2000}
        ]

    def execute(self):
        return DatabaseResultIterator(self.results)

# Usage
query = DatabaseQuery("sales")
results = query.execute()

for row in results:
    print(f"ID: {row['id']}, Name: {row['name']}, Sales: ${row['sales']}")
```

```python
# Example 3: Time Series Window Iterator
class TimeSeriesWindowIterator:
    def __init__(self, data, window_size, step=1):
        self.data = data
        self.window_size = window_size
        self.step = step
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index + self.window_size > len(self.data):
            raise StopIteration

        window = self.data[self.index:self.index + self.window_size]
        self.index += self.step
        return window

# Usage - Sliding window for time series analysis
time_series = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
window_iterator = TimeSeriesWindowIterator(time_series, window_size=3, step=1)

for i, window in enumerate(window_iterator):
    print(f"Window {i+1}: {window}, Mean: {sum(window)/len(window)}")
```

---

### 5. Mediator

**Purpose:** Defines an object that encapsulates how a set of objects interact, promoting loose coupling.

**When to Use:** When you have complex communication between multiple objects.

**Data Context:** Coordinating data pipeline components, managing ML experiment workflows, or orchestrating ETL processes.

**Examples:**

```python
# Example 1: Data Pipeline Mediator
class PipelineMediator:
    def __init__(self):
        self.components = {}

    def register_component(self, name, component):
        self.components[name] = component
        component.mediator = self

    def notify(self, sender, event, data=None):
        if event == "data_extracted":
            print(f"[MEDIATOR] Data extracted, notifying transformer")
            self.components["transformer"].process(data)
        elif event == "data_transformed":
            print(f"[MEDIATOR] Data transformed, notifying validator")
            self.components["validator"].validate(data)
        elif event == "data_validated":
            print(f"[MEDIATOR] Data validated, notifying loader")
            self.components["loader"].load(data)
        elif event == "data_loaded":
            print(f"[MEDIATOR] Pipeline completed")

class PipelineComponent:
    def __init__(self, name):
        self.name = name
        self.mediator = None

class Extractor(PipelineComponent):
    def extract(self):
        data = "raw_data"
        print(f"[EXTRACTOR] Extracting data")
        self.mediator.notify(self, "data_extracted", data)

class Transformer(PipelineComponent):
    def process(self, data):
        print(f"[TRANSFORMER] Processing {data}")
        processed = f"processed_{data}"
        self.mediator.notify(self, "data_transformed", processed)

class Validator(PipelineComponent):
    def validate(self, data):
        print(f"[VALIDATOR] Validating {data}")
        self.mediator.notify(self, "data_validated", data)

class Loader(PipelineComponent):
    def load(self, data):
        print(f"[LOADER] Loading {data}")
        self.mediator.notify(self, "data_loaded", data)

# Usage
mediator = PipelineMediator()

extractor = Extractor("extractor")
transformer = Transformer("transformer")
validator = Validator("validator")
loader = Loader("loader")

mediator.register_component("extractor", extractor)
mediator.register_component("transformer", transformer)
mediator.register_component("validator", validator)
mediator.register_component("loader", loader)

# Start pipeline
extractor.extract()
```

```python
# Example 2: ML Training Coordinator Mediator
class TrainingMediator:
    def __init__(self):
        self.data_loader = None
        self.model_trainer = None
        self.evaluator = None
        self.logger = None

    def register_components(self, data_loader, trainer, evaluator, logger):
        self.data_loader = data_loader
        self.model_trainer = trainer
        self.evaluator = evaluator
        self.logger = logger

    def start_training(self):
        # Coordinate the training workflow
        data = self.data_loader.load()
        self.logger.log(f"Data loaded: {data}")

        model = self.model_trainer.train(data)
        self.logger.log(f"Model trained: {model}")

        metrics = self.evaluator.evaluate(model)
        self.logger.log(f"Evaluation: {metrics}")

        return metrics

class DataLoader:
    def load(self):
        return "training_data"

class ModelTrainer:
    def train(self, data):
        return f"trained_model_on_{data}"

class ModelEvaluator:
    def evaluate(self, model):
        return {"accuracy": 0.92, "f1": 0.89}

class TrainingLogger:
    def log(self, message):
        print(f"[LOG] {message}")

# Usage
mediator = TrainingMediator()
loader = DataLoader()
trainer = ModelTrainer()
evaluator = ModelEvaluator()
logger = TrainingLogger()

mediator.register_components(loader, trainer, evaluator, logger)
results = mediator.start_training()
```

```python
# Example 3: Data Quality Monitoring Mediator
class DataQualityMediator:
    def __init__(self):
        self.monitors = []
        self.alert_system = None

    def add_monitor(self, monitor):
        self.monitors.append(monitor)
        monitor.mediator = self

    def set_alert_system(self, alert_system):
        self.alert_system = alert_system

    def check_quality(self, data):
        issues = []
        for monitor in self.monitors:
            issue = monitor.check(data)
            if issue:
                issues.append(issue)

        if issues and self.alert_system:
            self.alert_system.send_alert(issues)
        return issues

class QualityMonitor:
    def __init__(self, name):
        self.name = name
        self.mediator = None

    def check(self, data):
        pass

class NullMonitor(QualityMonitor):
    def check(self, data):
        null_count = sum(1 for x in data if x is None)
        if null_count > 0:
            return f"{self.name}: Found {null_count} null values"
        return None

class RangeMonitor(QualityMonitor):
    def check(self, data):
        out_of_range = sum(1 for x in data if x is not None and (x < 0 or x > 100))
        if out_of_range > 0:
            return f"{self.name}: Found {out_of_range} out-of-range values"
        return None

class AlertSystem:
    def send_alert(self, issues):
        print(f"[ALERT] Data quality issues detected:")
        for issue in issues:
            print(f"  - {issue}")

# Usage
mediator = DataQualityMediator()
mediator.add_monitor(NullMonitor("NullCheck"))
mediator.add_monitor(RangeMonitor("RangeCheck"))
mediator.set_alert_system(AlertSystem())

test_data = [10, None, 150, 50, None, -5]
issues = mediator.check_quality(test_data)
```

---

### 6. Memento

**Purpose:** Captures and externalizes an object's internal state so it can be restored later without violating encapsulation.

**When to Use:** When you need to implement undo/redo functionality or save snapshots of state.

**Data Context:** Model checkpointing, experiment versioning, data transformation rollback, or pipeline state management.

**Examples:**

```python
# Example 1: Model Training Checkpoint
class ModelMemento:
    def __init__(self, weights, epoch, loss):
        self._weights = weights.copy()
        self._epoch = epoch
        self._loss = loss

    def get_state(self):
        return {
            "weights": self._weights,
            "epoch": self._epoch,
            "loss": self._loss
        }

class MLModel:
    def __init__(self):
        self.weights = [0.1, 0.2, 0.3]
        self.epoch = 0
        self.loss = 1.0

    def train_epoch(self):
        self.epoch += 1
        # Simulate training
        self.weights = [w + 0.01 for w in self.weights]
        self.loss = self.loss * 0.9
        print(f"Epoch {self.epoch}: Loss = {self.loss:.4f}")

    def save(self):
        print(f"Saving checkpoint at epoch {self.epoch}")
        return ModelMemento(self.weights, self.epoch, self.loss)

    def restore(self, memento):
        state = memento.get_state()
        self.weights = state["weights"]
        self.epoch = state["epoch"]
        self.loss = state["loss"]
        print(f"Restored to epoch {self.epoch}")

class CheckpointManager:
    def __init__(self):
        self.checkpoints = []

    def save_checkpoint(self, memento):
        self.checkpoints.append(memento)

    def get_checkpoint(self, index):
        if 0 <= index < len(self.checkpoints):
            return self.checkpoints[index]
        return None

# Usage
model = MLModel()
manager = CheckpointManager()

# Train and save checkpoints
for i in range(5):
    model.train_epoch()
    if i % 2 == 0:  # Save every 2 epochs
        manager.save_checkpoint(model.save())

# Continue training
model.train_epoch()
model.train_epoch()

# Restore to earlier checkpoint
checkpoint = manager.get_checkpoint(1)
if checkpoint:
    model.restore(checkpoint)
```

```python
# Example 2: Data Transformation History
class DataMemento:
    def __init__(self, data, transformation_name):
        self._data = data.copy()
        self._transformation = transformation_name

    def get_data(self):
        return self._data

    def get_transformation(self):
        return self._transformation

class Dataset:
    def __init__(self, data):
        self.data = data
        self.transformations = []

    def apply_transformation(self, name, transform_func):
        print(f"Applying {name}")
        self.data = transform_func(self.data)
        self.transformations.append(name)

    def save_state(self):
        return DataMemento(self.data, ", ".join(self.transformations))

    def restore_state(self, memento):
        self.data = memento.get_data()
        print(f"Restored to state: {memento.get_transformation()}")

class TransformationHistory:
    def __init__(self):
        self.history = []

    def save(self, memento):
        self.history.append(memento)
        print(f"Saved state {len(self.history)}")

    def restore(self, index):
        if 0 <= index < len(self.history):
            return self.history[index]
        return None

# Usage
dataset = Dataset([10, 20, 30, 40, 50])
history = TransformationHistory()

# Save initial state
history.save(dataset.save_state())

# Apply transformations and save
dataset.apply_transformation("normalize", lambda d: [x/max(d) for x in d])
history.save(dataset.save_state())

dataset.apply_transformation("square", lambda d: [x**2 for x in d])
history.save(dataset.save_state())

print(f"Current data: {dataset.data}")

# Restore to previous state
memento = history.restore(1)
if memento:
    dataset.restore_state(memento)
    print(f"Data after restore: {dataset.data}")
```

```python
# Example 3: Configuration Version Control
class ConfigMemento:
    def __init__(self, config, version):
        self._config = config.copy()
        self._version = version

    def get_config(self):
        return self._config

    def get_version(self):
        return self._version

class PipelineConfig:
    def __init__(self):
        self.config = {
            "batch_size": 32,
            "learning_rate": 0.001,
            "epochs": 100
        }
        self.version = 1

    def update_config(self, key, value):
        self.config[key] = value
        self.version += 1
        print(f"Updated {key} to {value} (v{self.version})")

    def save_version(self):
        return ConfigMemento(self.config, self.version)

    def restore_version(self, memento):
        self.config = memento.get_config()
        self.version = memento.get_version()
        print(f"Restored to version {self.version}")

    def display(self):
        print(f"Config v{self.version}: {self.config}")

class ConfigVersionControl:
    def __init__(self):
        self.versions = {}

    def save_version(self, version_name, memento):
        self.versions[version_name] = memento
        print(f"Saved as '{version_name}'")

    def load_version(self, version_name):
        return self.versions.get(version_name)

# Usage
config = PipelineConfig()
vcs = ConfigVersionControl()

# Save initial config
vcs.save_version("baseline", config.save_version())

# Make changes
config.update_config("batch_size", 64)
config.update_config("learning_rate", 0.01)
vcs.save_version("experiment_1", config.save_version())

# More changes
config.update_config("epochs", 200)
config.display()

# Restore to baseline
baseline = vcs.load_version("baseline")
if baseline:
    config.restore_version(baseline)
    config.display()
```

---

### 7. Observer

**Purpose:** Defines a one-to-many dependency where when one object changes state, all its dependents are notified automatically.

**When to Use:** When changes to one object require changing others, and you don't know how many objects need to change.

**Data Context:** Real-time dashboards, data pipeline monitoring, model performance tracking, or event-driven data processing.

**Examples:**

```python
# Example 1: Data Pipeline Monitoring
class Subject:
    def __init__(self):
        self._observers = []

    def attach(self, observer):
        self._observers.append(observer)

    def detach(self, observer):
        self._observers.remove(observer)

    def notify(self, event):
        for observer in self._observers:
            observer.update(event)

class DataPipeline(Subject):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.status = "idle"

    def start(self):
        self.status = "running"
        self.notify({"pipeline": self.name, "status": "started"})

    def complete(self, records_processed):
        self.status = "completed"
        self.notify({
            "pipeline": self.name,
            "status": "completed",
            "records": records_processed
        })

    def fail(self, error):
        self.status = "failed"
        self.notify({
            "pipeline": self.name,
            "status": "failed",
            "error": error
        })

class Observer:
    def update(self, event):
        pass

class LoggingObserver(Observer):
    def update(self, event):
        print(f"[LOG] Pipeline: {event['pipeline']}, Status: {event['status']}")

class MetricsObserver(Observer):
    def __init__(self):
        self.metrics = []

    def update(self, event):
        self.metrics.append(event)
        print(f"[METRICS] Recorded: {event}")

class AlertObserver(Observer):
    def update(self, event):
        if event['status'] == 'failed':
            print(f"[ALERT] CRITICAL: {event['pipeline']} failed - {event.get('error')}")

# Usage
pipeline = DataPipeline("ETL_Daily_Sales")

# Attach observers
logger = LoggingObserver()
metrics = MetricsObserver()
alerter = AlertObserver()

pipeline.attach(logger)
pipeline.attach(metrics)
pipeline.attach(alerter)

# Run pipeline
pipeline.start()
pipeline.complete(10000)

# Simulate failure
pipeline.fail("Database connection timeout")
```

```python
# Example 2: Model Performance Tracker
class ModelSubject:
    def __init__(self):
        self._observers = []
        self.metrics = {}

    def attach(self, observer):
        self._observers.append(observer)

    def update_metric(self, metric_name, value):
        self.metrics[metric_name] = value
        self.notify_observers(metric_name, value)

    def notify_observers(self, metric_name, value):
        for observer in self._observers:
            observer.update(metric_name, value)

class PerformanceMonitor(Observer):
    def __init__(self, threshold):
        self.threshold = threshold

    def update(self, metric_name, value):
        if metric_name == "accuracy" and value < self.threshold:
            print(f"[MONITOR] WARNING: Accuracy {value} below threshold {self.threshold}")

class PerformanceDashboard(Observer):
    def __init__(self):
        self.metrics = {}

    def update(self, metric_name, value):
        self.metrics[metric_name] = value
        print(f"[DASHBOARD] Updated {metric_name}: {value}")

class ModelRetrainer(Observer):
    def update(self, metric_name, value):
        if metric_name == "accuracy" and value < 0.8:
            print(f"[RETRAINER] Triggering model retraining due to low accuracy: {value}")

# Usage
model = ModelSubject()

# Attach observers
monitor = PerformanceMonitor(threshold=0.85)
dashboard = PerformanceDashboard()
retrainer = ModelRetrainer()

model.attach(monitor)
model.attach(dashboard)
model.attach(retrainer)

# Update metrics
model.update_metric("accuracy", 0.92)
model.update_metric("precision", 0.89)
model.update_metric("accuracy", 0.75)  # Triggers alerts
```

```python
# Example 3: Real-time Data Stream Observer
class DataStreamSubject:
    def __init__(self):
        self._observers = []

    def attach(self, observer):
        self._observers.append(observer)

    def receive_data(self, data):
        # Notify all observers when new data arrives
        for observer in self._observers:
            observer.on_data(data)

class AnomalyDetector(Observer):
    def __init__(self, threshold):
        self.threshold = threshold

    def on_data(self, data):
        if data > self.threshold:
            print(f"[ANOMALY] Detected unusual value: {data}")

class DataAggregator(Observer):
    def __init__(self):
        self.sum = 0
        self.count = 0

    def on_data(self, data):
        self.sum += data
        self.count += 1
        avg = self.sum / self.count
        print(f"[AGGREGATOR] Running average: {avg:.2f}")

class DataWriter(Observer):
    def __init__(self, filename):
        self.filename = filename
        self.buffer = []

    def on_data(self, data):
        self.buffer.append(data)
        if len(self.buffer) >= 5:
            print(f"[WRITER] Writing batch to {self.filename}: {self.buffer}")
            self.buffer = []

# Usage
stream = DataStreamSubject()

# Attach observers
detector = AnomalyDetector(threshold=100)
aggregator = DataAggregator()
writer = DataWriter("data.txt")

stream.attach(detector)
stream.attach(aggregator)
stream.attach(writer)

# Simulate streaming data
data_points = [50, 75, 120, 60, 80, 150, 70]
for data in data_points:
    print(f"\nReceived: {data}")
    stream.receive_data(data)
```

---

### 8. State

**Purpose:** Allows an object to alter its behavior when its internal state changes, appearing to change its class.

**When to Use:** When an object's behavior depends on its state and it must change behavior at runtime.

**Data Context:** Pipeline execution states, model lifecycle management, data processing workflows, or connection state management.

**Examples:**

```python
# Example 1: Data Pipeline State Machine
class PipelineState:
    def start(self, pipeline):
        pass

    def pause(self, pipeline):
        pass

    def resume(self, pipeline):
        pass

    def stop(self, pipeline):
        pass

    def process(self, pipeline, data):
        pass

class IdleState(PipelineState):
    def start(self, pipeline):
        print("[IDLE] Starting pipeline...")
        pipeline.state = RunningState()

    def process(self, pipeline, data):
        print("[IDLE] Cannot process, pipeline not started")

class RunningState(PipelineState):
    def pause(self, pipeline):
        print("[RUNNING] Pausing pipeline...")
        pipeline.state = PausedState()

    def stop(self, pipeline):
        print("[RUNNING] Stopping pipeline...")
        pipeline.state = StoppedState()

    def process(self, pipeline, data):
        print(f"[RUNNING] Processing: {data}")
        pipeline.records_processed += len(data)

class PausedState(PipelineState):
    def resume(self, pipeline):
        print("[PAUSED] Resuming pipeline...")
        pipeline.state = RunningState()

    def stop(self, pipeline):
        print("[PAUSED] Stopping pipeline...")
        pipeline.state = StoppedState()

    def process(self, pipeline, data):
        print("[PAUSED] Cannot process, pipeline is paused")

class StoppedState(PipelineState):
    def start(self, pipeline):
        print("[STOPPED] Cannot restart, pipeline is stopped")

    def process(self, pipeline, data):
        print("[STOPPED] Cannot process, pipeline is stopped")

class Pipeline:
    def __init__(self):
        self.state = IdleState()
        self.records_processed = 0

    def start(self):
        self.state.start(self)

    def pause(self):
        self.state.pause(self)

    def resume(self):
        self.state.resume(self)

    def stop(self):
        self.state.stop(self)

    def process(self, data):
        self.state.process(self, data)

# Usage
pipeline = Pipeline()

pipeline.process([1, 2, 3])  # Cannot process in idle state
pipeline.start()
pipeline.process([1, 2, 3])  # Processing
pipeline.pause()
pipeline.process([4, 5, 6])  # Cannot process while paused
pipeline.resume()
pipeline.process([4, 5, 6])  # Processing
pipeline.stop()
pipeline.process([7, 8, 9])  # Cannot process when stopped

print(f"Total records processed: {pipeline.records_processed}")
```

```python
# Example 2: ML Model Lifecycle States
class ModelState:
    def train(self, model):
        pass

    def evaluate(self, model):
        pass

    def deploy(self, model):
        pass

    def retire(self, model):
        pass

class UntrainedState(ModelState):
    def train(self, model):
        print("[UNTRAINED] Training model...")
        model.accuracy = 0.85
        model.state = TrainedState()

    def evaluate(self, model):
        print("[UNTRAINED] Cannot evaluate untrained model")

class TrainedState(ModelState):
    def evaluate(self, model):
        print(f"[TRAINED] Evaluating model... Accuracy: {model.accuracy}")
        if model.accuracy >= 0.8:
            model.state = ValidatedState()
        else:
            print("[TRAINED] Model failed validation")

    def train(self, model):
        print("[TRAINED] Retraining model...")
        model.accuracy = 0.90

class ValidatedState(ModelState):
    def deploy(self, model):
        print("[VALIDATED] Deploying model to production...")
        model.state = DeployedState()

    def train(self, model):
        print("[VALIDATED] Retraining model...")
        model.state = TrainedState()

class DeployedState(ModelState):
    def retire(self, model):
        print("[DEPLOYED] Retiring model from production...")
        model.state = RetiredState()

    def evaluate(self, model):
        print(f"[DEPLOYED] Monitoring production performance: {model.accuracy}")

class RetiredState(ModelState):
    def train(self, model):
        print("[RETIRED] Cannot retrain retired model")

    def deploy(self, model):
        print("[RETIRED] Cannot redeploy retired model")

class MLModel:
    def __init__(self):
        self.state = UntrainedState()
        self.accuracy = 0.0

    def train(self):
        self.state.train(self)

    def evaluate(self):
        self.state.evaluate(self)

    def deploy(self):
        self.state.deploy(self)

    def retire(self):
        self.state.retire(self)

# Usage
model = MLModel()

model.evaluate()  # Cannot evaluate untrained
model.train()
model.evaluate()  # Passes validation
model.deploy()
model.evaluate()  # Monitor production
model.retire()
model.deploy()    # Cannot redeploy retired
```

```python
# Example 3: Database Connection State
class ConnectionState:
    def connect(self, connection):
        pass

    def disconnect(self, connection):
        pass

    def execute_query(self, connection, query):
        pass

    def begin_transaction(self, connection):
        pass

    def commit(self, connection):
        pass

class DisconnectedState(ConnectionState):
    def connect(self, connection):
        print("[DISCONNECTED] Connecting to database...")
        connection.state = ConnectedState()

    def execute_query(self, connection, query):
        print("[DISCONNECTED] Cannot execute query, not connected")

class ConnectedState(ConnectionState):
    def disconnect(self, connection):
        print("[CONNECTED] Disconnecting from database...")
        connection.state = DisconnectedState()

    def execute_query(self, connection, query):
        print(f"[CONNECTED] Executing query: {query}")
        return f"Result for {query}"

    def begin_transaction(self, connection):
        print("[CONNECTED] Beginning transaction...")
        connection.state = TransactionState()

class TransactionState(ConnectionState):
    def execute_query(self, connection, query):
        print(f"[TRANSACTION] Executing query in transaction: {query}")
        connection.transaction_queries.append(query)

    def commit(self, connection):
        print(f"[TRANSACTION] Committing {len(connection.transaction_queries)} queries")
        connection.transaction_queries = []
        connection.state = ConnectedState()

    def disconnect(self, connection):
        print("[TRANSACTION] Cannot disconnect during transaction")

class DatabaseConnection:
    def __init__(self):
        self.state = DisconnectedState()
        self.transaction_queries = []

    def connect(self):
        self.state.connect(self)

    def disconnect(self):
        self.state.disconnect(self)

    def execute_query(self, query):
        return self.state.execute_query(self, query)

    def begin_transaction(self):
        self.state.begin_transaction(self)

    def commit(self):
        self.state.commit(self)

# Usage
db = DatabaseConnection()

db.execute_query("SELECT * FROM users")  # Cannot execute
db.connect()
db.execute_query("SELECT * FROM users")  # Executes
db.begin_transaction()
db.execute_query("UPDATE users SET active=1")
db.execute_query("DELETE FROM logs")
db.disconnect()  # Cannot disconnect during transaction
db.commit()
db.disconnect()  # Now can disconnect
```

---

## Summary

This cheat sheet covers all major design patterns with practical examples for Data Analytics, Data Engineering, and Data Science:

### **Creational Patterns** - Object creation

- **Singleton**: Single instance (DB connections, config managers)
- **Factory Method**: Create objects dynamically (data readers, models)
- **Abstract Factory**: Create families of related objects (cloud platforms)
- **Builder**: Construct complex objects step-by-step (ML pipelines, queries)
- **Prototype**: Clone existing objects (model configs, datasets)

### **Structural Patterns** - Object composition

- **Adapter**: Make incompatible interfaces compatible (legacy systems, APIs)
- **Bridge**: Separate abstraction from implementation (visualizations, reports)
- **Composite**: Tree structures of objects (data hierarchies, pipelines)
- **Decorator**: Add behavior dynamically (logging, caching, monitoring)
- **Facade**: Simplify complex subsystems (ETL workflows, ML training)
- **Flyweight**: Share common data to reduce memory (feature values, metadata)
- **Proxy**: Control access to objects (lazy loading, caching, access control)

### **Behavioral Patterns** - Object communication

- **Chain of Responsibility**: Pass requests through handler chain (validation, logging)
- **Command**: Encapsulate requests as objects (ETL jobs, experiments)
- **Interpreter**: Interpret domain-specific languages (filters, metrics, formulas)
- **Iterator**: Access collection elements sequentially (batching, streaming)
- **Mediator**: Coordinate object interactions (pipeline orchestration)
- **Memento**: Save and restore object state (checkpoints, versioning)
- **Observer**: Notify dependents of state changes (monitoring, dashboards)
- **State**: Change behavior based on state (pipeline lifecycle, model states)

These patterns help build scalable, maintainable, and robust data systems!
