# Scikit-Learn Feature Engineering Cheatsheet

## Table of Contents

1. [Preprocessing & Scaling](#preprocessing--scaling)
2. [Encoding Categorical Variables](#encoding-categorical-variables)
3. [Feature Selection](#feature-selection)
4. [Dimensionality Reduction](#dimensionality-reduction)
5. [Feature Construction](#feature-construction)
6. [Text Feature Engineering](#text-feature-engineering)
7. [Pipeline & Column Transformer](#pipeline--column-transformer)
8. [Handling Missing Values](#handling-missing-values)
9. [Time Series Features](#time-series-features)
10. [Advanced Techniques](#advanced-techniques)

---

## Preprocessing & Scaling

### StandardScaler

**Definition**: Standardizes features by removing the mean and scaling to unit variance (z-score normalization).
**When to use**: When features have different scales and you want mean=0, std=1. Required for algorithms sensitive to scale (SVM, neural networks, PCA).

```python
from sklearn.preprocessing import StandardScaler
import numpy as np

# Sample data
X = np.array([[1, 2], [3, 4], [5, 6]])

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"Mean: {X_scaled.mean(axis=0)}")  # [0, 0]
print(f"Std: {X_scaled.std(axis=0)}")   # [1, 1]

# Transform new data
X_new = np.array([[2, 3]])
X_new_scaled = scaler.transform(X_new)
```

### MinMaxScaler

**Definition**: Scales features to a fixed range, typically [0, 1].
**When to use**: When you need features in a specific range, preserving relationships. Good for neural networks and algorithms that don't assume normal distribution.

```python
from sklearn.preprocessing import MinMaxScaler

# Scale to [0, 1]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Custom range [0, 10]
scaler = MinMaxScaler(feature_range=(0, 10))
X_scaled = scaler.fit_transform(X)
```

### RobustScaler

**Definition**: Scales features using statistics that are robust to outliers (median and IQR).
**When to use**: When your data contains outliers that you want to preserve but not let dominate the scaling.

```python
from sklearn.preprocessing import RobustScaler

# Scale using median and IQR
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
# Formula: (X - median) / IQR
```

### MaxAbsScaler

**Definition**: Scales each feature by its maximum absolute value.
**When to use**: For sparse data where you want to preserve sparsity (zeros remain zeros).

```python
from sklearn.preprocessing import MaxAbsScaler

scaler = MaxAbsScaler()
X_scaled = scaler.fit_transform(X)
# All values will be in [-1, 1] range
```

### PowerTransformer

**Definition**: Applies power transformation to make data more Gaussian-like.
**When to use**: When features are skewed and you need normal distribution for algorithms like linear regression.

```python
from sklearn.preprocessing import PowerTransformer

# Yeo-Johnson (works with negative values)
transformer = PowerTransformer(method='yeo-johnson')
X_transformed = transformer.fit_transform(X)

# Box-Cox (only positive values)
transformer = PowerTransformer(method='box-cox')
X_transformed = transformer.fit_transform(X_positive)
```

### QuantileTransformer

**Definition**: Transforms features to follow a uniform or normal distribution using quantiles.
**When to use**: When you want to reduce impact of outliers and make distribution uniform or normal.

```python
from sklearn.preprocessing import QuantileTransformer

# Transform to uniform distribution
transformer = QuantileTransformer(output_distribution='uniform')
X_uniform = transformer.fit_transform(X)

# Transform to normal distribution
transformer = QuantileTransformer(output_distribution='normal')
X_normal = transformer.fit_transform(X)
```

---

## Encoding Categorical Variables

### LabelEncoder

**Definition**: Converts categorical labels into numeric labels (0 to n_classes-1).
**When to use**: For target variables or ordinal features where order matters.

```python
from sklearn.preprocessing import LabelEncoder

# Encode labels
le = LabelEncoder()
labels = ['cat', 'dog', 'fish', 'cat', 'dog']
encoded = le.fit_transform(labels)
print(encoded)  # [0 1 2 0 1]

# Decode back
decoded = le.inverse_transform(encoded)
print(decoded)  # ['cat' 'dog' 'fish' 'cat' 'dog']
```

### OneHotEncoder

**Definition**: Creates binary columns for each category (dummy variables).
**When to use**: For nominal categorical features where no order exists. Prevents algorithms from assuming ordinal relationships.

```python
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# Basic usage
encoder = OneHotEncoder()
categories = [['red'], ['blue'], ['green'], ['red']]
encoded = encoder.fit_transform(categories).toarray()

# With pandas
df = pd.DataFrame({'color': ['red', 'blue', 'green', 'red']})
encoder = OneHotEncoder(sparse_output=False)
encoded = encoder.fit_transform(df[['color']])

# Handle unknown categories
encoder = OneHotEncoder(handle_unknown='ignore')
```

### OrdinalEncoder

**Definition**: Encodes categorical features as ordinal integers.
**When to use**: When categories have a meaningful order (low, medium, high) or when you want to preserve ordinal relationships.

```python
from sklearn.preprocessing import OrdinalEncoder

# Define order
categories = [['low', 'medium', 'high']]
encoder = OrdinalEncoder(categories=categories)

data = [['low'], ['high'], ['medium'], ['low']]
encoded = encoder.fit_transform(data)
print(encoded)  # [[0], [2], [1], [0]]
```

### TargetEncoder

**Definition**: Encodes categorical features using target variable statistics.
**When to use**: For high-cardinality categorical features. Use with cross-validation to prevent overfitting.

```python
from sklearn.preprocessing import TargetEncoder

# Mean target encoding
encoder = TargetEncoder()
X_encoded = encoder.fit_transform(X_categorical, y)
```

---

## Feature Selection

### SelectKBest

**Definition**: Selects k best features based on univariate statistical tests.
**When to use**: When you want to quickly reduce dimensionality by keeping most informative features.

```python
from sklearn.feature_selection import SelectKBest, f_classif, chi2

# For classification
selector = SelectKBest(score_func=f_classif, k=5)
X_selected = selector.fit_transform(X, y)

# For categorical features
selector = SelectKBest(score_func=chi2, k=5)
X_selected = selector.fit_transform(X_positive, y)

# Get selected feature names
selected_features = selector.get_support(indices=True)
```

### SelectPercentile

**Definition**: Selects features based on percentile of highest scores.
**When to use**: When you want to keep a percentage of best features rather than a fixed number.

```python
from sklearn.feature_selection import SelectPercentile

selector = SelectPercentile(score_func=f_classif, percentile=25)
X_selected = selector.fit_transform(X, y)
```

### RFE (Recursive Feature Elimination)

**Definition**: Recursively eliminates features by fitting model and removing weakest features.
**When to use**: When you want feature selection based on model performance rather than statistical tests.

```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# Select 5 features using logistic regression
estimator = LogisticRegression()
selector = RFE(estimator, n_features_to_select=5)
X_selected = selector.fit_transform(X, y)

# Get ranking
print(selector.ranking_)
```

### SelectFromModel

**Definition**: Selects features based on importance weights from any model.
**When to use**: When you want to use model-specific feature importance for selection.

```python
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

# Use Random Forest feature importance
rf = RandomForestClassifier()
selector = SelectFromModel(rf, threshold='median')
X_selected = selector.fit_transform(X, y)
```

### VarianceThreshold

**Definition**: Removes features with low variance (quasi-constant features).
**When to use**: As first step to remove features with little information content.

```python
from sklearn.feature_selection import VarianceThreshold

# Remove features with variance < 0.1
selector = VarianceThreshold(threshold=0.1)
X_selected = selector.fit_transform(X)
```

---

## Dimensionality Reduction

### PCA (Principal Component Analysis)

**Definition**: Projects data onto lower-dimensional space preserving maximum variance.
**When to use**: For dimensionality reduction, data visualization, noise reduction. Works best with standardized data.

```python
from sklearn.decomposition import PCA

# Reduce to 2 components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Explained variance ratio
print(pca.explained_variance_ratio_)

# Determine optimal components
pca = PCA()
pca.fit(X_scaled)
cumsum = pca.explained_variance_ratio_.cumsum()
n_components = (cumsum >= 0.95).argmax() + 1
```

### TruncatedSVD

**Definition**: Dimensionality reduction using SVD, works with sparse matrices.
**When to use**: For sparse data (text, recommender systems) where PCA is not suitable.

```python
from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=50)
X_svd = svd.fit_transform(X_sparse)
```

### LDA (Linear Discriminant Analysis)

**Definition**: Finds linear combinations that best separate different classes.
**When to use**: For supervised dimensionality reduction and classification preprocessing.

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)
```

### t-SNE

**Definition**: Non-linear dimensionality reduction good for visualization.
**When to use**: For data visualization and exploration, not for preprocessing ML models.

```python
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)
```

---

## Feature Construction

### PolynomialFeatures

**Definition**: Generates polynomial and interaction features.
**When to use**: To capture non-linear relationships and feature interactions.

```python
from sklearn.preprocessing import PolynomialFeatures

# Degree-2 polynomials with interactions
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Only interaction terms
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_interactions = poly.fit_transform(X)

# Get feature names
feature_names = poly.get_feature_names_out(['x1', 'x2'])
```

### SplineTransformer

**Definition**: Creates spline basis functions for non-linear transformations.
**When to use**: For smooth non-linear transformations of continuous features.

```python
from sklearn.preprocessing import SplineTransformer

# B-spline transformation
spline = SplineTransformer(n_knots=4, degree=3)
X_spline = spline.fit_transform(X)
```

### FunctionTransformer

**Definition**: Applies custom functions to features.
**When to use**: For domain-specific transformations or mathematical functions.

```python
from sklearn.preprocessing import FunctionTransformer
import numpy as np

# Log transformation
log_transformer = FunctionTransformer(np.log1p)
X_log = log_transformer.fit_transform(X_positive)

# Custom function
def custom_transform(X):
    return X ** 2 + np.sin(X)

custom_transformer = FunctionTransformer(custom_transform)
X_custom = custom_transformer.fit_transform(X)
```

---

## Text Feature Engineering

### CountVectorizer

**Definition**: Converts text to matrix of token counts.
**When to use**: For basic text classification when word frequency matters.

```python
from sklearn.feature_extraction.text import CountVectorizer

texts = ["hello world", "world of data", "hello data"]
vectorizer = CountVectorizer()
X_counts = vectorizer.fit_transform(texts)

# With parameters
vectorizer = CountVectorizer(
    max_features=1000,    # Top 1000 features
    ngram_range=(1, 2),   # Unigrams and bigrams
    stop_words='english', # Remove stop words
    min_df=2,             # Minimum document frequency
    max_df=0.8            # Maximum document frequency
)
```

### TfidfVectorizer

**Definition**: Converts text to TF-IDF (Term Frequency-Inverse Document Frequency) matrix.
**When to use**: For text classification when you want to weight terms by importance across corpus.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 3),
    stop_words='english',
    sublinear_tf=True     # Apply log scaling
)
X_tfidf = vectorizer.fit_transform(texts)
```

### HashingVectorizer

**Definition**: Converts text to matrix using hashing trick.
**When to use**: For large datasets where memory is limited or vocabulary is very large.

```python
from sklearn.feature_extraction.text import HashingVectorizer

vectorizer = HashingVectorizer(n_features=1000, ngram_range=(1, 2))
X_hash = vectorizer.fit_transform(texts)
```

---

## Pipeline & Column Transformer

### Pipeline

**Definition**: Chains multiple preprocessing steps and estimators.
**When to use**: To ensure consistent preprocessing and prevent data leakage.

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Simple pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', SelectKBest(k=10)),
    ('classifier', LogisticRegression())
])

pipe.fit(X, y)
predictions = pipe.predict(X_test)
```

### ColumnTransformer

**Definition**: Applies different transformations to different columns.
**When to use**: When you have mixed data types requiring different preprocessing.

```python
from sklearn.compose import ColumnTransformer

# Define transformers for different column types
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine transformers
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_columns),
    ('cat', categorical_transformer, categorical_columns)
])

# Full pipeline
full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])
```

---

## Handling Missing Values

### SimpleImputer

**Definition**: Fills missing values using simple strategies.
**When to use**: For basic missing value imputation.

```python
from sklearn.impute import SimpleImputer

# Numerical imputation
num_imputer = SimpleImputer(strategy='mean')  # 'median', 'most_frequent'
X_imputed = num_imputer.fit_transform(X_numeric)

# Categorical imputation
cat_imputer = SimpleImputer(strategy='most_frequent')
X_imputed = cat_imputer.fit_transform(X_categorical)

# Constant value
const_imputer = SimpleImputer(strategy='constant', fill_value=0)
```

### IterativeImputer

**Definition**: Imputes missing values using other features in an iterative process.
**When to use**: When features are correlated and you want sophisticated imputation.

```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imputer = IterativeImputer(random_state=42)
X_imputed = imputer.fit_transform(X)
```

### KNNImputer

**Definition**: Imputes missing values using k-nearest neighbors.
**When to use**: When similar samples should have similar values.

```python
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)
X_imputed = imputer.fit_transform(X)
```

---

## Time Series Features

### Creating Time-based Features

**Definition**: Extract temporal features from datetime columns.
**When to use**: For time series analysis and forecasting.

```python
import pandas as pd
from sklearn.preprocessing import FunctionTransformer

def create_time_features(df):
    df = df.copy()
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
    return df

time_transformer = FunctionTransformer(create_time_features)
```

### Lag Features

**Definition**: Create features from previous time steps.
**When to use**: To capture temporal dependencies in time series.

```python
def create_lag_features(df, lags=[1, 2, 3, 7]):
    df = df.copy()
    for lag in lags:
        df[f'lag_{lag}'] = df['value'].shift(lag)
    return df
```

---

## Advanced Techniques

### Feature Unions

**Definition**: Combines multiple feature extraction methods.
**When to use**: When you want to apply different transformations and combine results.

```python
from sklearn.pipeline import FeatureUnion

# Combine PCA and SelectKBest
feature_union = FeatureUnion([
    ('pca', PCA(n_components=10)),
    ('select_best', SelectKBest(k=15))
])

X_combined = feature_union.fit_transform(X, y)
```

### Custom Transformers

**Definition**: Create your own transformer classes.
**When to use**: For domain-specific transformations not available in scikit-learn.

```python
from sklearn.base import BaseEstimator, TransformerMixin

class OutlierClipper(BaseEstimator, TransformerMixin):
    def __init__(self, quantiles=(0.05, 0.95)):
        self.quantiles = quantiles

    def fit(self, X, y=None):
        self.clip_values_ = {}
        for i in range(X.shape[1]):
            q_low, q_high = np.percentile(X[:, i],
                                         [self.quantiles[0]*100,
                                          self.quantiles[1]*100])
            self.clip_values_[i] = (q_low, q_high)
        return self

    def transform(self, X):
        X_clipped = X.copy()
        for i in range(X.shape[1]):
            q_low, q_high = self.clip_values_[i]
            X_clipped[:, i] = np.clip(X[:, i], q_low, q_high)
        return X_clipped

# Usage
clipper = OutlierClipper()
X_clipped = clipper.fit_transform(X)
```

### Feature Binning

**Definition**: Convert continuous features into discrete bins.
**When to use**: To capture non-linear relationships or reduce noise.

```python
from sklearn.preprocessing import KBinsDiscretizer

# Equal-width binning
discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
X_binned = discretizer.fit_transform(X)

# Quantile-based binning
discretizer = KBinsDiscretizer(n_bins=5, encode='onehot', strategy='quantile')
X_binned = discretizer.fit_transform(X)
```

---

## Best Practices

### 1. Order of Operations

```python
# Recommended order:
# 1. Handle missing values
# 2. Encode categorical variables
# 3. Scale/normalize features
# 4. Feature selection/dimensionality reduction
# 5. Model training
```

### 2. Preventing Data Leakage

```python
# Always fit on training data only
transformer.fit(X_train)
X_train_transformed = transformer.transform(X_train)
X_test_transformed = transformer.transform(X_test)  # Use same transformation
```

### 3. Cross-Validation with Pipelines

```python
from sklearn.model_selection import cross_val_score

# Proper cross-validation
scores = cross_val_score(pipeline, X, y, cv=5)
```

### 4. Feature Engineering Checklist

- Handle missing values appropriately
- Encode categorical variables correctly
- Scale numerical features when needed
- Create domain-specific features
- Select relevant features
- Validate transformations don't leak information
- Use pipelines for reproducibility
- Test preprocessing on validation data

This cheatsheet covers the most important feature engineering techniques in scikit-learn. Remember to always validate your preprocessing steps and use pipelines to ensure reproducible results.
