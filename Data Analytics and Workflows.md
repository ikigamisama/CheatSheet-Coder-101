# Four Types of Data Analytics: Complete Guide

Data analytics is the science of analyzing raw data to make conclusions about information. There are four main types of data analytics, each serving different purposes and answering different business questions.

## 1. Descriptive Analytics

### Definition

**What happened?**
Summarizes historical data to identify patterns and trends. Descriptive analytics examines past performance and provides insights into what has occurred in your business or organization.

### 10-Step Workflow

1. **Collect Data** - Gather historical data from various sources
2. **Clean Data** - Remove inconsistencies and handle missing values
3. **Aggregate Metrics** - Summarize data using statistical measures
4. **Filter Noise** - Remove irrelevant or erroneous data points
5. **Segment Data** - Break data into meaningful categories
6. **Visualize Trends** - Create charts and graphs to show patterns
7. **Compare Periods** - Analyze data across different time frames
8. **Generate Reports** - Create comprehensive summaries
9. **Identify Patterns** - Recognize recurring trends and behaviors
10. **Share Insights** - Communicate findings to stakeholders

### Python Example

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Step 1: Collect Data
np.random.seed(42)
data = {
    'date': pd.date_range('2023-01-01', periods=365, freq='D'),
    'sales': np.random.normal(1000, 200, 365),
    'region': np.random.choice(['North', 'South', 'East', 'West'], 365),
    'product': np.random.choice(['A', 'B', 'C'], 365),
    'customer_type': np.random.choice(['New', 'Returning'], 365)
}

df = pd.DataFrame(data)
df['sales'] = np.abs(df['sales'])  # Ensure positive sales
df['month'] = df['date'].dt.month
df['quarter'] = df['date'].dt.quarter

# Step 2: Clean Data
print("Data Info:")
print(df.info())
print(f"Missing values: {df.isnull().sum().sum()}")

# Step 3: Aggregate Metrics
print("\nDescriptive Statistics:")
print(df['sales'].describe())

# Step 4-5: Filter Noise & Segment Data
# Remove outliers (sales > 3 standard deviations)
sales_std = df['sales'].std()
sales_mean = df['sales'].mean()
df_clean = df[abs(df['sales'] - sales_mean) <= 3 * sales_std]

# Step 6: Visualize Trends
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Monthly trends
monthly_sales = df_clean.groupby('month')['sales'].sum()
axes[0,0].plot(monthly_sales.index, monthly_sales.values, marker='o')
axes[0,0].set_title('Monthly Sales Trend')
axes[0,0].set_xlabel('Month')
axes[0,0].set_ylabel('Total Sales')

# Sales by region
region_sales = df_clean.groupby('region')['sales'].sum()
axes[0,1].bar(region_sales.index, region_sales.values,
              color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
axes[0,1].set_title('Sales by Region')
axes[0,1].set_xlabel('Region')
axes[0,1].set_ylabel('Total Sales')

# Step 7: Compare Periods (Quarterly comparison)
quarterly_sales = df_clean.groupby('quarter')['sales'].sum()
axes[1,0].bar(quarterly_sales.index, quarterly_sales.values,
              color=['red', 'blue', 'green', 'orange'])
axes[1,0].set_title('Quarterly Sales Comparison')
axes[1,0].set_xlabel('Quarter')
axes[1,0].set_ylabel('Total Sales')

# Customer type analysis
customer_sales = df_clean.groupby('customer_type')['sales'].sum()
axes[1,1].pie(customer_sales.values, labels=customer_sales.index, autopct='%1.1f%%')
axes[1,1].set_title('Sales by Customer Type')

plt.tight_layout()
plt.show()

# Step 8: Generate Reports
print("\nSales Report Summary:")
print(f"Total Sales: ${df_clean['sales'].sum():,.2f}")
print(f"Average Daily Sales: ${df_clean['sales'].mean():,.2f}")
print(f"Best Performing Region: {region_sales.idxmax()}")
print(f"Best Month: {monthly_sales.idxmax()}")

# Step 9: Identify Patterns
correlation_matrix = df_clean[['sales', 'month']].corr()
print(f"\nSales-Month Correlation: {correlation_matrix.loc['sales', 'month']:.3f}")
```

---

## 2. Diagnostic Analytics

### Definition

**Why did it happen?**
Examines data to discover the root cause of issues. Goes beyond descriptive analytics to understand the underlying reasons behind trends, anomalies, and performance variations.

### 10-Step Workflow

1. **Identify Anomaly** - Spot unusual patterns or deviations
2. **Gather Logs** - Collect detailed transactional data
3. **Segment Data** - Break down data by relevant dimensions
4. **Drill Down** - Examine data at granular levels
5. **Correlate Events** - Find relationships between variables
6. **Analyze Metrics** - Deep dive into key performance indicators
7. **Compare Segments** - Analyze differences between groups
8. **Test Hypotheses** - Validate potential explanations
9. **Validate Causes** - Confirm root cause relationships
10. **Document Findings** - Record insights and conclusions

### Python Example

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency

# Step 1-2: Identify Anomaly & Gather Logs
np.random.seed(42)
dates = pd.date_range('2023-01-01', periods=365, freq='D')
data = {
    'date': dates,
    'sales': np.random.normal(1000, 200, 365),
    'temperature': np.random.normal(20, 15, 365),
    'marketing_spend': np.random.normal(500, 150, 365),
    'competitor_promotion': np.random.choice([0, 1], 365, p=[0.8, 0.2]),
    'day_of_week': [d.strftime('%A') for d in dates],
    'region': np.random.choice(['North', 'South', 'East', 'West'], 365),
    'weather': np.random.choice(['Sunny', 'Rainy', 'Cloudy'], 365)
}

# Create sales influenced by various factors
weather_effect = np.where(np.array(data['weather']) == 'Rainy', -100, 0)
competitor_effect = np.array(data['competitor_promotion']) * -200
weekend_effect = np.where(pd.Series(data['day_of_week']).isin(['Saturday', 'Sunday']), 150, 0)

data['sales'] = (1000 +
                data['temperature'] * 10 +
                data['marketing_spend'] * 0.5 +
                weather_effect +
                competitor_effect +
                weekend_effect +
                np.random.normal(0, 100, 365))

df = pd.DataFrame(data)
df['sales'] = np.maximum(df['sales'], 0)  # Ensure non-negative sales

# Step 1: Identify Anomaly - Find days with unusually low sales
low_threshold = df['sales'].quantile(0.1)
anomaly_days = df[df['sales'] < low_threshold]

print(f"Diagnostic Analytics: Investigating Low Sales Days")
print(f"Low sales threshold: {low_threshold:.2f}")
print(f"Number of anomaly days: {len(anomaly_days)} out of {len(df)}")

# Step 3: Segment Data
print(f"\nStep 3: Data Segmentation")
print("Sales distribution by weather:")
weather_sales = df.groupby('weather')['sales'].agg(['mean', 'count'])
print(weather_sales)

# Step 4: Drill Down Analysis
print(f"\nStep 4: Drill Down Analysis")
print("Average factors during low sales days vs normal days:")
normal_days = df[df['sales'] >= low_threshold]

comparison = pd.DataFrame({
    'Low_Sales_Days': [
        anomaly_days['temperature'].mean(),
        anomaly_days['marketing_spend'].mean(),
        anomaly_days['competitor_promotion'].mean(),
        anomaly_days[anomaly_days['weather'] == 'Rainy'].shape[0] / len(anomaly_days)
    ],
    'Normal_Days': [
        normal_days['temperature'].mean(),
        normal_days['marketing_spend'].mean(),
        normal_days['competitor_promotion'].mean(),
        normal_days[normal_days['weather'] == 'Rainy'].shape[0] / len(normal_days)
    ]
}, index=['Temperature', 'Marketing_Spend', 'Competitor_Promotion_Rate', 'Rainy_Day_Rate'])

print(comparison)

# Step 5: Correlate Events
print(f"\nStep 5: Correlation Analysis")
correlation_matrix = df[['sales', 'temperature', 'marketing_spend', 'competitor_promotion']].corr()
print("Correlation with Sales:")
print(correlation_matrix['sales'].sort_values(ascending=False))

# Step 6-7: Analyze Metrics & Compare Segments
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Temperature vs Sales
axes[0,0].scatter(df['temperature'], df['sales'], alpha=0.6)
axes[0,0].set_xlabel('Temperature')
axes[0,0].set_ylabel('Sales')
axes[0,0].set_title('Temperature vs Sales')

# Weather impact
weather_avg = df.groupby('weather')['sales'].mean()
axes[0,1].bar(weather_avg.index, weather_avg.values, color=['orange', 'blue', 'gray'])
axes[0,1].set_title('Average Sales by Weather')
axes[0,1].set_ylabel('Average Sales')

# Competitor promotion impact
promo_sales = df.groupby('competitor_promotion')['sales'].mean()
axes[1,0].bar(['No Promotion', 'Promotion'], promo_sales.values, color=['green', 'red'])
axes[1,0].set_title('Sales: Competitor Promotion Impact')
axes[1,0].set_ylabel('Average Sales')

# Day of week analysis
dow_sales = df.groupby('day_of_week')['sales'].mean().reindex(
    ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
axes[1,1].bar(range(len(dow_sales)), dow_sales.values)
axes[1,1].set_xticks(range(len(dow_sales)))
axes[1,1].set_xticklabels(dow_sales.index, rotation=45)
axes[1,1].set_title('Average Sales by Day of Week')
axes[1,1].set_ylabel('Average Sales')

plt.tight_layout()
plt.show()

# Step 8: Test Hypotheses
print(f"\nStep 8: Hypothesis Testing")
# Test if weather significantly impacts sales
weather_groups = [df[df['weather'] == weather]['sales'] for weather in df['weather'].unique()]
f_stat, p_value = stats.f_oneway(*weather_groups)
print(f"Weather impact on sales - F-statistic: {f_stat:.3f}, p-value: {p_value:.3f}")

# Step 9: Validate Causes
print(f"\nStep 9: Root Cause Validation")
print("Key findings:")
if correlation_matrix.loc['sales', 'competitor_promotion'] < -0.1:
    print("✓ Competitor promotions significantly reduce sales")
if weather_sales.loc['Rainy', 'mean'] < weather_sales.mean()['mean']:
    print("✓ Rainy weather negatively impacts sales")
if correlation_matrix.loc['sales', 'temperature'] > 0.1:
    print("✓ Higher temperatures correlate with increased sales")
```

---

## 3. Predictive Analytics

### Definition

**What will happen?**
Uses historical data to forecast future outcomes. Employs statistical algorithms and machine learning techniques to identify patterns and predict likely future scenarios.

### 10-Step Workflow

1. **Define Problem** - Clearly specify what you want to predict
2. **Collect Data** - Gather relevant historical datasets
3. **Clean Data** - Prepare and preprocess the data
4. **Select Features** - Choose relevant variables for modeling
5. **Choose Model** - Pick appropriate algorithms for prediction
6. **Train Model** - Fit the model using historical data
7. **Test Accuracy** - Validate model performance on unseen data
8. **Tune Parameters** - Optimize model settings for better performance
9. **Make Forecasts** - Generate predictions for future periods
10. **Monitor Model** - Track performance and retrain as needed

### Python Example

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Step 1: Define Problem - Predict next month's sales
print("Predictive Analytics: Forecasting Sales for Next 30 Days")

# Step 2: Collect Data
np.random.seed(42)
dates = pd.date_range('2022-01-01', '2023-12-31', freq='D')
n_days = len(dates)

# Step 3: Create comprehensive dataset
data = {
    'date': dates,
    'temperature': np.random.normal(20, 10, n_days),
    'marketing_spend': np.random.normal(500, 150, n_days),
    'competitor_price': np.random.normal(25, 5, n_days),
    'day_of_week': [d.weekday() for d in dates],
    'month': [d.month for d in dates],
    'is_weekend': [d.weekday() >= 5 for d in dates],
    'is_holiday': np.random.choice([0, 1], n_days, p=[0.95, 0.05])
}

# Create realistic sales with seasonal patterns
seasonal_pattern = np.sin(2 * np.pi * np.arange(n_days) / 365) * 200
trend = np.linspace(0, 500, n_days)  # Growing trend
weekend_boost = np.array(data['is_weekend']) * 100
holiday_boost = np.array(data['is_holiday']) * 300

data['sales'] = (1000 +
                np.array(data['temperature']) * 10 +
                np.array(data['marketing_spend']) * 0.8 +
                seasonal_pattern +
                trend +
                weekend_boost +
                holiday_boost +
                np.random.normal(0, 80, n_days))

df = pd.DataFrame(data)
df['sales'] = np.maximum(df['sales'], 0)

# Step 3: Clean Data
print(f"Dataset shape: {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")

# Step 4: Select Features
feature_columns = ['temperature', 'marketing_spend', 'competitor_price',
                  'day_of_week', 'month', 'is_weekend', 'is_holiday']

# Create lagged features (previous day's sales)
df['sales_lag1'] = df['sales'].shift(1)
df['sales_lag7'] = df['sales'].shift(7)  # Previous week
df['marketing_ma7'] = df['marketing_spend'].rolling(7).mean()  # 7-day moving average

# Add lagged features to feature list
feature_columns.extend(['sales_lag1', 'sales_lag7', 'marketing_ma7'])

# Remove rows with NaN values created by lagging
df_clean = df.dropna()

X = df_clean[feature_columns]
y = df_clean['sales']

# Step 5: Choose Model & Split Data
# Use time-based split (important for time series)
split_date = '2023-07-01'
train_mask = df_clean['date'] < split_date
X_train, X_test = X[train_mask], X[~train_mask]
y_train, y_test = y[train_mask], y[~train_mask]

print(f"\nTraining set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Step 6: Train Models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

# Step 7: Test Accuracy & Step 8: Tune Parameters
results = {}
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

for name, model in models.items():
    print(f"\nTraining {name}...")

    if name == 'Linear Regression':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        # Grid search for Random Forest
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5]
        }
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='r2', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        y_pred = model.predict(X_test)
        print(f"Best parameters: {grid_search.best_params_}")

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results[name] = {
        'model': model,
        'predictions': y_pred,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R² Score: {r2:.3f}")

# Step 9: Make Forecasts
print(f"\nStep 9: Making Future Forecasts")

# Select best model
best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
best_model = results[best_model_name]['model']
print(f"Best model: {best_model_name}")

# Create future data for next 30 days
future_dates = pd.date_range('2024-01-01', periods=30, freq='D')
future_data = pd.DataFrame({
    'date': future_dates,
    'temperature': np.random.normal(15, 8, 30),  # Winter temperatures
    'marketing_spend': np.random.normal(600, 100, 30),  # Increased marketing
    'competitor_price': np.random.normal(26, 4, 30),
    'day_of_week': [d.weekday() for d in future_dates],
    'month': [d.month for d in future_dates],
    'is_weekend': [d.weekday() >= 5 for d in future_dates],
    'is_holiday': [1 if d.day == 1 else 0 for d in future_dates]  # New Year
})

# For lagged features, use recent actual data
recent_sales = df_clean['sales'].tail(7).values
future_predictions = []

for i in range(30):
    # Create features for prediction
    future_row = future_data.iloc[i:i+1].copy()

    # Add lagged features
    if i == 0:
        future_row['sales_lag1'] = recent_sales[-1]
        future_row['sales_lag7'] = recent_sales[-7]
    elif i < 7:
        future_row['sales_lag1'] = future_predictions[i-1] if i > 0 else recent_sales[-1]
        future_row['sales_lag7'] = recent_sales[-7+i]
    else:
        future_row['sales_lag1'] = future_predictions[i-1]
        future_row['sales_lag7'] = future_predictions[i-7]

    future_row['marketing_ma7'] = future_data['marketing_spend'].iloc[max(0,i-6):i+1].mean()

    # Make prediction
    if best_model_name == 'Linear Regression':
        features_scaled = scaler.transform(future_row[feature_columns])
        pred = best_model.predict(features_scaled)[0]
    else:
        pred = best_model.predict(future_row[feature_columns])[0]

    future_predictions.append(max(0, pred))  # Ensure non-negative

# Visualize results
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Model comparison
test_dates = df_clean[~train_mask]['date']
axes[0,0].plot(test_dates, y_test, label='Actual', alpha=0.7)
for name, result in results.items():
    axes[0,0].plot(test_dates, result['predictions'], label=f'{name} (R²={result["r2"]:.3f})', alpha=0.7)
axes[0,0].set_title('Model Performance Comparison')
axes[0,0].legend()
axes[0,0].tick_params(axis='x', rotation=45)

# Future forecasts
axes[0,1].plot(future_dates, future_predictions, 'ro-', linewidth=2, markersize=4)
axes[0,1].set_title('30-Day Sales Forecast')
axes[0,1].set_xlabel('Date')
axes[0,1].set_ylabel('Predicted Sales')
axes[0,1].tick_params(axis='x', rotation=45)

# Feature importance (for Random Forest)
if best_model_name == 'Random Forest':
    feature_imp = pd.DataFrame({
        'feature': feature_columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=True)

    axes[1,0].barh(feature_imp['feature'], feature_imp['importance'])
    axes[1,0].set_title('Feature Importance')
    axes[1,0].set_xlabel('Importance')

# Residuals analysis
best_predictions = results[best_model_name]['predictions']
residuals = y_test - best_predictions
axes[1,1].scatter(best_predictions, residuals, alpha=0.6)
axes[1,1].axhline(y=0, color='r', linestyle='--')
axes[1,1].set_xlabel('Predicted Values')
axes[1,1].set_ylabel('Residuals')
axes[1,1].set_title('Residuals vs Predicted')

plt.tight_layout()
plt.show()

# Step 10: Monitor Model
print(f"\nStep 10: Model Monitoring Summary")
print(f"Next 30 days forecast summary:")
print(f"Average daily sales: ${np.mean(future_predictions):,.2f}")
print(f"Total monthly sales: ${np.sum(future_predictions):,.2f}")
print(f"Highest sales day: ${np.max(future_predictions):,.2f}")
print(f"Lowest sales day: ${np.min(future_predictions):,.2f}")

print(f"\nModel should be retrained when:")
print(f"- R² score drops below 0.7 (currently {results[best_model_name]['r2']:.3f})")
print(f"- RMSE increases significantly (currently {results[best_model_name]['rmse']:.2f})")
print(f"- New data patterns emerge")
```

---

## 4. Prescriptive Analytics

### Definition

**What should we do?**
Recommends actions using data-driven insights and simulations. Goes beyond prediction to suggest optimal decisions and strategies based on various scenarios and constraints.

### 10-Step Workflow

1. **Define Goal** - Establish clear business objectives
2. **Collect Data** - Gather all relevant information and constraints
3. **Build Models** - Create simulation and optimization models
4. **Add Constraints** - Include business rules and limitations
5. **Run Simulations** - Test different scenarios and strategies
6. **Analyze Scenarios** - Evaluate outcomes of various options
7. **Optimize Output** - Find the best possible solution
8. **Recommend Actions** - Provide specific, actionable strategies
9. **Validate Strategy** - Test recommendations against reality
10. **Implement Plan** - Execute the recommended actions

### Python Example

```python
import pandas as pd
import numpy as np
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# Step 1: Define Goal - Maximize profit while maintaining service levels
print("Prescriptive Analytics: Optimizing Marketing Budget Allocation")
print("Goal: Maximize ROI while maintaining minimum sales targets")

# Step 2: Collect Data - Historical performance data
np.random.seed(42)

# Marketing channels and their historical performance
channels = ['Digital_Ads', 'Social_Media', 'Email', 'Print', 'Radio']
channel_data = {
    'Digital_Ads': {'cost_per_unit': 2.5, 'conversion_rate': 0.08, 'max_budget': 10000},
    'Social_Media': {'cost_per_unit': 1.5, 'conversion_rate': 0.05, 'max_budget': 8000},
    'Email': {'cost_per_unit': 0.1, 'conversion_rate': 0.03, 'max_budget': 2000},
    'Print': {'cost_per_unit': 5.0, 'conversion_rate': 0.02, 'max_budget': 15000},
    'Radio': {'cost_per_unit': 8.0, 'conversion_rate': 0.04, 'max_budget': 12000}
}

# Business constraints
total_budget = 30000
min_sales_target = 2000
avg_sale_value = 50
fixed_costs = 10000

print(f"\nBusiness Constraints:")
print(f"Total Budget: ${total_budget:,}")
print(f"Minimum Sales Target: {min_sales_target} units")
print(f"Average Sale Value: ${avg_sale_value}")

# Step 3: Build Models - ROI and Sales Prediction Models
def calculate_sales(budget_allocation):
    """Calculate expected sales from budget allocation"""
    total_sales = 0
    for i, channel in enumerate(channels):
        budget = budget_allocation[i]
        cost_per_unit = channel_data[channel]['cost_per_unit']
        conversion_rate = channel_data[channel]['conversion_rate']

        # Units that can be reached with budget
        units_reached = budget / cost_per_unit
        # Expected sales with diminishing returns
        expected_sales = units_reached * conversion_rate * (1 - np.exp(-budget/5000))
        total_sales += expected_sales

    return total_sales

def calculate_profit(budget_allocation):
    """Calculate profit from budget allocation"""
    sales = calculate_sales(budget_allocation)
    revenue = sales * avg_sale_value
    total_marketing_cost = sum(budget_allocation)
    profit = revenue - total_marketing_cost - fixed_costs
    return profit

def calculate_roi(budget_allocation):
    """Calculate ROI from budget allocation"""
    profit = calculate_profit(budget_allocation)
    total_investment = sum(budget_allocation) + fixed_costs
    roi = (profit / total_investment) * 100 if total_investment > 0 else 0
    return roi

# Step 4: Add Constraints
def constraint_budget(budget_allocation):
    """Budget constraint"""
    return total_budget - sum(budget_allocation)

def constraint_min_sales(budget_allocation):
    """Minimum sales constraint"""
    return calculate_sales(budget_allocation) - min_sales_target

def constraint_max_channel_budget(budget_allocation):
    """Maximum budget per channel constraints"""
    constraints = []
    for i, channel in enumerate(channels):
        max_budget = channel_data[channel]['max_budget']
        constraints.append(max_budget - budget_allocation[i])
    return np.array(constraints)

# Step 5: Run Simulations - Test different scenarios
print(f"\nStep 5: Running Scenario Simulations")

scenarios = {
    'Equal_Split': [total_budget/5] * 5,
    'Digital_Heavy': [15000, 8000, 2000, 3000, 2000],
    'Traditional_Heavy': [5000, 3000, 2000, 12000, 8000],
    'Cost_Effective': [8000, 6000, 2000, 7000, 7000]
}

scenario_results = {}
for name, allocation in scenarios.items():
    sales = calculate_sales(allocation)
    profit = calculate_profit(allocation)
    roi = calculate_roi(allocation)

    scenario_results[name] = {
        'allocation': allocation,
        'sales': sales,
        'profit': profit,
        'roi': roi,
        'meets_target': sales >= min_sales_target
    }

# Display scenario results
results_df = pd.DataFrame({
    name: {
        'Sales': f"{result['sales']:.0f}",
        'Profit': f"${result['profit']:,.0f}",
        'ROI': f"{result['roi']:.1f}%",
        'Meets_Target': result['meets_target']
    }
    for name, result in scenario_results.items()
}).T

print("Scenario Analysis Results:")
print(results_df)

# Step 6-7: Analyze Scenarios & Optimize Output
print(f"\nStep 7: Finding Optimal Solution")

# Optimization function (minimize negative profit = maximize profit)
def objective(budget_allocation):
    return -calculate_profit(budget_allocation)

# Bounds for each channel (0 to max budget)
bounds = [(0, channel_data[channel]['max_budget']) for channel in channels]

# Constraints for optimization
constraints = [
    {'type': 'eq', 'fun': constraint_budget},  # Budget constraint
    {'type': 'ineq', 'fun': constraint_min_sales}  # Min sales constraint
]

# Initial guess (equal split)
x0 = [total_budget/5] * 5

# Run optimization
result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)

if result.success:
    optimal_allocation = result.x
    optimal_sales = calculate_sales(optimal_allocation)
    optimal_profit = calculate_profit(optimal_allocation)
    optimal_roi = calculate_roi(optimal_allocation)

    print(f"Optimization successful!")
    print(f"Optimal Sales: {optimal_sales:.0f} units")
    print(f"Optimal Profit: ${optimal_profit:,.0f}")
    print(f"Optimal ROI: {optimal_roi:.1f}%")

    # Step 8: Recommend Actions
    print(f"\nStep 8: Recommended Budget Allocation")
    allocation_df = pd.DataFrame({
        'Channel': channels,
        'Recommended_Budget': [f"${x:,.0f}" for x in optimal_allocation],
        'Percentage_of_Total': [f"{(x/total_budget)*100:.1f}%" for x in optimal_allocation],
        'Expected_Sales': [calculate_sales([x if i==j else 0 for j in range(5)])
                          for i, x in enumerate(optimal_allocation)]
    })
    print(allocation_df)

else:
    print("Optimization failed. Using best scenario from simulations.")
    best_scenario = max(scenario_results.keys(),
                       key=lambda x: scenario_results[x]['profit']
                       if scenario_results[x]['meets_target'] else -float('inf'))
    optimal_allocation = scenarios[best_scenario]
    optimal_profit = scenario_results[best_scenario]['profit']
    optimal_roi = scenario_results[best_scenario]['roi']

# Step 9: Validate Strategy - Monte Carlo simulation
print(f"\nStep 9: Strategy Validation through Monte Carlo Simulation")

def monte_carlo_validation(allocation, n_simulations=1000):
    """Run Monte Carlo simulation to validate strategy"""
    profits = []
    sales_results = []

    for _ in range(n_simulations):
        # Add uncertainty to conversion rates (±20%)
        simulated_results = []
        total_simulated_sales = 0

        for i, channel in enumerate(channels):
            budget = allocation[i]
            base_conversion = channel_data[channel]['conversion_rate']
            cost_per_unit = channel_data[channel]['cost_per_unit']

            # Simulate conversion rate with uncertainty
            simulated_conversion = base_conversion * np.random.uniform(0.8, 1.2)

            units_reached = budget / cost_per_unit
            expected_sales = units_reached * simulated_conversion * (1 - np.exp(-budget/5000))
            total_simulated_sales += expected_sales

        # Calculate simulated profit
        simulated_revenue = total_simulated_sales * avg_sale_value * np.random.uniform(0.9, 1.1)
        simulated_profit = simulated_revenue - sum(allocation) - fixed_costs

        profits.append(simulated_profit)
        sales_results.append(total_simulated_sales)

    return np.array(profits), np.array(sales_results)

# Run Monte Carlo for optimal allocation
mc_profits, mc_sales = monte_carlo_validation(optimal_allocation)

print(f"Monte Carlo Results (1000 simulations):")
print(f"Expected Profit: ${np.mean(mc_profits):,.0f} ± ${np.std(mc_profits):,.0f}")
print(f"Expected Sales: {np.mean(mc_sales):.0f} ± {np.std(mc_sales):.0f} units")
print(f"Probability of meeting sales target: {(mc_sales >= min_sales_target).mean()*100:.1f}%")
print(f"Profit at risk (5th percentile): ${np.percentile(mc_profits, 5):,.0f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Budget allocation comparison
scenarios_for_plot = list(scenarios.keys()) + ['Optimal']
allocations_for_plot = list(scenarios.values()) + [optimal_allocation]

x_pos = np.arange(len(channels))
width = 0.15

for i, (name, allocation) in enumerate(zip(scenarios_for_plot, allocations_for_plot)):
    axes[0,0].bar(x_pos + i*width, allocation, width, label=name, alpha=0.8)

axes[0,0].set_xlabel('Marketing Channels')
axes[0,0].set_ylabel('Budget Allocation ($)')
axes[0,0].set_title('Budget Allocation Comparison')
axes[0,0].set_xticks(x_pos + width * (len(scenarios_for_plot)-1)/2)
axes[0,0].set_xticklabels(channels, rotation=45)
axes[0,0].legend()

# ROI comparison
roi_values = [scenario_results[name]['roi'] for name in scenarios.keys()] + [optimal_roi]
colors = ['red' if not scenario_results.get(name, {'meets_target': True})['meets_target']
          else 'green' for name in scenarios.keys()] + ['blue']

axes[0,1].bar(scenarios_for_plot, roi_values, color=colors, alpha=0.7)
axes[0,1].set_ylabel('ROI (%)')
axes[0,1].set_title('ROI Comparison by Strategy')
axes[0,1].tick_params(axis='x', rotation=45)
axes[0,1].axhline(y=0, color='black', linestyle='--', alpha=0.5)

# Monte Carlo profit distribution
axes[1,0].hist(mc_profits, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
axes[1,0].axvline(np.mean(mc_profits), color='red', linestyle='--',
                  label=f'Mean: ${np.mean(mc_profits):,.0f}')
axes[1,0].axvline(np.percentile(mc_profits, 5), color='orange', linestyle='--',
                  label=f'5th percentile: ${np.percentile(mc_profits, 5):,.0f}')
axes[1,0].set_xlabel('Profit ($)')
axes[1,0].set_ylabel('Frequency')
axes[1,0].set_title('Profit Distribution (Monte Carlo)')
axes[1,0].legend()

# Sales target achievement probability
axes[1,1].hist(mc_sales, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
axes[1,1].axvline(min_sales_target, color='red', linestyle='-', linewidth=2,
                  label=f'Sales Target: {min_sales_target}')
axes[1,1].axvline(np.mean(mc_sales), color='blue', linestyle='--',
                  label=f'Expected Sales: {np.mean(mc_sales):.0f}')
axes[1,1].set_xlabel('Sales (units)')
axes[1,1].set_ylabel('Frequency')
axes[1,1].set_title('Sales Distribution (Monte Carlo)')
axes[1,1].legend()

plt.tight_layout()
plt.show()

# Step 10: Implement Plan
print(f"\nStep 10: Implementation Plan")
print("="*60)

implementation_plan = {
    'Phase 1 (Week 1-2)': [
        'Set up tracking systems for each channel',
        'Allocate budgets according to optimal allocation',
        'Brief marketing teams on new strategy'
    ],
    'Phase 2 (Week 3-4)': [
        'Launch campaigns with allocated budgets',
        'Monitor daily performance metrics',
        'Adjust tactics within budget constraints'
    ],
    'Phase 3 (Week 5-8)': [
        'Collect performance data',
        'Compare actual vs predicted results',
        'Identify optimization opportunities'
    ],
    'Phase 4 (Week 9-12)': [
        'Analyze campaign effectiveness',
        'Update models with new data',
        'Plan next quarter\'s optimization'
    ]
}

for phase, tasks in implementation_plan.items():
    print(f"\n{phase}:")
    for task in tasks:
        print(f"  • {task}")

# Key Performance Indicators (KPIs) to monitor
print(f"\nKey Performance Indicators to Monitor:")
kpis = {
    'Sales Volume': f"Target: {min_sales_target}+ units/month",
    'ROI': f"Target: {optimal_roi:.1f}%+",
    'Cost per Acquisition': f"Target: <${(sum(optimal_allocation)/optimal_sales):.2f}",
    'Channel Performance': "Monitor individual channel ROI weekly",
    'Budget Utilization': "Track spend vs. allocation daily"
}

for kpi, target in kpis.items():
    print(f"  • {kpi}: {target}")

# Risk mitigation strategies
print(f"\nRisk Mitigation Strategies:")
risks = {
    'Market Changes': 'Weekly model retraining and sensitivity analysis',
    'Seasonal Variations': 'Monthly budget reallocation based on trends',
    'Competitor Actions': 'Bi-weekly competitive analysis and response planning',
    'Channel Performance': 'Real-time monitoring with 20% variance triggers',
    'Budget Overruns': 'Daily spend tracking with automatic alerts'
}

for risk, mitigation in risks.items():
    print(f"  • {risk}: {mitigation}")

# Expected outcomes summary
print(f"\nExpected Outcomes Summary:")
print(f"  • Monthly Sales: {optimal_sales:.0f} units ({((optimal_sales/min_sales_target-1)*100):+.1f}% vs target)")
print(f"  • Monthly Profit: ${optimal_profit:,.0f}")
print(f"  • ROI: {optimal_roi:.1f}%")
print(f"  • Success Probability: {(mc_sales >= min_sales_target).mean()*100:.1f}%")
print(f"  • Revenue Growth: ${(optimal_sales * avg_sale_value):,.0f}/month")

print(f"\n" + "="*60)
print("RECOMMENDATION: Implement the optimal allocation strategy")
print("with continuous monitoring and monthly model updates.")
print("="*60)
```

---

## Summary

These four types of analytics work together to provide a comprehensive approach to data-driven decision making:

- **Descriptive Analytics** tells you what happened by summarizing historical data
- **Diagnostic Analytics** explains why it happened through root cause analysis
- **Predictive Analytics** forecasts what will happen using statistical modeling
- **Prescriptive Analytics** recommends what should be done through optimization

Each type builds upon the previous one, creating a complete analytics framework that transforms raw data into actionable business insights and strategies.
