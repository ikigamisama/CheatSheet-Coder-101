# 10 Clustering Datasets Template for Data Modeling

## Dataset 1: Customer Segmentation

**Use Case**: E-commerce customer behavior analysis
**Clustering Goal**: Identify customer segments for targeted marketing

### Features:

- `customer_id`: Unique identifier
- `age`: Customer age (18-80)
- `annual_income`: Annual income in thousands (20-150)
- `spending_score`: Spending behavior score (1-100)
- `frequency_purchases`: Number of purchases per month (0-20)
- `avg_order_value`: Average order value in dollars (10-500)
- `days_since_last_purchase`: Days since last purchase (0-365)
- `preferred_category`: Most purchased category (Electronics, Clothing, Books, Home, Sports)

### Expected Clusters:

- High-value customers (high income, high spending)
- Budget-conscious customers (low income, low spending)
- Occasional buyers (medium income, low frequency)
- Loyal customers (high frequency, consistent spending)

---

## Dataset 2: Market Segmentation

**Use Case**: Geographic and demographic market analysis
**Clustering Goal**: Identify market segments for business expansion

### Features:

- `region_id`: Geographic region identifier
- `population_density`: People per square km (10-5000)
- `median_age`: Median age of population (25-65)
- `education_level`: Average education score (1-10)
- `unemployment_rate`: Unemployment percentage (2-15)
- `median_income`: Median household income in thousands (30-120)
- `internet_penetration`: Internet usage percentage (40-98)
- `urbanization_index`: Urban development score (1-10)

### Expected Clusters:

- Urban high-income areas
- Rural developing regions
- Suburban middle-class areas
- Industrial/working-class regions

---

## Dataset 3: Employee Performance Analysis

**Use Case**: HR analytics and workforce optimization
**Clustering Goal**: Identify employee performance patterns

### Features:

- `employee_id`: Unique identifier
- `years_experience`: Total work experience (0-30)
- `performance_score`: Annual performance rating (1-10)
- `training_hours`: Hours of training completed (0-200)
- `projects_completed`: Number of projects finished (1-50)
- `collaboration_score`: Teamwork rating (1-10)
- `innovation_index`: Innovation contribution score (1-10)
- `attendance_rate`: Attendance percentage (85-100)
- `overtime_hours`: Monthly overtime hours (0-80)

### Expected Clusters:

- High performers (high scores across metrics)
- Steady contributors (consistent performance)
- Developing talent (lower experience, high potential)
- At-risk employees (low performance indicators)

---

## Dataset 4: Healthcare Patient Segmentation

**Use Case**: Patient care optimization and resource allocation
**Clustering Goal**: Group patients by health risk and care needs

### Features:

- `patient_id`: Unique identifier
- `age`: Patient age (0-100)
- `bmi`: Body Mass Index (15-50)
- `blood_pressure_systolic`: Systolic BP (90-200)
- `cholesterol_level`: Cholesterol mg/dL (120-350)
- `glucose_level`: Blood glucose mg/dL (70-400)
- `hospital_visits`: Annual visits (0-50)
- `medication_count`: Number of medications (0-20)
- `chronic_conditions`: Number of chronic conditions (0-8)
- `lifestyle_score`: Healthy lifestyle rating (1-10)

### Expected Clusters:

- Healthy patients (low risk, minimal care)
- Chronic disease patients (high medication, frequent visits)
- High-risk patients (multiple conditions)
- Preventive care candidates (moderate risk)

---

## Dataset 5: Product Performance Clustering

**Use Case**: Product portfolio optimization
**Clustering Goal**: Group products by performance characteristics

### Features:

- `product_id`: Unique product identifier
- `price`: Product price in dollars (5-1000)
- `sales_volume`: Monthly units sold (10-5000)
- `profit_margin`: Profit margin percentage (5-60)
- `customer_rating`: Average rating (1.0-5.0)
- `return_rate`: Return percentage (0-25)
- `inventory_turnover`: Inventory turns per year (2-24)
- `seasonality_factor`: Seasonal demand variation (0.5-2.0)
- `marketing_spend`: Monthly marketing budget (100-10000)

### Expected Clusters:

- High-performers (high sales, margin, rating)
- Cash cows (steady sales, good margin)
- Problem products (low sales, high returns)
- Growth products (increasing sales, high marketing)

---

## Dataset 6: Financial Risk Assessment

**Use Case**: Credit scoring and loan default prediction
**Clustering Goal**: Segment customers by financial risk profiles

### Features:

- `client_id`: Unique identifier
- `credit_score`: Credit score (300-850)
- `debt_to_income_ratio`: Debt-to-income percentage (0-80)
- `loan_amount`: Requested loan amount (1000-500000)
- `employment_length`: Years in current job (0-40)
- `monthly_income`: Monthly income (2000-50000)
- `payment_history`: Payment history score (1-10)
- `credit_utilization`: Credit utilization percentage (0-100)
- `number_of_accounts`: Number of credit accounts (1-30)

### Expected Clusters:

- Low-risk borrowers (high credit score, stable income)
- Medium-risk borrowers (average metrics)
- High-risk borrowers (low credit score, high debt)
- Premium customers (high income, excellent credit)

---

## Dataset 7: Social Media User Behavior

**Use Case**: Social platform optimization and content strategy
**Clustering Goal**: Identify user engagement patterns

### Features:

- `user_id`: Unique identifier
- `daily_active_minutes`: Minutes spent daily (5-480)
- `posts_per_week`: Number of posts per week (0-50)
- `likes_given`: Likes given per week (0-1000)
- `shares_per_week`: Content shared per week (0-100)
- `followers_count`: Number of followers (0-100000)
- `engagement_rate`: Engagement rate percentage (0-15)
- `content_type_preference`: Primary content type (Video, Image, Text, Link)
- `peak_activity_hour`: Most active hour (0-23)

### Expected Clusters:

- Power users (high activity, high engagement)
- Casual browsers (low posting, moderate consumption)
- Content creators (high posts, moderate followers)
- Lurkers (high consumption, low interaction)

---

## Dataset 8: Supply Chain Optimization

**Use Case**: Supplier performance and logistics optimization
**Clustering Goal**: Segment suppliers by performance metrics

### Features:

- `supplier_id`: Unique identifier
- `delivery_time_avg`: Average delivery days (1-30)
- `quality_score`: Quality rating (1-10)
- `price_competitiveness`: Price competitiveness score (1-10)
- `order_fulfillment_rate`: Fulfillment percentage (70-100)
- `defect_rate`: Defect percentage (0-15)
- `payment_terms`: Payment terms in days (15-90)
- `geographic_distance`: Distance in km (10-10000)
- `annual_volume`: Annual order volume (1000-1000000)

### Expected Clusters:

- Premium suppliers (high quality, fast delivery)
- Cost-effective suppliers (good price, average quality)
- Reliable suppliers (consistent performance)
- Problematic suppliers (quality/delivery issues)

---

## Dataset 9: Student Academic Performance

**Use Case**: Educational analytics and personalized learning
**Clustering Goal**: Group students by learning patterns and needs

### Features:

- `student_id`: Unique identifier
- `gpa`: Grade Point Average (0.0-4.0)
- `study_hours_weekly`: Hours studied per week (0-60)
- `attendance_rate`: Class attendance percentage (60-100)
- `assignment_completion`: Assignment completion rate (40-100)
- `exam_scores_avg`: Average exam scores (0-100)
- `participation_score`: Class participation rating (1-10)
- `extracurricular_hours`: Weekly extracurricular hours (0-20)
- `preferred_learning_style`: Learning preference (Visual, Auditory, Kinesthetic, Reading)

### Expected Clusters:

- High achievers (high GPA, consistent performance)
- Struggling students (low GPA, poor attendance)
- Balanced students (good academics, active lifestyle)
- Underperformers (capable but inconsistent)

---

## Dataset 10: Smart City Infrastructure

**Use Case**: Urban planning and resource allocation
**Clustering Goal**: Optimize city services and infrastructure

### Features:

- `zone_id`: City zone identifier
- `population_density`: People per square km (100-10000)
- `traffic_volume`: Daily vehicle count (500-50000)
- `energy_consumption`: Daily kWh consumption (1000-100000)
- `waste_generation`: Daily waste in tons (0.5-50)
- `water_usage`: Daily water usage in liters (10000-1000000)
- `green_space_ratio`: Green space percentage (5-60)
- `public_transport_usage`: Daily public transport users (100-20000)
- `crime_incidents`: Monthly crime incidents (0-100)
- `air_quality_index`: Air quality score (20-300)

### Expected Clusters:

- Residential areas (high population, low traffic)
- Commercial districts (high energy, high waste)
- Industrial zones (high pollution, low green space)
- Mixed-use areas (balanced metrics)

---

## Data Generation Guidelines

### Sample Size Recommendations:

- **Small dataset**: 500-1,000 records (good for testing)
- **Medium dataset**: 2,000-5,000 records (standard analysis)
- **Large dataset**: 10,000+ records (production scenarios)

### Data Quality Considerations:

1. **Realistic Distributions**: Use normal, log-normal, or uniform distributions as appropriate
2. **Feature Correlations**: Implement logical relationships between features
3. **Categorical Balance**: Ensure reasonable distribution across categories
4. **Outlier Injection**: Include 1-3% outliers for robustness testing
5. **Missing Values**: Add 2-5% missing values strategically

### Preprocessing Checklist:

- [ ] Handle missing values (imputation/removal)
- [ ] Scale numerical features (StandardScaler/MinMaxScaler)
- [ ] Encode categorical variables (OneHot/Label encoding)
- [ ] Remove or treat outliers
- [ ] Feature selection/PCA if high dimensionality
- [ ] Check feature correlations

### Clustering Algorithms by Dataset Type:

- **Spherical clusters**: K-Means, K-Medoids
- **Hierarchical structure**: Agglomerative/Divisive clustering
- **Density-based**: DBSCAN, OPTICS
- **Probabilistic**: Gaussian Mixture Models
- **High-dimensional**: Spectral clustering

### Evaluation Metrics:

- **Silhouette Score**: Cluster cohesion and separation (-1 to 1)
- **Calinski-Harabasz Index**: Ratio of between/within cluster variance
- **Davies-Bouldin Index**: Average similarity between clusters (lower is better)
- **Inertia/WCSS**: Within-cluster sum of squares
- **Adjusted Rand Index**: Similarity to ground truth (if available)

### Visualization Techniques:

- **2D/3D Scatter plots**: For low-dimensional data
- **t-SNE/UMAP**: For high-dimensional visualization
- **Parallel coordinates**: For multi-dimensional features
- **Heatmaps**: For cluster centers comparison
- **Dendrograms**: For hierarchical clustering

### Business Value Extraction:

1. **Cluster Profiling**: Describe each cluster's characteristics
2. **Actionable Insights**: Translate clusters to business strategies
3. **Performance Monitoring**: Track cluster stability over time
4. **Personalization**: Use clusters for targeted approaches
5. **Resource Allocation**: Optimize resources based on cluster needs
