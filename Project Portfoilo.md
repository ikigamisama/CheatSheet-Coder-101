# Data Engineering & Data Science Portfolio (Final 5 Projects)

This portfolio (Oct–Dec 2025) is designed to align with my **astrology (Gemini 10th House MC, multiple Gemini placements)** and **Korean Saju themes (finance, career adaptability, and foreign expansion)**.

It showcases **ETL/ELT pipelines, clustering, A/B testing, big data, and streaming analytics** with clear **business value**.

---

## 🌟 Final 5 Projects

---

### 1. 🛒 E-commerce Pipeline + A/B Testing

**Why**

- Classic business use case (recruiter-friendly).
- Demonstrates **ETL pipeline, medallion architecture, Airflow orchestration, and experimentation design**.
- Links to Gemini storytelling (understanding customer behavior).

**Steps**

1. Ingest e-commerce sales, customer, and product data.
2. Store in **PostgreSQL/DuckDB**, raw layer = bronze.
3. Build transformations with **dbt** → silver (cleaned) → gold (aggregated metrics).
4. Simulate **A/B experiment** (checkout flow A vs B).
5. Analyze lift: conversion, revenue, average order value.
6. Build dashboard for experiment results.

**Tools**

- Airflow (orchestration)
- dbt (transformations)
- PostgreSQL/DuckDB + MinIO (storage)
- Python (pandas, statsmodels for A/B tests)
- Streamlit/PowerBI (dashboard)

**Datasets**

- [Olist Brazilian E-commerce (Kaggle)](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
- Synthetic A/B data (Python Faker simulation)

**Deliverables**

- Airflow DAGs
- dbt models (bronze → gold)
- SQL queries for experiment metrics
- A/B analysis notebook (frequentist & Bayesian)
- Dashboard with funnel + experiment results

---

### 2. 👥 Customer Segmentation (Clustering)

**Why**

- Classic **clustering project** → actionable insights for businesses.
- Recruiters value segmentation for marketing & personalization.
- Astrology: Gemini adaptability = grouping diverse customers.

**Steps**

1. Ingest customer purchase behavior (frequency, recency, monetary value).
2. Preprocess data (feature scaling, encoding categories).
3. Apply clustering algorithms: KMeans, DBSCAN, Gaussian Mixture Models.
4. Evaluate clusters with Silhouette Score & domain validation.
5. Create customer personas (e.g., “loyal premium buyers”, “bargain hunters”).
6. Visualize clusters with PCA/UMAP.

**Tools**

- Airflow + dbt (pipeline + transformations)
- Python (scikit-learn, pandas, matplotlib/seaborn)
- PostgreSQL/DuckDB (storage)
- Streamlit/Plotly Dash (dashboard)

**Datasets**

- [Olist E-commerce Dataset (Kaggle)](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

**Deliverables**

- Cleaned dataset pipeline
- Clustering notebook + evaluation metrics
- Persona summary report (business-friendly)
- Interactive cluster visualization dashboard

---
