# E-Commerce - Data Modeling Template

## Medallion Architecture Overview

- **Bronze Layer**: Raw e-commerce data from web analytics, orders, products, customers
- **Silver Layer**: Cleaned, validated, and enriched e-commerce data
- **Gold Layer**: Business-ready analytics with star schema for sales, customer behavior, and product performance

---

## Bronze Layer - Raw E-Commerce Data

### orders_raw

| column_name         | type      | description                       |
| ------------------- | --------- | --------------------------------- |
| order_id            | string    | Raw order identifier              |
| customer_id         | string    | Customer identifier               |
| order_date          | string    | Order date from source            |
| order_status        | string    | Order status                      |
| total_amount        | string    | Total order amount                |
| subtotal            | string    | Subtotal before tax and shipping  |
| tax_amount          | string    | Tax amount                        |
| shipping_amount     | string    | Shipping cost                     |
| discount_amount     | string    | Discount applied                  |
| currency_code       | string    | Order currency                    |
| payment_method      | string    | Payment method used               |
| shipping_address    | string    | Shipping address                  |
| billing_address     | string    | Billing address                   |
| coupon_code         | string    | Coupon code used                  |
| sales_channel       | string    | Sales channel (web, mobile, etc.) |
| ingestion_timestamp | timestamp | When record was ingested          |
| source_system       | string    | Source system identifier          |

### order_items_raw

| column_name         | type      | description               |
| ------------------- | --------- | ------------------------- |
| order_item_id       | string    | Raw order item identifier |
| order_id            | string    | Order identifier          |
| product_id          | string    | Product identifier        |
| product_sku         | string    | Product SKU               |
| product_name        | string    | Product name              |
| quantity            | string    | Quantity ordered          |
| unit_price          | string    | Unit price                |
| total_price         | string    | Total line item price     |
| discount_amount     | string    | Item discount amount      |
| category            | string    | Product category          |
| brand               | string    | Product brand             |
| ingestion_timestamp | timestamp | When record was ingested  |
| source_system       | string    | Source system identifier  |

### customers_raw

| column_name         | type      | description               |
| ------------------- | --------- | ------------------------- |
| customer_id         | string    | Raw customer identifier   |
| email               | string    | Customer email            |
| first_name          | string    | Customer first name       |
| last_name           | string    | Customer last name        |
| phone               | string    | Phone number              |
| date_of_birth       | string    | Date of birth             |
| gender              | string    | Gender                    |
| registration_date   | string    | Account registration date |
| last_login_date     | string    | Last login date           |
| customer_status     | string    | Customer status           |
| preferred_language  | string    | Preferred language        |
| marketing_opt_in    | string    | Marketing opt-in flag     |
| address_line1       | string    | Address line 1            |
| address_line2       | string    | Address line 2            |
| city                | string    | City                      |
| state               | string    | State/Province            |
| postal_code         | string    | Postal code               |
| country             | string    | Country                   |
| ingestion_timestamp | timestamp | When record was ingested  |
| source_system       | string    | Source system identifier  |

### products_raw

| column_name         | type      | description              |
| ------------------- | --------- | ------------------------ |
| product_id          | string    | Raw product identifier   |
| product_sku         | string    | Product SKU              |
| product_name        | string    | Product name             |
| description         | string    | Product description      |
| category            | string    | Product category         |
| subcategory         | string    | Product subcategory      |
| brand               | string    | Product brand            |
| price               | string    | Current price            |
| cost                | string    | Product cost             |
| weight              | string    | Product weight           |
| dimensions          | string    | Product dimensions       |
| color               | string    | Product color            |
| size                | string    | Product size             |
| stock_quantity      | string    | Current stock quantity   |
| status              | string    | Product status           |
| launch_date         | string    | Product launch date      |
| ingestion_timestamp | timestamp | When record was ingested |
| source_system       | string    | Source system identifier |

### web_events_raw

| column_name         | type      | description                                         |
| ------------------- | --------- | --------------------------------------------------- |
| event_id            | string    | Raw event identifier                                |
| session_id          | string    | Session identifier                                  |
| customer_id         | string    | Customer identifier (if logged in)                  |
| timestamp           | string    | Event timestamp                                     |
| event_type          | string    | Type of event (page_view, click, add_to_cart, etc.) |
| page_url            | string    | Page URL                                            |
| referrer_url        | string    | Referrer URL                                        |
| user_agent          | string    | User agent string                                   |
| ip_address          | string    | IP address                                          |
| product_id          | string    | Product ID (if applicable)                          |
| category            | string    | Category viewed/interacted with                     |
| search_term         | string    | Search term (if search event)                       |
| device_type         | string    | Device type                                         |
| browser             | string    | Browser type                                        |
| operating_system    | string    | Operating system                                    |
| country             | string    | Country from IP                                     |
| ingestion_timestamp | timestamp | When record was ingested                            |
| source_system       | string    | Source system identifier                            |

---

## Silver Layer - Cleaned E-Commerce Data

### orders_clean

| column_name           | type          | description                                                                        |
| --------------------- | ------------- | ---------------------------------------------------------------------------------- |
| order_id              | bigint        | Cleaned order identifier                                                           |
| customer_id           | bigint        | Customer identifier                                                                |
| order_date            | date          | Validated order date                                                               |
| order_timestamp       | timestamp     | Order timestamp                                                                    |
| order_status          | string        | Standardized status (Pending, Processing, Shipped, Delivered, Cancelled, Returned) |
| total_amount          | decimal(10,2) | Total order amount                                                                 |
| subtotal              | decimal(10,2) | Subtotal before tax and shipping                                                   |
| tax_amount            | decimal(8,2)  | Tax amount                                                                         |
| shipping_amount       | decimal(8,2)  | Shipping cost                                                                      |
| discount_amount       | decimal(8,2)  | Total discount applied                                                             |
| currency_code         | string        | Order currency                                                                     |
| payment_method        | string        | Standardized payment method                                                        |
| sales_channel         | string        | Standardized sales channel                                                         |
| coupon_code           | string        | Coupon code used                                                                   |
| is_first_order        | boolean       | First order indicator                                                              |
| days_since_last_order | int           | Days since customer's last order                                                   |
| order_value_tier      | string        | Order value tier (Low, Medium, High, Premium)                                      |
| shipping_country      | string        | Shipping country                                                                   |
| shipping_region       | string        | Shipping region                                                                    |
| created_date          | timestamp     | Record creation timestamp                                                          |
| updated_date          | timestamp     | Last update timestamp                                                              |

### order_items_clean

| column_name       | type          | description                   |
| ----------------- | ------------- | ----------------------------- |
| order_item_id     | bigint        | Cleaned order item identifier |
| order_id          | bigint        | Order identifier              |
| product_id        | bigint        | Product identifier            |
| product_sku       | string        | Product SKU                   |
| quantity          | int           | Quantity ordered              |
| unit_price        | decimal(8,2)  | Unit price at time of order   |
| total_price       | decimal(10,2) | Total line item price         |
| discount_amount   | decimal(8,2)  | Item discount amount          |
| margin_amount     | decimal(8,2)  | Profit margin                 |
| margin_percentage | decimal(5,2)  | Margin percentage             |
| product_category  | string        | Product category              |
| product_brand     | string        | Product brand                 |
| is_returned       | boolean       | Return indicator              |
| return_reason     | string        | Return reason if applicable   |
| created_date      | timestamp     | Record creation timestamp     |
| updated_date      | timestamp     | Last update timestamp         |

### customers_clean

| column_name            | type      | description                                       |
| ---------------------- | --------- | ------------------------------------------------- |
| customer_id            | bigint    | Cleaned customer identifier                       |
| email                  | string    | Validated email address                           |
| first_name             | string    | Standardized first name                           |
| last_name              | string    | Standardized last name                            |
| full_name              | string    | Full customer name                                |
| phone                  | string    | Formatted phone number                            |
| date_of_birth          | date      | Date of birth                                     |
| age                    | int       | Calculated age                                    |
| age_group              | string    | Age group category                                |
| gender                 | string    | Standardized gender                               |
| registration_date      | date      | Account registration date                         |
| last_login_date        | date      | Last login date                                   |
| customer_status        | string    | Standardized status (Active, Inactive, Suspended) |
| preferred_language     | string    | Preferred language                                |
| marketing_opt_in       | boolean   | Marketing consent flag                            |
| country                | string    | Primary country                                   |
| region                 | string    | Primary region                                    |
| city                   | string    | Primary city                                      |
| customer_lifetime_days | int       | Days since registration                           |
| is_active              | boolean   | Active customer indicator                         |
| created_date           | timestamp | Record creation timestamp                         |
| updated_date           | timestamp | Last update timestamp                             |

### products_clean

| column_name       | type         | description                                              |
| ----------------- | ------------ | -------------------------------------------------------- |
| product_id        | bigint       | Cleaned product identifier                               |
| product_sku       | string       | Product SKU                                              |
| product_name      | string       | Standardized product name                                |
| description       | string       | Product description                                      |
| category          | string       | Standardized category                                    |
| subcategory       | string       | Standardized subcategory                                 |
| brand             | string       | Standardized brand                                       |
| current_price     | decimal(8,2) | Current selling price                                    |
| cost              | decimal(8,2) | Product cost                                             |
| margin_amount     | decimal(8,2) | Profit margin                                            |
| margin_percentage | decimal(5,2) | Margin percentage                                        |
| weight_grams      | decimal(8,2) | Weight in grams                                          |
| color             | string       | Standardized color                                       |
| size              | string       | Standardized size                                        |
| current_stock     | int          | Current stock quantity                                   |
| product_status    | string       | Standardized status (Active, Discontinued, Out of Stock) |
| launch_date       | date         | Product launch date                                      |
| days_since_launch | int          | Days since product launch                                |
| is_active         | boolean      | Active product indicator                                 |
| created_date      | timestamp    | Record creation timestamp                                |
| updated_date      | timestamp    | Last update timestamp                                    |

### web_events_clean

| column_name          | type      | description                                         |
| -------------------- | --------- | --------------------------------------------------- |
| event_id             | bigint    | Cleaned event identifier                            |
| session_id           | string    | Session identifier                                  |
| customer_id          | bigint    | Customer identifier (null if anonymous)             |
| event_timestamp      | timestamp | Event timestamp                                     |
| event_type           | string    | Standardized event type                             |
| page_url             | string    | Cleaned page URL                                    |
| page_category        | string    | Page category                                       |
| referrer_url         | string    | Referrer URL                                        |
| referrer_type        | string    | Referrer type (Direct, Search, Social, Email, etc.) |
| product_id           | bigint    | Product ID (if applicable)                          |
| search_term          | string    | Search term (if search event)                       |
| device_type          | string    | Standardized device type (Desktop, Mobile, Tablet)  |
| browser              | string    | Browser type                                        |
| operating_system     | string    | Operating system                                    |
| country              | string    | Country from IP geolocation                         |
| region               | string    | Region from IP geolocation                          |
| is_conversion        | boolean   | Conversion event indicator                          |
| session_sequence     | int       | Event sequence within session                       |
| time_on_page_seconds | int       | Time spent on page                                  |
| created_date         | timestamp | Record creation timestamp                           |

---

## Gold Layer - E-Commerce Analytics Star Schema

### Dimension Tables

#### dim_customer

| column_name         | type      | description                                   |
| ------------------- | --------- | --------------------------------------------- |
| customer_key        | bigint    | Surrogate key for customer dimension          |
| customer_id         | bigint    | Business key from source system               |
| customer_name       | string    | Full customer name                            |
| email               | string    | Customer email                                |
| age_group           | string    | Age group category                            |
| gender              | string    | Customer gender                               |
| country             | string    | Customer country                              |
| region              | string    | Customer region                               |
| city                | string    | Customer city                                 |
| registration_date   | date      | Account registration date                     |
| customer_segment    | string    | Customer segment (New, Regular, VIP, Churned) |
| lifetime_value_tier | string    | LTV tier (Low, Medium, High, Premium)         |
| preferred_language  | string    | Preferred language                            |
| marketing_opt_in    | boolean   | Marketing consent                             |
| acquisition_channel | string    | How customer was acquired                     |
| is_active           | boolean   | Active customer indicator                     |
| is_current          | boolean   | Current record indicator                      |
| effective_date      | date      | Record effective date                         |
| expiry_date         | date      | Record expiry date                            |
| created_date        | timestamp | Record creation timestamp                     |

#### dim_product

| column_name             | type      | description                                      |
| ----------------------- | --------- | ------------------------------------------------ |
| product_key             | bigint    | Surrogate key for product dimension              |
| product_id              | bigint    | Business key from source system                  |
| product_sku             | string    | Product SKU                                      |
| product_name            | string    | Product name                                     |
| category                | string    | Product category                                 |
| subcategory             | string    | Product subcategory                              |
| brand                   | string    | Product brand                                    |
| color                   | string    | Product color                                    |
| size                    | string    | Product size                                     |
| price_tier              | string    | Price tier (Budget, Mid-range, Premium, Luxury)  |
| margin_tier             | string    | Margin tier (Low, Medium, High)                  |
| launch_date             | date      | Product launch date                              |
| product_lifecycle_stage | string    | Lifecycle stage (New, Growth, Mature, Decline)   |
| seasonal_indicator      | string    | Seasonal pattern (Year-round, Seasonal, Holiday) |
| is_active               | boolean   | Active product indicator                         |
| is_current              | boolean   | Current record indicator                         |
| effective_date          | date      | Record effective date                            |
| expiry_date             | date      | Record expiry date                               |
| created_date            | timestamp | Record creation timestamp                        |

#### dim_date

| column_name     | type    | description                                     |
| --------------- | ------- | ----------------------------------------------- |
| date_key        | int     | Surrogate key (YYYYMMDD format)                 |
| date            | date    | Actual date                                     |
| year            | int     | Year                                            |
| quarter         | int     | Quarter                                         |
| month           | int     | Month                                           |
| month_name      | string  | Month name                                      |
| week_of_year    | int     | Week of year                                    |
| day_of_month    | int     | Day of month                                    |
| day_of_week     | int     | Day of week                                     |
| day_name        | string  | Day name                                        |
| is_weekend      | boolean | Weekend indicator                               |
| is_holiday      | boolean | Holiday indicator                               |
| is_black_friday | boolean | Black Friday indicator                          |
| is_cyber_monday | boolean | Cyber Monday indicator                          |
| is_prime_day    | boolean | Prime Day indicator (if applicable)             |
| shopping_season | string  | Shopping season (Back-to-School, Holiday, etc.) |

#### dim_geography

| column_name    | type         | description                           |
| -------------- | ------------ | ------------------------------------- |
| geography_key  | bigint       | Surrogate key for geography dimension |
| country_code   | string       | ISO country code                      |
| country_name   | string       | Country name                          |
| region_name    | string       | Region name                           |
| subregion_name | string       | Subregion name                        |
| continent      | string       | Continent                             |
| currency_code  | string       | Local currency                        |
| timezone       | string       | Primary timezone                      |
| is_eu          | boolean      | European Union indicator              |
| is_domestic    | boolean      | Domestic market indicator             |
| shipping_zone  | string       | Shipping zone                         |
| tax_rate       | decimal(5,4) | Standard tax rate                     |

#### dim_sales_channel

| column_name      | type         | description                                        |
| ---------------- | ------------ | -------------------------------------------------- |
| channel_key      | bigint       | Surrogate key for sales channel dimension          |
| channel_id       | string       | Channel identifier                                 |
| channel_name     | string       | Channel name                                       |
| channel_type     | string       | Channel type (Online, Mobile, Marketplace, Retail) |
| channel_category | string       | Channel category                                   |
| is_owned         | boolean      | Owned channel indicator                            |
| commission_rate  | decimal(5,4) | Commission rate                                    |
| is_active        | boolean      | Active channel indicator                           |

### Fact Tables

#### fact_sales

| column_name          | type          | description                      |
| -------------------- | ------------- | -------------------------------- |
| sales_key            | bigint        | Surrogate key for fact table     |
| order_date_key       | int           | Foreign key to dim_date          |
| customer_key         | bigint        | Foreign key to dim_customer      |
| product_key          | bigint        | Foreign key to dim_product       |
| geography_key        | bigint        | Foreign key to dim_geography     |
| channel_key          | bigint        | Foreign key to dim_sales_channel |
| order_id             | bigint        | Business key from source         |
| order_item_id        | bigint        | Order item identifier            |
| quantity_sold        | int           | Quantity sold                    |
| unit_price           | decimal(8,2)  | Unit price                       |
| gross_sales          | decimal(10,2) | Gross sales amount               |
| discount_amount      | decimal(8,2)  | Discount amount                  |
| net_sales            | decimal(10,2) | Net sales amount                 |
| cost_of_goods        | decimal(10,2) | Cost of goods sold               |
| gross_profit         | decimal(10,2) | Gross profit                     |
| tax_amount           | decimal(8,2)  | Tax amount                       |
| shipping_amount      | decimal(8,2)  | Shipping amount                  |
| total_amount         | decimal(10,2) | Total transaction amount         |
| currency_code        | string        | Transaction currency             |
| exchange_rate        | decimal(10,6) | Exchange rate to base currency   |
| base_currency_amount | decimal(10,2) | Amount in base currency          |
| order_sequence       | int           | Customer's order sequence number |
| is_first_purchase    | boolean       | First purchase indicator         |
| is_return            | boolean       | Return indicator                 |
| return_reason        | string        | Return reason                    |
| payment_method       | string        | Payment method used              |
| created_date         | timestamp     | Record creation timestamp        |

#### fact_customer_behavior

| column_name              | type         | description                       |
| ------------------------ | ------------ | --------------------------------- |
| behavior_key             | bigint       | Surrogate key for fact table      |
| event_date_key           | int          | Foreign key to dim_date           |
| customer_key             | bigint       | Foreign key to dim_customer       |
| product_key              | bigint       | Foreign key to dim_product        |
| geography_key            | bigint       | Foreign key to dim_geography      |
| channel_key              | bigint       | Foreign key to dim_sales_channel  |
| session_id               | string       | Session identifier                |
| event_type               | string       | Event type                        |
| page_views               | int          | Number of page views              |
| unique_pages_viewed      | int          | Unique pages viewed               |
| session_duration_minutes | decimal(8,2) | Session duration in minutes       |
| bounce_indicator         | boolean      | Bounce session indicator          |
| conversion_indicator     | boolean      | Conversion indicator              |
| add_to_cart_count        | int          | Add to cart events                |
| remove_from_cart_count   | int          | Remove from cart events           |
| search_count             | int          | Search events                     |
| product_views            | int          | Product view events               |
| category_views           | int          | Category view events              |
| time_to_purchase_hours   | decimal(8,2) | Time from first visit to purchase |
| referrer_type            | string       | Traffic source type               |
| device_type              | string       | Device type used                  |
| created_date             | timestamp    | Record creation timestamp         |

#### fact_inventory

| column_name          | type          | description                  |
| -------------------- | ------------- | ---------------------------- |
| inventory_key        | bigint        | Surrogate key for fact table |
| date_key             | int           | Foreign key to dim_date      |
| product_key          | bigint        | Foreign key to dim_product   |
| geography_key        | bigint        | Foreign key to dim_geography |
| beginning_inventory  | int           | Starting inventory           |
| ending_inventory     | int           | Ending inventory             |
| inventory_received   | int           | Inventory received           |
| inventory_sold       | int           | Inventory sold               |
| inventory_adjustment | int           | Inventory adjustments        |
| inventory_shrinkage  | int           | Inventory shrinkage          |
| stockout_days        | int           | Days out of stock            |
| inventory_value      | decimal(12,2) | Total inventory value        |
| average_inventory    | decimal(8,2)  | Average inventory level      |
| inventory_turnover   | decimal(6,3)  | Inventory turnover ratio     |
| days_of_supply       | int           | Days of inventory supply     |
| reorder_point        | int           | Reorder point quantity       |
| safety_stock         | int           | Safety stock level           |
| is_stockout          | boolean       | Stockout indicator           |
| is_overstock         | boolean       | Overstock indicator          |
| created_date         | timestamp     | Record creation timestamp    |

#### fact_customer_lifetime_value

| column_name               | type          | description                                                |
| ------------------------- | ------------- | ---------------------------------------------------------- |
| clv_key                   | bigint        | Surrogate key for fact table                               |
| snapshot_date_key         | int           | Foreign key to dim_date                                    |
| customer_key              | bigint        | Foreign key to dim_customer                                |
| geography_key             | bigint        | Foreign key to dim_geography                               |
| acquisition_channel_key   | bigint        | Foreign key to dim_sales_channel                           |
| customer_age_days         | int           | Days since customer acquisition                            |
| total_orders              | int           | Total number of orders                                     |
| total_items_purchased     | int           | Total items purchased                                      |
| total_gross_sales         | decimal(12,2) | Total gross sales                                          |
| total_net_sales           | decimal(12,2) | Total net sales                                            |
| total_profit              | decimal(12,2) | Total profit generated                                     |
| average_order_value       | decimal(8,2)  | Average order value                                        |
| purchase_frequency_days   | decimal(6,2)  | Average days between purchases                             |
| recency_days              | int           | Days since last purchase                                   |
| monetary_score            | int           | RFM monetary score (1-5)                                   |
| frequency_score           | int           | RFM frequency score (1-5)                                  |
| recency_score             | int           | RFM recency score (1-5)                                    |
| rfm_segment               | string        | RFM segment classification                                 |
| predicted_clv_12m         | decimal(10,2) | Predicted 12-month CLV                                     |
| predicted_clv_24m         | decimal(10,2) | Predicted 24-month CLV                                     |
| churn_probability         | decimal(3,2)  | Churn probability score                                    |
| next_purchase_probability | decimal(3,2)  | Next purchase probability                                  |
| customer_lifetime_stage   | string        | Lifecycle stage (New, Growing, Mature, Declining, Churned) |
| created_date              | timestamp     | Record creation timestamp                                  |

---

## Sample Queries for E-Commerce Analytics

### Sales Performance Dashboard

```sql
SELECT
    dd.year,
    dd.month_name,
    dp.category,
    SUM(fs.quantity_sold) as units_sold,
    SUM(fs.net_sales) as revenue,
    SUM(fs.gross_profit) as profit,
    AVG(fs.unit_price) as avg_unit_price,
    COUNT(DISTINCT fs.customer_key) as unique_customers
FROM fact_sales fs
JOIN dim_date dd ON fs.order_date_key = dd.date_key
JOIN dim_product dp ON fs.product_key = dp.product_key
WHERE dd.year = 2024
GROUP BY dd.year, dd.month_name, dp.category
ORDER BY dd.year, dd.month, revenue DESC;
```

### Customer Segmentation Analysis

```sql
SELECT
    dc.customer_segment,
    dc.lifetime_value_tier,
    COUNT(*) as customer_count,
    AVG(fclv.total_orders) as avg_orders,
    AVG(fclv.average_order_value) as avg_order_value,
    AVG(fclv.total_net_sales) as avg_lifetime_value,
    AVG(fclv.purchase_frequency_days) as avg_purchase_frequency
FROM fact_customer_lifetime_value fclv
JOIN dim_customer dc ON fclv.customer_key = dc.customer_key
WHERE dc.is_current = true
GROUP BY dc.customer_segment, dc.lifetime_value_tier
ORDER BY avg_lifetime_value DESC;
```

### Product Performance & Inventory Analysis

```sql
SELECT
    dp.product_name,
    dp.category,
    dp.brand,
    SUM(fs.quantity_sold) as units_sold,
    SUM(fs.net_sales) as revenue,
    AVG(fi.inventory_turnover) as avg_turnover,
    AVG(fi.days_of_supply) as avg_days_supply,
    SUM(fi.stockout_days) as total_stockout_days
FROM fact_sales fs
JOIN dim_product dp ON fs.product_key = dp.product_key
LEFT JOIN fact_inventory fi ON fs.product_key = fi.product_key
WHERE fs.order_date_key >= 20240101
GROUP BY dp.product_name, dp.category, dp.brand
HAVING units_sold > 0
ORDER BY revenue DESC;
```
