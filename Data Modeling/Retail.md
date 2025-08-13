# Retail Data Modeling Template - Medallion Architecture

## Bronze Layer (Raw Data Ingestion)

### bronze_customers_raw

| column_name         | type      | description                                   |
| ------------------- | --------- | --------------------------------------------- |
| customer_id         | string    | Unique customer identifier from source system |
| first_name          | string    | Customer first name (raw, unvalidated)        |
| last_name           | string    | Customer last name (raw, unvalidated)         |
| email               | string    | Customer email address (raw format)           |
| phone               | string    | Customer phone number (various formats)       |
| address             | string    | Full address string from source               |
| registration_date   | string    | Date customer registered (various formats)    |
| customer_type       | string    | Customer type/segment from source             |
| source_system       | string    | Source system identifier                      |
| ingestion_timestamp | timestamp | Timestamp when data was ingested              |
| file_name           | string    | Source file name                              |

### bronze_products_raw

| column_name         | type      | description                                   |
| ------------------- | --------- | --------------------------------------------- |
| product_id          | string    | Product identifier from source                |
| product_name        | string    | Product name (raw, may contain special chars) |
| category            | string    | Product category (raw format)                 |
| subcategory         | string    | Product subcategory                           |
| brand               | string    | Product brand name                            |
| price               | string    | Product price (various formats)               |
| cost                | string    | Product cost (various formats)                |
| sku                 | string    | Stock Keeping Unit                            |
| description         | string    | Product description                           |
| weight              | string    | Product weight                                |
| dimensions          | string    | Product dimensions                            |
| supplier_id         | string    | Supplier identifier                           |
| created_date        | string    | Product creation date                         |
| source_system       | string    | Source system identifier                      |
| ingestion_timestamp | timestamp | Timestamp when data was ingested              |

### bronze_transactions_raw

| column_name         | type      | description                        |
| ------------------- | --------- | ---------------------------------- |
| transaction_id      | string    | Transaction identifier             |
| customer_id         | string    | Customer identifier                |
| product_id          | string    | Product identifier                 |
| store_id            | string    | Store identifier                   |
| transaction_date    | string    | Transaction date (various formats) |
| quantity            | string    | Quantity purchased                 |
| unit_price          | string    | Unit price at time of purchase     |
| discount            | string    | Discount applied                   |
| payment_method      | string    | Payment method used                |
| sales_channel       | string    | Channel (online, in-store, mobile) |
| cashier_id          | string    | Cashier identifier                 |
| source_system       | string    | Source system identifier           |
| ingestion_timestamp | timestamp | Timestamp when data was ingested   |

### bronze_stores_raw

| column_name         | type      | description                         |
| ------------------- | --------- | ----------------------------------- |
| store_id            | string    | Store identifier                    |
| store_name          | string    | Store name                          |
| address             | string    | Store address                       |
| city                | string    | Store city                          |
| state               | string    | Store state/province                |
| country             | string    | Store country                       |
| postal_code         | string    | Store postal code                   |
| manager_id          | string    | Store manager identifier            |
| store_type          | string    | Store type (flagship, outlet, etc.) |
| opening_date        | string    | Store opening date                  |
| square_footage      | string    | Store size in square feet           |
| source_system       | string    | Source system identifier            |
| ingestion_timestamp | timestamp | Timestamp when data was ingested    |

## Silver Layer (Cleaned and Transformed Data)

### silver_customers

| column_name       | type      | description                        |
| ----------------- | --------- | ---------------------------------- |
| customer_key      | bigint    | Surrogate key for customer         |
| customer_id       | string    | Cleaned customer identifier        |
| full_name         | string    | Concatenated and cleaned full name |
| email_clean       | string    | Validated and cleaned email        |
| phone_clean       | string    | Standardized phone number          |
| address_clean     | string    | Cleaned and standardized address   |
| city              | string    | Extracted city                     |
| state             | string    | Extracted state/province           |
| country           | string    | Extracted country                  |
| postal_code       | string    | Extracted postal code              |
| registration_date | date      | Parsed registration date           |
| customer_type     | string    | Standardized customer type         |
| is_active         | boolean   | Customer active status             |
| created_timestamp | timestamp | Record creation timestamp          |
| updated_timestamp | timestamp | Record last update timestamp       |

### silver_products

| column_name              | type          | description                  |
| ------------------------ | ------------- | ---------------------------- |
| product_key              | bigint        | Surrogate key for product    |
| product_id               | string        | Cleaned product identifier   |
| product_name_clean       | string        | Cleaned product name         |
| category_standardized    | string        | Standardized category        |
| subcategory_standardized | string        | Standardized subcategory     |
| brand_clean              | string        | Cleaned brand name           |
| current_price            | decimal(10,2) | Current product price        |
| cost                     | decimal(10,2) | Product cost                 |
| sku_clean                | string        | Cleaned SKU                  |
| description_clean        | string        | Cleaned description          |
| weight_kg                | decimal(8,3)  | Weight in kilograms          |
| length_cm                | decimal(8,2)  | Length in centimeters        |
| width_cm                 | decimal(8,2)  | Width in centimeters         |
| height_cm                | decimal(8,2)  | Height in centimeters        |
| supplier_id              | string        | Supplier identifier          |
| is_active                | boolean       | Product active status        |
| created_timestamp        | timestamp     | Record creation timestamp    |
| updated_timestamp        | timestamp     | Record last update timestamp |

### silver_transactions

| column_name          | type          | description                    |
| -------------------- | ------------- | ------------------------------ |
| transaction_key      | bigint        | Surrogate key for transaction  |
| transaction_id       | string        | Cleaned transaction identifier |
| customer_id          | string        | Customer identifier            |
| product_id           | string        | Product identifier             |
| store_id             | string        | Store identifier               |
| transaction_date     | date          | Transaction date               |
| transaction_time     | time          | Transaction time               |
| quantity             | int           | Quantity purchased             |
| unit_price           | decimal(10,2) | Unit price                     |
| gross_amount         | decimal(12,2) | Gross transaction amount       |
| discount_amount      | decimal(10,2) | Discount amount                |
| net_amount           | decimal(12,2) | Net transaction amount         |
| payment_method_clean | string        | Standardized payment method    |
| sales_channel_clean  | string        | Standardized sales channel     |
| cashier_id           | string        | Cashier identifier             |
| created_timestamp    | timestamp     | Record creation timestamp      |

### silver_stores

| column_name       | type         | description                  |
| ----------------- | ------------ | ---------------------------- |
| store_key         | bigint       | Surrogate key for store      |
| store_id          | string       | Store identifier             |
| store_name_clean  | string       | Cleaned store name           |
| address_clean     | string       | Cleaned address              |
| city              | string       | Store city                   |
| state             | string       | Store state/province         |
| country           | string       | Store country                |
| postal_code       | string       | Store postal code            |
| latitude          | decimal(9,6) | Store latitude               |
| longitude         | decimal(9,6) | Store longitude              |
| manager_id        | string       | Store manager identifier     |
| store_type_clean  | string       | Standardized store type      |
| opening_date      | date         | Store opening date           |
| square_footage    | int          | Store size in square feet    |
| is_active         | boolean      | Store active status          |
| created_timestamp | timestamp    | Record creation timestamp    |
| updated_timestamp | timestamp    | Record last update timestamp |

## Gold Layer (Facts and Dimensions)

### dim_customer

| column_name        | type    | description                |
| ------------------ | ------- | -------------------------- |
| customer_key       | bigint  | Primary key (surrogate)    |
| customer_id        | string  | Business key               |
| full_name          | string  | Customer full name         |
| email              | string  | Customer email             |
| phone              | string  | Customer phone             |
| city               | string  | Customer city              |
| state              | string  | Customer state             |
| country            | string  | Customer country           |
| postal_code        | string  | Customer postal code       |
| registration_date  | date    | Customer registration date |
| customer_type      | string  | Customer type/segment      |
| customer_age_group | string  | Age group category         |
| is_active          | boolean | Customer active status     |
| effective_date     | date    | SCD Type 2 effective date  |
| expiry_date        | date    | SCD Type 2 expiry date     |
| is_current         | boolean | Current record indicator   |
| version            | int     | Record version number      |

### dim_product

| column_name    | type         | description                    |
| -------------- | ------------ | ------------------------------ |
| product_key    | bigint       | Primary key (surrogate)        |
| product_id     | string       | Business key                   |
| product_name   | string       | Product name                   |
| category       | string       | Product category               |
| subcategory    | string       | Product subcategory            |
| brand          | string       | Product brand                  |
| sku            | string       | Stock Keeping Unit             |
| description    | string       | Product description            |
| weight_kg      | decimal(8,3) | Product weight                 |
| dimensions     | string       | Product dimensions             |
| price_tier     | string       | Price tier (low, medium, high) |
| supplier_id    | string       | Supplier identifier            |
| is_active      | boolean      | Product active status          |
| effective_date | date         | SCD Type 2 effective date      |
| expiry_date    | date         | SCD Type 2 expiry date         |
| is_current     | boolean      | Current record indicator       |
| version        | int          | Record version number          |

### dim_store

| column_name         | type    | description               |
| ------------------- | ------- | ------------------------- |
| store_key           | bigint  | Primary key (surrogate)   |
| store_id            | string  | Business key              |
| store_name          | string  | Store name                |
| address             | string  | Store address             |
| city                | string  | Store city                |
| state               | string  | Store state               |
| country             | string  | Store country             |
| postal_code         | string  | Store postal code         |
| region              | string  | Geographic region         |
| store_type          | string  | Store type                |
| store_size_category | string  | Store size category       |
| manager_id          | string  | Store manager             |
| opening_date        | date    | Store opening date        |
| is_active           | boolean | Store active status       |
| effective_date      | date    | SCD Type 2 effective date |
| expiry_date         | date    | SCD Type 2 expiry date    |
| is_current          | boolean | Current record indicator  |
| version             | int     | Record version number     |

### dim_date

| column_name    | type    | description                      |
| -------------- | ------- | -------------------------------- |
| date_key       | int     | Primary key (YYYYMMDD)           |
| full_date      | date    | Full date                        |
| day_of_week    | int     | Day of week (1-7)                |
| day_name       | string  | Day name (Monday, Tuesday, etc.) |
| day_of_month   | int     | Day of month (1-31)              |
| day_of_year    | int     | Day of year (1-366)              |
| week_of_year   | int     | Week of year (1-53)              |
| month          | int     | Month number (1-12)              |
| month_name     | string  | Month name                       |
| quarter        | int     | Quarter (1-4)                    |
| year           | int     | Year                             |
| is_weekend     | boolean | Weekend indicator                |
| is_holiday     | boolean | Holiday indicator                |
| fiscal_year    | int     | Fiscal year                      |
| fiscal_quarter | int     | Fiscal quarter                   |
| fiscal_month   | int     | Fiscal month                     |

### fact_sales

| column_name        | type          | description                 |
| ------------------ | ------------- | --------------------------- |
| sales_key          | bigint        | Primary key (surrogate)     |
| transaction_id     | string        | Business key                |
| customer_key       | bigint        | Foreign key to dim_customer |
| product_key        | bigint        | Foreign key to dim_product  |
| store_key          | bigint        | Foreign key to dim_store    |
| date_key           | int           | Foreign key to dim_date     |
| transaction_date   | date          | Transaction date            |
| transaction_time   | time          | Transaction time            |
| quantity           | int           | Quantity sold               |
| unit_price         | decimal(10,2) | Unit price at sale          |
| unit_cost          | decimal(10,2) | Unit cost                   |
| gross_sales_amount | decimal(12,2) | Gross sales amount          |
| discount_amount    | decimal(10,2) | Discount applied            |
| net_sales_amount   | decimal(12,2) | Net sales amount            |
| cost_amount        | decimal(12,2) | Total cost amount           |
| profit_amount      | decimal(12,2) | Profit amount               |
| payment_method     | string        | Payment method              |
| sales_channel      | string        | Sales channel               |
| cashier_id         | string        | Cashier identifier          |
| created_timestamp  | timestamp     | Record creation timestamp   |

### fact_inventory

| column_name         | type          | description                       |
| ------------------- | ------------- | --------------------------------- |
| inventory_key       | bigint        | Primary key (surrogate)           |
| product_key         | bigint        | Foreign key to dim_product        |
| store_key           | bigint        | Foreign key to dim_store          |
| date_key            | int           | Foreign key to dim_date           |
| snapshot_date       | date          | Inventory snapshot date           |
| beginning_inventory | int           | Beginning inventory count         |
| receipts            | int           | Inventory receipts                |
| sales               | int           | Inventory sold                    |
| adjustments         | int           | Inventory adjustments             |
| ending_inventory    | int           | Ending inventory count            |
| inventory_value     | decimal(12,2) | Total inventory value             |
| days_of_supply      | decimal(8,2)  | Days of supply                    |
| stock_status        | string        | Stock status (in-stock, low, out) |
| created_timestamp   | timestamp     | Record creation timestamp         |
