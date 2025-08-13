# Supply Chain Data Modeling Template - Medallion Architecture

## Bronze Layer (Raw Data Ingestion)

### bronze_suppliers_raw

| column_name         | type      | description                            |
| ------------------- | --------- | -------------------------------------- |
| supplier_id         | string    | Supplier identifier from source system |
| supplier_name       | string    | Supplier name (raw format)             |
| contact_person      | string    | Primary contact person                 |
| email               | string    | Supplier email (raw format)            |
| phone               | string    | Supplier phone (various formats)       |
| address             | string    | Supplier address (raw)                 |
| country             | string    | Supplier country                       |
| supplier_type       | string    | Type of supplier                       |
| registration_number | string    | Business registration number           |
| tax_id              | string    | Tax identification number              |
| payment_terms       | string    | Payment terms                          |
| credit_rating       | string    | Credit rating                          |
| certification       | string    | Certifications held                    |
| source_system       | string    | Source system identifier               |
| ingestion_timestamp | timestamp | Timestamp when data was ingested       |

### bronze_purchase_orders_raw

| column_name         | type      | description                           |
| ------------------- | --------- | ------------------------------------- |
| po_number           | string    | Purchase order number                 |
| supplier_id         | string    | Supplier identifier                   |
| requester_id        | string    | Employee who requested                |
| approver_id         | string    | Employee who approved                 |
| po_date             | string    | Purchase order date (various formats) |
| required_date       | string    | Required delivery date                |
| po_status           | string    | Purchase order status                 |
| total_amount        | string    | Total PO amount (various formats)     |
| currency            | string    | Currency code                         |
| payment_terms       | string    | Payment terms                         |
| delivery_address    | string    | Delivery address                      |
| notes               | string    | Additional notes                      |
| source_system       | string    | Source system identifier              |
| ingestion_timestamp | timestamp | Timestamp when data was ingested      |

### bronze_po_line_items_raw

| column_name         | type      | description                      |
| ------------------- | --------- | -------------------------------- |
| po_line_id          | string    | Purchase order line identifier   |
| po_number           | string    | Purchase order number            |
| item_code           | string    | Item/product code                |
| item_description    | string    | Item description                 |
| quantity_ordered    | string    | Quantity ordered                 |
| unit_price          | string    | Unit price                       |
| line_total          | string    | Line total amount                |
| uom                 | string    | Unit of measure                  |
| delivery_date       | string    | Expected delivery date           |
| source_system       | string    | Source system identifier         |
| ingestion_timestamp | timestamp | Timestamp when data was ingested |

### bronze_receipts_raw

| column_name         | type      | description                      |
| ------------------- | --------- | -------------------------------- |
| receipt_id          | string    | Receipt identifier               |
| po_number           | string    | Purchase order number            |
| po_line_id          | string    | Purchase order line identifier   |
| supplier_id         | string    | Supplier identifier              |
| receipt_date        | string    | Receipt date                     |
| quantity_received   | string    | Quantity received                |
| quantity_accepted   | string    | Quantity accepted                |
| quantity_rejected   | string    | Quantity rejected                |
| received_by         | string    | Employee who received            |
| condition           | string    | Condition of goods               |
| warehouse_location  | string    | Storage location                 |
| batch_number        | string    | Batch/lot number                 |
| expiry_date         | string    | Expiry date (if applicable)      |
| source_system       | string    | Source system identifier         |
| ingestion_timestamp | timestamp | Timestamp when data was ingested |

### bronze_inventory_raw

| column_name         | type      | description                      |
| ------------------- | --------- | -------------------------------- |
| item_code           | string    | Item/product code                |
| warehouse_id        | string    | Warehouse identifier             |
| location            | string    | Specific location in warehouse   |
| quantity_on_hand    | string    | Current quantity                 |
| reserved_quantity   | string    | Reserved quantity                |
| available_quantity  | string    | Available quantity               |
| unit_cost           | string    | Unit cost                        |
| last_updated        | string    | Last update timestamp            |
| batch_number        | string    | Batch/lot number                 |
| expiry_date         | string    | Expiry date                      |
| source_system       | string    | Source system identifier         |
| ingestion_timestamp | timestamp | Timestamp when data was ingested |

## Silver Layer (Cleaned and Transformed Data)

### silver_suppliers

| column_name               | type      | description                  |
| ------------------------- | --------- | ---------------------------- |
| supplier_key              | bigint    | Surrogate key for supplier   |
| supplier_id               | string    | Cleaned supplier identifier  |
| supplier_name_clean       | string    | Cleaned supplier name        |
| contact_person_clean      | string    | Cleaned contact person name  |
| email_clean               | string    | Validated email address      |
| phone_clean               | string    | Standardized phone number    |
| address_clean             | string    | Cleaned address              |
| city                      | string    | Extracted city               |
| state                     | string    | Extracted state/province     |
| country                   | string    | Standardized country         |
| postal_code               | string    | Extracted postal code        |
| supplier_type_clean       | string    | Standardized supplier type   |
| registration_number_clean | string    | Cleaned registration number  |
| tax_id_clean              | string    | Cleaned tax ID               |
| payment_terms_days        | int       | Payment terms in days        |
| credit_rating_clean       | string    | Standardized credit rating   |
| certification_clean       | string    | Cleaned certifications       |
| is_active                 | boolean   | Supplier active status       |
| created_timestamp         | timestamp | Record creation timestamp    |
| updated_timestamp         | timestamp | Record last update timestamp |

### silver_purchase_orders

| column_name            | type          | description                       |
| ---------------------- | ------------- | --------------------------------- |
| po_key                 | bigint        | Surrogate key for purchase order  |
| po_number              | string        | Purchase order number             |
| supplier_id            | string        | Supplier identifier               |
| requester_id           | string        | Requester employee ID             |
| approver_id            | string        | Approver employee ID              |
| po_date                | date          | Purchase order date               |
| required_date          | date          | Required delivery date            |
| po_status_clean        | string        | Standardized PO status            |
| total_amount           | decimal(15,2) | Total PO amount                   |
| currency_code          | string        | Currency code                     |
| payment_terms_days     | int           | Payment terms in days             |
| delivery_address_clean | string        | Cleaned delivery address          |
| priority_level         | string        | Priority level                    |
| notes_clean            | string        | Cleaned notes                     |
| days_to_deliver        | int           | Days between PO and required date |
| created_timestamp      | timestamp     | Record creation timestamp         |
| updated_timestamp      | timestamp     | Record last update timestamp      |

### silver_po_line_items

| column_name            | type          | description                  |
| ---------------------- | ------------- | ---------------------------- |
| po_line_key            | bigint        | Surrogate key for PO line    |
| po_line_id             | string        | PO line identifier           |
| po_number              | string        | Purchase order number        |
| item_code              | string        | Item/product code            |
| item_description_clean | string        | Cleaned item description     |
| quantity_ordered       | int           | Quantity ordered             |
| unit_price             | decimal(12,4) | Unit price                   |
| line_total             | decimal(15,2) | Line total amount            |
| uom_clean              | string        | Standardized unit of measure |
| delivery_date          | date          | Expected delivery date       |
| item_category          | string        | Item category                |
| created_timestamp      | timestamp     | Record creation timestamp    |

### silver_receipts

| column_name        | type         | description                      |
| ------------------ | ------------ | -------------------------------- |
| receipt_key        | bigint       | Surrogate key for receipt        |
| receipt_id         | string       | Receipt identifier               |
| po_number          | string       | Purchase order number            |
| po_line_id         | string       | Purchase order line identifier   |
| supplier_id        | string       | Supplier identifier              |
| receipt_date       | date         | Receipt date                     |
| quantity_received  | int          | Quantity received                |
| quantity_accepted  | int          | Quantity accepted                |
| quantity_rejected  | int          | Quantity rejected                |
| received_by        | string       | Employee who received            |
| condition_clean    | string       | Standardized condition           |
| warehouse_location | string       | Storage location                 |
| batch_number       | string       | Batch/lot number                 |
| expiry_date        | date         | Expiry date                      |
| days_late          | int          | Days late from expected delivery |
| quality_score      | decimal(3,2) | Quality score (0-1)              |
| created_timestamp  | timestamp    | Record creation timestamp        |

### silver_inventory

| column_name        | type          | description                  |
| ------------------ | ------------- | ---------------------------- |
| inventory_key      | bigint        | Surrogate key for inventory  |
| item_code          | string        | Item/product code            |
| warehouse_id       | string        | Warehouse identifier         |
| location_clean     | string        | Cleaned location             |
| quantity_on_hand   | int           | Current quantity             |
| reserved_quantity  | int           | Reserved quantity            |
| available_quantity | int           | Available quantity           |
| unit_cost          | decimal(12,4) | Unit cost                    |
| total_value        | decimal(15,2) | Total inventory value        |
| last_movement_date | date          | Last movement date           |
| batch_number       | string        | Batch/lot number             |
| expiry_date        | date          | Expiry date                  |
| days_to_expiry     | int           | Days until expiry            |
| abc_classification | string        | ABC classification           |
| safety_stock_level | int           | Safety stock level           |
| reorder_point      | int           | Reorder point                |
| created_timestamp  | timestamp     | Record creation timestamp    |
| updated_timestamp  | timestamp     | Record last update timestamp |

## Gold Layer (Facts and Dimensions)

### dim_supplier

| column_name         | type    | description               |
| ------------------- | ------- | ------------------------- |
| supplier_key        | bigint  | Primary key (surrogate)   |
| supplier_id         | string  | Business key              |
| supplier_name       | string  | Supplier name             |
| contact_person      | string  | Primary contact           |
| email               | string  | Supplier email            |
| phone               | string  | Supplier phone            |
| address             | string  | Full address              |
| city                | string  | Supplier city             |
| state               | string  | Supplier state            |
| country             | string  | Supplier country          |
| postal_code         | string  | Postal code               |
| supplier_type       | string  | Supplier type             |
| supplier_category   | string  | Supplier category         |
| payment_terms_days  | int     | Payment terms in days     |
| credit_rating       | string  | Credit rating             |
| risk_level          | string  | Risk assessment level     |
| certification_level | string  | Certification level       |
| is_strategic        | boolean | Strategic supplier flag   |
| is_active           | boolean | Active status             |
| effective_date      | date    | SCD Type 2 effective date |
| expiry_date         | date    | SCD Type 2 expiry date    |
| is_current          | boolean | Current record indicator  |
| version             | int     | Record version number     |

### dim_item

| column_name               | type    | description                               |
| ------------------------- | ------- | ----------------------------------------- |
| item_key                  | bigint  | Primary key (surrogate)                   |
| item_code                 | string  | Business key                              |
| item_description          | string  | Item description                          |
| item_category             | string  | Item category                             |
| item_subcategory          | string  | Item subcategory                          |
| item_type                 | string  | Item type (raw material, component, etc.) |
| unit_of_measure           | string  | Primary unit of measure                   |
| abc_classification        | string  | ABC classification                        |
| criticality_level         | string  | Criticality level                         |
| lead_time_days            | int     | Standard lead time in days                |
| shelf_life_days           | int     | Shelf life in days                        |
| is_hazardous              | boolean | Hazardous material flag                   |
| is_temperature_controlled | boolean | Temperature control required              |
| supplier_count            | int     | Number of approved suppliers              |
| is_active                 | boolean | Active status                             |
| effective_date            | date    | SCD Type 2 effective date                 |
| expiry_date               | date    | SCD Type 2 expiry date                    |
| is_current                | boolean | Current record indicator                  |
| version                   | int     | Record version number                     |

### dim_warehouse

| column_name            | type    | description                   |
| ---------------------- | ------- | ----------------------------- |
| warehouse_key          | bigint  | Primary key (surrogate)       |
| warehouse_id           | string  | Business key                  |
| warehouse_name         | string  | Warehouse name                |
| warehouse_type         | string  | Warehouse type                |
| address                | string  | Warehouse address             |
| city                   | string  | Warehouse city                |
| state                  | string  | Warehouse state               |
| country                | string  | Warehouse country             |
| postal_code            | string  | Postal code                   |
| region                 | string  | Geographic region             |
| manager_id             | string  | Warehouse manager             |
| capacity_sqft          | int     | Capacity in square feet       |
| temperature_controlled | boolean | Temperature control available |
| is_automated           | boolean | Automation level              |
| operating_hours        | string  | Operating hours               |
| is_active              | boolean | Active status                 |
| effective_date         | date    | SCD Type 2 effective date     |
| expiry_date            | date    | SCD Type 2 expiry date        |
| is_current             | boolean | Current record indicator      |
| version                | int     | Record version number         |

### dim_date

| column_name     | type    | description            |
| --------------- | ------- | ---------------------- |
| date_key        | int     | Primary key (YYYYMMDD) |
| full_date       | date    | Full date              |
| day_of_week     | int     | Day of week (1-7)      |
| day_name        | string  | Day name               |
| day_of_month    | int     | Day of month           |
| day_of_year     | int     | Day of year            |
| week_of_year    | int     | Week of year           |
| month           | int     | Month number           |
| month_name      | string  | Month name             |
| quarter         | int     | Quarter                |
| year            | int     | Year                   |
| is_weekend      | boolean | Weekend indicator      |
| is_holiday      | boolean | Holiday indicator      |
| is_business_day | boolean | Business day indicator |
| fiscal_year     | int     | Fiscal year            |
| fiscal_quarter  | int     | Fiscal quarter         |
| fiscal_month    | int     | Fiscal month           |

### fact_purchase_orders

| column_name         | type          | description                 |
| ------------------- | ------------- | --------------------------- |
| po_fact_key         | bigint        | Primary key (surrogate)     |
| po_number           | string        | Business key                |
| supplier_key        | bigint        | Foreign key to dim_supplier |
| po_date_key         | int           | Foreign key to dim_date     |
| required_date_key   | int           | Foreign key to dim_date     |
| po_status           | string        | Purchase order status       |
| line_count          | int           | Number of line items        |
| total_amount        | decimal(15,2) | Total PO amount             |
| currency_code       | string        | Currency code               |
| payment_terms_days  | int           | Payment terms in days       |
| priority_level      | string        | Priority level              |
| lead_time_days      | int           | Lead time in days           |
| approval_cycle_days | int           | Days to approve             |
| created_timestamp   | timestamp     | Record creation timestamp   |

### fact_po_line_items

| column_name          | type          | description                     |
| -------------------- | ------------- | ------------------------------- |
| po_line_fact_key     | bigint        | Primary key (surrogate)         |
| po_line_id           | string        | Business key                    |
| po_number            | string        | Purchase order number           |
| supplier_key         | bigint        | Foreign key to dim_supplier     |
| item_key             | bigint        | Foreign key to dim_item         |
| po_date_key          | int           | Foreign key to dim_date         |
| delivery_date_key    | int           | Foreign key to dim_date         |
| quantity_ordered     | int           | Quantity ordered                |
| unit_price           | decimal(12,4) | Unit price                      |
| line_total           | decimal(15,2) | Line total amount               |
| quantity_received    | int           | Quantity received               |
| quantity_accepted    | int           | Quantity accepted               |
| quantity_rejected    | int           | Quantity rejected               |
| quantity_pending     | int           | Quantity pending receipt        |
| received_amount      | decimal(15,2) | Amount received                 |
| pending_amount       | decimal(15,2) | Amount pending                  |
| days_to_deliver      | int           | Days between order and delivery |
| delivery_performance | string        | On-time delivery performance    |
| quality_rating       | decimal(3,2)  | Quality rating (0-1)            |
| created_timestamp    | timestamp     | Record creation timestamp       |

### fact_inventory_snapshot

| column_name        | type          | description                  |
| ------------------ | ------------- | ---------------------------- |
| inventory_fact_key | bigint        | Primary key (surrogate)      |
| item_key           | bigint        | Foreign key to dim_item      |
| warehouse_key      | bigint        | Foreign key to dim_warehouse |
| date_key           | int           | Foreign key to dim_date      |
| snapshot_date      | date          | Snapshot date                |
| beginning_balance  | int           | Beginning quantity           |
| receipts           | int           | Receipts quantity            |
| issues             | int           | Issues quantity              |
| adjustments        | int           | Adjustments quantity         |
| ending_balance     | int           | Ending quantity              |
| reserved_quantity  | int           | Reserved quantity            |
| available_quantity | int           | Available quantity           |
| unit_cost          | decimal(12,4) | Unit cost                    |
| total_value        | decimal(15,2) | Total inventory value        |
| days_of_supply     | decimal(8,2)  | Days of supply               |
| stock_out_risk     | string        | Stock out risk level         |
| excess_stock_flag  | boolean       | Excess stock indicator       |
| slow_moving_flag   | boolean       | Slow moving indicator        |
| created_timestamp  | timestamp     | Record creation timestamp    |

### fact_supplier_performance

| column_name             | type          | description                 |
| ----------------------- | ------------- | --------------------------- |
| supplier_perf_key       | bigint        | Primary key (surrogate)     |
| supplier_key            | bigint        | Foreign key to dim_supplier |
| date_key                | int           | Foreign key to dim_date     |
| evaluation_period       | string        | Evaluation period           |
| total_pos               | int           | Total purchase orders       |
| total_po_value          | decimal(15,2) | Total PO value              |
| on_time_deliveries      | int           | On-time deliveries count    |
| late_deliveries         | int           | Late deliveries count       |
| average_lead_time       | decimal(8,2)  | Average lead time days      |
| quality_rejections      | int           | Quality rejections count    |
| total_receipts          | int           | Total receipts count        |
| on_time_delivery_rate   | decimal(5,4)  | On-time delivery rate       |
| quality_acceptance_rate | decimal(5,4)  | Quality acceptance rate     |
| cost_performance_index  | decimal(8,4)  | Cost performance index      |
| overall_rating          | decimal(3,2)  | Overall supplier rating     |
| created_timestamp       | timestamp     | Record creation timestamp   |
