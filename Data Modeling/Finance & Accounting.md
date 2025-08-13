# Finance & Accounting - Data Modeling Template

## Medallion Architecture Overview

- **Bronze Layer**: Raw financial data from various systems (ERP, Banking, etc.)
- **Silver Layer**: Cleaned, validated, and standardized financial data
- **Gold Layer**: Business-ready financial reporting with star schema

---

## Bronze Layer - Raw Financial Data

### general_ledger_raw

| column_name         | type      | description                  |
| ------------------- | --------- | ---------------------------- |
| transaction_id      | string    | Raw transaction identifier   |
| account_code        | string    | Chart of accounts code       |
| transaction_date    | string    | Transaction date from source |
| posting_date        | string    | Posting date from source     |
| description         | string    | Transaction description      |
| debit_amount        | string    | Debit amount as string       |
| credit_amount       | string    | Credit amount as string      |
| reference_number    | string    | Reference document number    |
| journal_entry_id    | string    | Journal entry identifier     |
| cost_center         | string    | Cost center code             |
| department          | string    | Department code              |
| currency_code       | string    | Currency code                |
| exchange_rate       | string    | Exchange rate                |
| created_by          | string    | User who created transaction |
| ingestion_timestamp | timestamp | When record was ingested     |
| source_system       | string    | Source ERP system            |

### accounts_payable_raw

| column_name           | type      | description              |
| --------------------- | --------- | ------------------------ |
| invoice_id            | string    | Raw invoice identifier   |
| vendor_id             | string    | Vendor identifier        |
| vendor_name           | string    | Vendor name              |
| invoice_number        | string    | Invoice number           |
| invoice_date          | string    | Invoice date             |
| due_date              | string    | Payment due date         |
| invoice_amount        | string    | Total invoice amount     |
| paid_amount           | string    | Amount paid              |
| payment_status        | string    | Payment status           |
| purchase_order_number | string    | PO number                |
| approval_status       | string    | Approval status          |
| payment_terms         | string    | Payment terms            |
| currency_code         | string    | Invoice currency         |
| ingestion_timestamp   | timestamp | When record was ingested |
| source_system         | string    | Source system identifier |

### accounts_receivable_raw

| column_name         | type      | description              |
| ------------------- | --------- | ------------------------ |
| invoice_id          | string    | Raw invoice identifier   |
| customer_id         | string    | Customer identifier      |
| customer_name       | string    | Customer name            |
| invoice_number      | string    | Invoice number           |
| invoice_date        | string    | Invoice date             |
| due_date            | string    | Payment due date         |
| invoice_amount      | string    | Total invoice amount     |
| paid_amount         | string    | Amount paid              |
| outstanding_amount  | string    | Outstanding amount       |
| payment_status      | string    | Payment status           |
| payment_terms       | string    | Payment terms            |
| sales_rep           | string    | Sales representative     |
| currency_code       | string    | Invoice currency         |
| ingestion_timestamp | timestamp | When record was ingested |
| source_system       | string    | Source system identifier |

### bank_transactions_raw

| column_name         | type      | description                       |
| ------------------- | --------- | --------------------------------- |
| transaction_id      | string    | Bank transaction ID               |
| account_number      | string    | Bank account number               |
| transaction_date    | string    | Transaction date                  |
| description         | string    | Transaction description           |
| amount              | string    | Transaction amount                |
| transaction_type    | string    | Transaction type                  |
| balance             | string    | Account balance after transaction |
| reference_number    | string    | Bank reference number             |
| counterparty        | string    | Counterparty name                 |
| currency            | string    | Transaction currency              |
| ingestion_timestamp | timestamp | When record was ingested          |
| source_system       | string    | Source bank system                |

---

## Silver Layer - Cleaned Financial Data

### general_ledger_clean

| column_name          | type          | description                     |
| -------------------- | ------------- | ------------------------------- |
| transaction_id       | bigint        | Cleaned transaction identifier  |
| account_id           | bigint        | Standardized account identifier |
| transaction_date     | date          | Validated transaction date      |
| posting_date         | date          | Validated posting date          |
| description          | string        | Standardized description        |
| debit_amount         | decimal(15,2) | Debit amount in base currency   |
| credit_amount        | decimal(15,2) | Credit amount in base currency  |
| net_amount           | decimal(15,2) | Net amount (debit - credit)     |
| reference_number     | string        | Reference document number       |
| journal_entry_id     | bigint        | Journal entry identifier        |
| cost_center_id       | bigint        | Cost center identifier          |
| department_id        | bigint        | Department identifier           |
| currency_code        | string        | Original currency code          |
| exchange_rate        | decimal(10,6) | Exchange rate to base currency  |
| base_currency_amount | decimal(15,2) | Amount in base currency         |
| fiscal_year          | int           | Fiscal year                     |
| fiscal_period        | int           | Fiscal period                   |
| is_reversed          | boolean       | Reversal indicator              |
| created_date         | timestamp     | Record creation timestamp       |
| updated_date         | timestamp     | Last update timestamp           |

### accounts_payable_clean

| column_name          | type          | description                                         |
| -------------------- | ------------- | --------------------------------------------------- |
| invoice_id           | bigint        | Cleaned invoice identifier                          |
| vendor_id            | bigint        | Vendor identifier                                   |
| invoice_number       | string        | Standardized invoice number                         |
| invoice_date         | date          | Invoice date                                        |
| due_date             | date          | Payment due date                                    |
| invoice_amount       | decimal(12,2) | Total invoice amount                                |
| paid_amount          | decimal(12,2) | Amount paid                                         |
| outstanding_amount   | decimal(12,2) | Remaining amount                                    |
| payment_status       | string        | Standardized status (Open, Paid, Overdue, Disputed) |
| purchase_order_id    | bigint        | Purchase order identifier                           |
| approval_status      | string        | Approval status                                     |
| payment_terms_days   | int           | Payment terms in days                               |
| days_outstanding     | int           | Days since due date                                 |
| currency_code        | string        | Invoice currency                                    |
| base_currency_amount | decimal(12,2) | Amount in base currency                             |
| is_overdue           | boolean       | Overdue indicator                                   |
| aging_bucket         | string        | Aging category (Current, 30, 60, 90+ days)          |
| created_date         | timestamp     | Record creation timestamp                           |
| updated_date         | timestamp     | Last update timestamp                               |

### accounts_receivable_clean

| column_name          | type          | description                                            |
| -------------------- | ------------- | ------------------------------------------------------ |
| invoice_id           | bigint        | Cleaned invoice identifier                             |
| customer_id          | bigint        | Customer identifier                                    |
| invoice_number       | string        | Standardized invoice number                            |
| invoice_date         | date          | Invoice date                                           |
| due_date             | date          | Payment due date                                       |
| invoice_amount       | decimal(12,2) | Total invoice amount                                   |
| paid_amount          | decimal(12,2) | Amount paid                                            |
| outstanding_amount   | decimal(12,2) | Outstanding amount                                     |
| payment_status       | string        | Standardized status (Open, Paid, Overdue, Written Off) |
| payment_terms_days   | int           | Payment terms in days                                  |
| days_outstanding     | int           | Days since due date                                    |
| sales_rep_id         | bigint        | Sales representative identifier                        |
| currency_code        | string        | Invoice currency                                       |
| base_currency_amount | decimal(12,2) | Amount in base currency                                |
| is_overdue           | boolean       | Overdue indicator                                      |
| aging_bucket         | string        | Aging category (Current, 30, 60, 90+ days)             |
| credit_risk_score    | decimal(3,2)  | Customer credit risk score                             |
| created_date         | timestamp     | Record creation timestamp                              |
| updated_date         | timestamp     | Last update timestamp                                  |

---

## Gold Layer - Financial Reporting Star Schema

### Dimension Tables

#### dim_account

| column_name               | type      | description                                               |
| ------------------------- | --------- | --------------------------------------------------------- |
| account_key               | bigint    | Surrogate key for account dimension                       |
| account_id                | bigint    | Business key from chart of accounts                       |
| account_code              | string    | Account code                                              |
| account_name              | string    | Account name                                              |
| account_type              | string    | Account type (Asset, Liability, Equity, Revenue, Expense) |
| account_category          | string    | Account category                                          |
| account_subcategory       | string    | Account subcategory                                       |
| parent_account_code       | string    | Parent account code                                       |
| account_level             | int       | Account hierarchy level                                   |
| is_active                 | boolean   | Active account indicator                                  |
| normal_balance            | string    | Normal balance (Debit/Credit)                             |
| is_balance_sheet          | boolean   | Balance sheet account indicator                           |
| is_income_statement       | boolean   | Income statement account indicator                        |
| is_cash_flow              | boolean   | Cash flow statement indicator                             |
| financial_statement_order | int       | Ordering for financial statements                         |
| created_date              | timestamp | Record creation timestamp                                 |

#### dim_vendor

| column_name      | type      | description                                 |
| ---------------- | --------- | ------------------------------------------- |
| vendor_key       | bigint    | Surrogate key for vendor dimension          |
| vendor_id        | bigint    | Business key from source system             |
| vendor_name      | string    | Vendor name                                 |
| vendor_category  | string    | Vendor category                             |
| vendor_type      | string    | Vendor type (Individual, Corporation, etc.) |
| country          | string    | Vendor country                              |
| region           | string    | Vendor region                               |
| payment_terms    | string    | Standard payment terms                      |
| tax_id           | string    | Vendor tax ID                               |
| credit_rating    | string    | Credit rating                               |
| preferred_vendor | boolean   | Preferred vendor indicator                  |
| is_active        | boolean   | Active vendor indicator                     |
| created_date     | timestamp | Record creation timestamp                   |

#### dim_customer

| column_name      | type          | description                              |
| ---------------- | ------------- | ---------------------------------------- |
| customer_key     | bigint        | Surrogate key for customer dimension     |
| customer_id      | bigint        | Business key from source system          |
| customer_name    | string        | Customer name                            |
| customer_type    | string        | Customer type (Individual, Business)     |
| industry         | string        | Customer industry                        |
| country          | string        | Customer country                         |
| region           | string        | Customer region                          |
| customer_segment | string        | Customer segment (SMB, Enterprise, etc.) |
| credit_limit     | decimal(12,2) | Credit limit                             |
| credit_rating    | string        | Credit rating                            |
| payment_terms    | string        | Standard payment terms                   |
| is_active        | boolean       | Active customer indicator                |
| acquisition_date | date          | Customer acquisition date                |
| created_date     | timestamp     | Record creation timestamp                |

#### dim_cost_center

| column_name         | type          | description                             |
| ------------------- | ------------- | --------------------------------------- |
| cost_center_key     | bigint        | Surrogate key for cost center dimension |
| cost_center_id      | bigint        | Business key from source system         |
| cost_center_code    | string        | Cost center code                        |
| cost_center_name    | string        | Cost center name                        |
| department_name     | string        | Department name                         |
| division_name       | string        | Division name                           |
| business_unit       | string        | Business unit                           |
| cost_center_manager | string        | Cost center manager                     |
| budget_amount       | decimal(12,2) | Annual budget amount                    |
| is_active           | boolean       | Active cost center indicator            |
| created_date        | timestamp     | Record creation timestamp               |

#### dim_date

| column_name        | type    | description                     |
| ------------------ | ------- | ------------------------------- |
| date_key           | int     | Surrogate key (YYYYMMDD format) |
| date               | date    | Actual date                     |
| year               | int     | Calendar year                   |
| quarter            | int     | Calendar quarter                |
| month              | int     | Calendar month                  |
| month_name         | string  | Month name                      |
| fiscal_year        | int     | Fiscal year                     |
| fiscal_quarter     | int     | Fiscal quarter                  |
| fiscal_period      | int     | Fiscal period                   |
| is_fiscal_year_end | boolean | Fiscal year end indicator       |
| is_quarter_end     | boolean | Quarter end indicator           |
| is_month_end       | boolean | Month end indicator             |
| is_weekend         | boolean | Weekend indicator               |
| is_holiday         | boolean | Holiday indicator               |

### Fact Tables

#### fact_general_ledger

| column_name          | type          | description                    |
| -------------------- | ------------- | ------------------------------ |
| gl_fact_key          | bigint        | Surrogate key for fact table   |
| account_key          | bigint        | Foreign key to dim_account     |
| cost_center_key      | bigint        | Foreign key to dim_cost_center |
| transaction_date_key | int           | Foreign key to dim_date        |
| posting_date_key     | int           | Foreign key to dim_date        |
| transaction_id       | bigint        | Business key from source       |
| journal_entry_id     | bigint        | Journal entry identifier       |
| debit_amount         | decimal(15,2) | Debit amount                   |
| credit_amount        | decimal(15,2) | Credit amount                  |
| net_amount           | decimal(15,2) | Net amount                     |
| base_currency_amount | decimal(15,2) | Amount in base currency        |
| exchange_rate        | decimal(10,6) | Exchange rate                  |
| currency_code        | string        | Transaction currency           |
| fiscal_year          | int           | Fiscal year                    |
| fiscal_period        | int           | Fiscal period                  |
| transaction_type     | string        | Transaction type               |
| reference_number     | string        | Reference number               |
| description          | string        | Transaction description        |
| is_adjustment        | boolean       | Adjustment entry indicator     |
| is_reversal          | boolean       | Reversal entry indicator       |
| created_date         | timestamp     | Record creation timestamp      |

#### fact_accounts_payable

| column_name           | type          | description                  |
| --------------------- | ------------- | ---------------------------- |
| ap_fact_key           | bigint        | Surrogate key for fact table |
| vendor_key            | bigint        | Foreign key to dim_vendor    |
| invoice_date_key      | int           | Foreign key to dim_date      |
| due_date_key          | int           | Foreign key to dim_date      |
| invoice_id            | bigint        | Business key from source     |
| invoice_number        | string        | Invoice number               |
| invoice_amount        | decimal(12,2) | Original invoice amount      |
| paid_amount           | decimal(12,2) | Amount paid                  |
| outstanding_amount    | decimal(12,2) | Outstanding amount           |
| discount_amount       | decimal(12,2) | Discount amount              |
| tax_amount            | decimal(12,2) | Tax amount                   |
| base_currency_amount  | decimal(12,2) | Amount in base currency      |
| days_outstanding      | int           | Days outstanding             |
| payment_terms_days    | int           | Payment terms in days        |
| aging_bucket          | string        | Aging category               |
| payment_status        | string        | Payment status               |
| is_overdue            | boolean       | Overdue indicator            |
| is_disputed           | boolean       | Disputed indicator           |
| purchase_order_number | string        | Purchase order number        |
| created_date          | timestamp     | Record creation timestamp    |

#### fact_accounts_receivable

| column_name            | type          | description                  |
| ---------------------- | ------------- | ---------------------------- |
| ar_fact_key            | bigint        | Surrogate key for fact table |
| customer_key           | bigint        | Foreign key to dim_customer  |
| invoice_date_key       | int           | Foreign key to dim_date      |
| due_date_key           | int           | Foreign key to dim_date      |
| invoice_id             | bigint        | Business key from source     |
| invoice_number         | string        | Invoice number               |
| invoice_amount         | decimal(12,2) | Original invoice amount      |
| paid_amount            | decimal(12,2) | Amount paid                  |
| outstanding_amount     | decimal(12,2) | Outstanding amount           |
| discount_amount        | decimal(12,2) | Discount amount              |
| tax_amount             | decimal(12,2) | Tax amount                   |
| base_currency_amount   | decimal(12,2) | Amount in base currency      |
| days_outstanding       | int           | Days outstanding             |
| payment_terms_days     | int           | Payment terms in days        |
| aging_bucket           | string        | Aging category               |
| payment_status         | string        | Payment status               |
| is_overdue             | boolean       | Overdue indicator            |
| credit_risk_score      | decimal(3,2)  | Credit risk score            |
| collection_probability | decimal(3,2)  | Collection probability       |
| sales_rep_id           | bigint        | Sales representative ID      |
| created_date           | timestamp     | Record creation timestamp    |

#### fact_cash_flow

| column_name          | type          | description                                          |
| -------------------- | ------------- | ---------------------------------------------------- |
| cash_flow_key        | bigint        | Surrogate key for fact table                         |
| account_key          | bigint        | Foreign key to dim_account                           |
| date_key             | int           | Foreign key to dim_date                              |
| transaction_id       | bigint        | Business key from source                             |
| cash_flow_category   | string        | Cash flow category (Operating, Investing, Financing) |
| cash_inflow          | decimal(15,2) | Cash inflow amount                                   |
| cash_outflow         | decimal(15,2) | Cash outflow amount                                  |
| net_cash_flow        | decimal(15,2) | Net cash flow                                        |
| running_balance      | decimal(15,2) | Running cash balance                                 |
| currency_code        | string        | Transaction currency                                 |
| base_currency_amount | decimal(15,2) | Amount in base currency                              |
| counterparty         | string        | Transaction counterparty                             |
| transaction_type     | string        | Transaction type                                     |
| reference_number     | string        | Reference number                                     |
| description          | string        | Transaction description                              |
| bank_account         | string        | Bank account identifier                              |
| created_date         | timestamp     | Record creation timestamp                            |
