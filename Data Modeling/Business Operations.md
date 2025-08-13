# Business Operations - Data Modeling Template

## Medallion Architecture Overview

- **Bronze Layer**: Raw ingested data with minimal transformation
- **Silver Layer**: Cleaned, validated, and enriched data
- **Gold Layer**: Business-ready aggregated data with star schema (Facts & Dimensions)

---

## Bronze Layer - Raw Data Ingestion

### employee_raw

| column_name         | type      | description                                |
| ------------------- | --------- | ------------------------------------------ |
| employee_id         | string    | Raw employee identifier from source system |
| first_name          | string    | Employee first name                        |
| last_name           | string    | Employee last name                         |
| email               | string    | Employee email address                     |
| department_code     | string    | Department code from source                |
| hire_date           | string    | Hire date in various formats               |
| salary              | string    | Salary as string from source               |
| manager_id          | string    | Manager identifier                         |
| status              | string    | Employment status                          |
| ingestion_timestamp | timestamp | When record was ingested                   |
| source_system       | string    | Source system identifier                   |

### project_raw

| column_name         | type      | description              |
| ------------------- | --------- | ------------------------ |
| project_id          | string    | Raw project identifier   |
| project_name        | string    | Project name             |
| description         | string    | Project description      |
| start_date          | string    | Project start date       |
| end_date            | string    | Project end date         |
| budget              | string    | Project budget as string |
| status              | string    | Project status           |
| owner_id            | string    | Project owner identifier |
| priority            | string    | Project priority level   |
| ingestion_timestamp | timestamp | When record was ingested |
| source_system       | string    | Source system identifier |

### timesheet_raw

| column_name         | type      | description              |
| ------------------- | --------- | ------------------------ |
| timesheet_id        | string    | Raw timesheet identifier |
| employee_id         | string    | Employee identifier      |
| project_id          | string    | Project identifier       |
| date                | string    | Work date                |
| hours_worked        | string    | Hours worked as string   |
| task_description    | string    | Task description         |
| billable            | string    | Billable flag            |
| approved            | string    | Approval status          |
| ingestion_timestamp | timestamp | When record was ingested |
| source_system       | string    | Source system identifier |

---

## Silver Layer - Cleaned and Validated Data

### employee_clean

| column_name        | type          | description                                        |
| ------------------ | ------------- | -------------------------------------------------- |
| employee_id        | bigint        | Cleaned employee identifier                        |
| first_name         | string        | Standardized first name                            |
| last_name          | string        | Standardized last name                             |
| full_name          | string        | Concatenated full name                             |
| email              | string        | Validated email address                            |
| department_code    | string        | Standardized department code                       |
| hire_date          | date          | Parsed hire date                                   |
| salary             | decimal(10,2) | Validated salary amount                            |
| manager_id         | bigint        | Manager identifier                                 |
| employment_status  | string        | Standardized status (Active, Inactive, Terminated) |
| is_active          | boolean       | Active employee flag                               |
| created_date       | timestamp     | Record creation timestamp                          |
| updated_date       | timestamp     | Last update timestamp                              |
| data_quality_score | decimal(3,2)  | Data quality score (0-1)                           |

### project_clean

| column_name    | type          | description                                                  |
| -------------- | ------------- | ------------------------------------------------------------ |
| project_id     | bigint        | Cleaned project identifier                                   |
| project_name   | string        | Standardized project name                                    |
| description    | string        | Project description                                          |
| start_date     | date          | Parsed start date                                            |
| end_date       | date          | Parsed end date                                              |
| budget_amount  | decimal(12,2) | Validated budget amount                                      |
| project_status | string        | Standardized status (Planning, Active, Completed, Cancelled) |
| owner_id       | bigint        | Project owner identifier                                     |
| priority_level | string        | Standardized priority (High, Medium, Low)                    |
| duration_days  | int           | Calculated project duration                                  |
| is_active      | boolean       | Active project flag                                          |
| created_date   | timestamp     | Record creation timestamp                                    |
| updated_date   | timestamp     | Last update timestamp                                        |

### timesheet_clean

| column_name      | type         | description                  |
| ---------------- | ------------ | ---------------------------- |
| timesheet_id     | bigint       | Cleaned timesheet identifier |
| employee_id      | bigint       | Employee identifier          |
| project_id       | bigint       | Project identifier           |
| work_date        | date         | Work date                    |
| hours_worked     | decimal(4,2) | Validated hours worked       |
| task_description | string       | Task description             |
| is_billable      | boolean      | Billable flag                |
| is_approved      | boolean      | Approval status              |
| overtime_hours   | decimal(4,2) | Calculated overtime hours    |
| regular_hours    | decimal(4,2) | Calculated regular hours     |
| created_date     | timestamp    | Record creation timestamp    |
| updated_date     | timestamp    | Last update timestamp        |

---

## Gold Layer - Business-Ready Star Schema

### Dimension Tables

#### dim_employee

| column_name       | type      | description                          |
| ----------------- | --------- | ------------------------------------ |
| employee_key      | bigint    | Surrogate key for employee dimension |
| employee_id       | bigint    | Business key from source system      |
| employee_name     | string    | Full employee name                   |
| email             | string    | Employee email                       |
| department_name   | string    | Department name                      |
| job_title         | string    | Job title                            |
| hire_date         | date      | Employee hire date                   |
| manager_name      | string    | Manager name                         |
| employment_status | string    | Current employment status            |
| salary_band       | string    | Salary band category                 |
| tenure_years      | int       | Years of service                     |
| is_current        | boolean   | Current record indicator             |
| effective_date    | date      | Record effective date                |
| expiry_date       | date      | Record expiry date                   |
| created_date      | timestamp | Record creation timestamp            |

#### dim_project

| column_name      | type    | description                         |
| ---------------- | ------- | ----------------------------------- |
| project_key      | bigint  | Surrogate key for project dimension |
| project_id       | bigint  | Business key from source system     |
| project_name     | string  | Project name                        |
| project_category | string  | Project category                    |
| project_type     | string  | Project type                        |
| priority_level   | string  | Priority level                      |
| project_manager  | string  | Project manager name                |
| department       | string  | Owning department                   |
| start_date       | date    | Project start date                  |
| planned_end_date | date    | Planned end date                    |
| actual_end_date  | date    | Actual end date                     |
| project_status   | string  | Current project status              |
| budget_range     | string  | Budget range category               |
| is_current       | boolean | Current record indicator            |
| effective_date   | date    | Record effective date               |
| expiry_date      | date    | Record expiry date                  |

#### dim_date

| column_name    | type    | description                     |
| -------------- | ------- | ------------------------------- |
| date_key       | int     | Surrogate key (YYYYMMDD format) |
| date           | date    | Actual date                     |
| year           | int     | Year                            |
| quarter        | int     | Quarter (1-4)                   |
| month          | int     | Month (1-12)                    |
| month_name     | string  | Month name                      |
| week_of_year   | int     | Week of year                    |
| day_of_month   | int     | Day of month                    |
| day_of_week    | int     | Day of week (1-7)               |
| day_name       | string  | Day name                        |
| is_weekend     | boolean | Weekend indicator               |
| is_holiday     | boolean | Holiday indicator               |
| fiscal_year    | int     | Fiscal year                     |
| fiscal_quarter | int     | Fiscal quarter                  |

### Fact Tables

#### fact_timesheet

| column_name        | type          | description                  |
| ------------------ | ------------- | ---------------------------- |
| timesheet_key      | bigint        | Surrogate key for fact table |
| employee_key       | bigint        | Foreign key to dim_employee  |
| project_key        | bigint        | Foreign key to dim_project   |
| date_key           | int           | Foreign key to dim_date      |
| timesheet_id       | bigint        | Business key from source     |
| hours_worked       | decimal(4,2)  | Total hours worked           |
| regular_hours      | decimal(4,2)  | Regular hours                |
| overtime_hours     | decimal(4,2)  | Overtime hours               |
| billable_hours     | decimal(4,2)  | Billable hours               |
| non_billable_hours | decimal(4,2)  | Non-billable hours           |
| hourly_rate        | decimal(8,2)  | Employee hourly rate         |
| total_cost         | decimal(10,2) | Total labor cost             |
| billable_amount    | decimal(10,2) | Billable amount              |
| is_approved        | boolean       | Approval status              |
| created_date       | timestamp     | Record creation timestamp    |

#### fact_project_metrics

| column_name            | type          | description                             |
| ---------------------- | ------------- | --------------------------------------- |
| project_metrics_key    | bigint        | Surrogate key for fact table            |
| project_key            | bigint        | Foreign key to dim_project              |
| date_key               | int           | Foreign key to dim_date (snapshot date) |
| total_budget           | decimal(12,2) | Total project budget                    |
| spent_to_date          | decimal(12,2) | Amount spent to date                    |
| remaining_budget       | decimal(12,2) | Remaining budget                        |
| budget_utilization_pct | decimal(5,2)  | Budget utilization percentage           |
| planned_hours          | decimal(8,2)  | Planned total hours                     |
| actual_hours           | decimal(8,2)  | Actual hours worked                     |
| remaining_hours        | decimal(8,2)  | Estimated remaining hours               |
| completion_percentage  | decimal(5,2)  | Project completion percentage           |
| active_employees       | int           | Number of active employees              |
| days_remaining         | int           | Days remaining to deadline              |
| is_on_track            | boolean       | On track indicator                      |
| risk_level             | string        | Risk level (Low, Medium, High)          |
| created_date           | timestamp     | Record creation timestamp               |
