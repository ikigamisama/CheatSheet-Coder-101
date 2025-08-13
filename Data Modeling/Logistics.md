# Logistics Data Modeling Template - Medallion Architecture

## Bronze Layer (Raw Data Ingestion)

### bronze_shipments_raw

| column_name            | type      | description                         |
| ---------------------- | --------- | ----------------------------------- |
| shipment_id            | string    | Shipment identifier from source     |
| order_id               | string    | Associated order identifier         |
| carrier_id             | string    | Carrier/shipping company identifier |
| tracking_number        | string    | Carrier tracking number             |
| shipment_date          | string    | Shipment date (various formats)     |
| expected_delivery_date | string    | Expected delivery date              |
| actual_delivery_date   | string    | Actual delivery date                |
| origin_address         | string    | Origin address (raw)                |
| destination_address    | string    | Destination address (raw)           |
| shipment_status        | string    | Current shipment status             |
| service_type           | string    | Service type (ground, air, express) |
| weight                 | string    | Package weight                      |
| dimensions             | string    | Package dimensions                  |
| declared_value         | string    | Declared value                      |
| shipping_cost          | string    | Shipping cost                       |
| insurance_cost         | string    | Insurance cost                      |
| source_system          | string    | Source system identifier            |
| ingestion_timestamp    | timestamp | Timestamp when data was ingested    |

### bronze_routes_raw

| column_name         | type      | description                      |
| ------------------- | --------- | -------------------------------- |
| route_id            | string    | Route identifier                 |
| route_name          | string    | Route name                       |
| vehicle_id          | string    | Vehicle identifier               |
| driver_id           | string    | Driver identifier                |
| start_location      | string    | Route start location             |
| end_location        | string    | Route end location               |
| route_date          | string    | Route date                       |
| planned_start_time  | string    | Planned start time               |
| actual_start_time   | string    | Actual start time                |
| planned_end_time    | string    | Planned end time                 |
| actual_end_time     | string    | Actual end time                  |
| total_distance      | string    | Total route distance             |
| total_stops         | string    | Number of stops                  |
| fuel_consumption    | string    | Fuel consumption                 |
| route_status        | string    | Route status                     |
| source_system       | string    | Source system identifier         |
| ingestion_timestamp | timestamp | Timestamp when data was ingested |

### bronze_deliveries_raw

| column_name             | type      | description                      |
| ----------------------- | --------- | -------------------------------- |
| delivery_id             | string    | Delivery identifier              |
| shipment_id             | string    | Associated shipment identifier   |
| route_id                | string    | Associated route identifier      |
| customer_id             | string    | Customer identifier              |
| delivery_address        | string    | Delivery address (raw)           |
| scheduled_delivery_time | string    | Scheduled delivery time          |
| actual_delivery_time    | string    | Actual delivery time             |
| delivery_status         | string    | Delivery status                  |
| delivery_attempt        | string    | Delivery attempt number          |
| signature_required      | string    | Signature required flag          |
| signature_obtained      | string    | Signature obtained flag          |
| delivery_notes          | string    | Delivery notes                   |
| driver_id               | string    | Driver identifier                |
| delivery_proof          | string    | Proof of delivery type           |
| exception_code          | string    | Exception code if any            |
| source_system           | string    | Source system identifier         |
| ingestion_timestamp     | timestamp | Timestamp when data was ingested |

### bronze_vehicles_raw

| column_name          | type      | description                      |
| -------------------- | --------- | -------------------------------- |
| vehicle_id           | string    | Vehicle identifier               |
| vehicle_type         | string    | Vehicle type                     |
| make                 | string    | Vehicle make                     |
| model                | string    | Vehicle model                    |
| year                 | string    | Vehicle year                     |
| license_plate        | string    | License plate number             |
| vin                  | string    | Vehicle identification number    |
| capacity_weight      | string    | Weight capacity                  |
| capacity_volume      | string    | Volume capacity                  |
| fuel_type            | string    | Fuel type                        |
| current_mileage      | string    | Current mileage                  |
| maintenance_due_date | string    | Next maintenance due date        |
| insurance_expiry     | string    | Insurance expiry date            |
| registration_expiry  | string    | Registration expiry date         |
| vehicle_status       | string    | Vehicle status                   |
| assigned_depot       | string    | Assigned depot                   |
| source_system        | string    | Source system identifier         |
| ingestion_timestamp  | timestamp | Timestamp when data was ingested |

### bronze_drivers_raw

| column_name         | type      | description                      |
| ------------------- | --------- | -------------------------------- |
| driver_id           | string    | Driver identifier                |
| first_name          | string    | Driver first name                |
| last_name           | string    | Driver last name                 |
| license_number      | string    | Driver license number            |
| license_class       | string    | License class                    |
| license_expiry      | string    | License expiry date              |
| phone               | string    | Driver phone number              |
| email               | string    | Driver email                     |
| hire_date           | string    | Hire date                        |
| employment_status   | string    | Employment status                |
| assigned_depot      | string    | Assigned depot                   |
| certification       | string    | Special certifications           |
| safety_score        | string    | Safety score                     |
| performance_rating  | string    | Performance rating               |
| source_system       | string    | Source system identifier         |
| ingestion_timestamp | timestamp | Timestamp when data was ingested |

### bronze_warehouses_raw

| column_name            | type      | description                      |
| ---------------------- | --------- | -------------------------------- |
| warehouse_id           | string    | Warehouse identifier             |
| warehouse_name         | string    | Warehouse name                   |
| warehouse_type         | string    | Warehouse type                   |
| address                | string    | Warehouse address (raw)          |
| manager_id             | string    | Warehouse manager identifier     |
| capacity               | string    | Warehouse capacity               |
| current_utilization    | string    | Current utilization              |
| operating_hours        | string    | Operating hours                  |
| dock_doors             | string    | Number of dock doors             |
| storage_zones          | string    | Number of storage zones          |
| temperature_controlled | string    | Temperature controlled flag      |
| automated_systems      | string    | Automated systems available      |
| source_system          | string    | Source system identifier         |
| ingestion_timestamp    | timestamp | Timestamp when data was ingested |

## Silver Layer (Cleaned and Transformed Data)

### silver_shipments

| column_name               | type          | description                  |
| ------------------------- | ------------- | ---------------------------- |
| shipment_key              | bigint        | Surrogate key for shipment   |
| shipment_id               | string        | Cleaned shipment identifier  |
| order_id                  | string        | Associated order identifier  |
| carrier_id                | string        | Carrier identifier           |
| tracking_number           | string        | Tracking number              |
| shipment_date             | date          | Shipment date                |
| shipment_time             | time          | Shipment time                |
| expected_delivery_date    | date          | Expected delivery date       |
| actual_delivery_date      | date          | Actual delivery date         |
| origin_address_clean      | string        | Cleaned origin address       |
| origin_city               | string        | Origin city                  |
| origin_state              | string        | Origin state                 |
| origin_country            | string        | Origin country               |
| origin_postal_code        | string        | Origin postal code           |
| destination_address_clean | string        | Cleaned destination address  |
| destination_city          | string        | Destination city             |
| destination_state         | string        | Destination state            |
| destination_country       | string        | Destination country          |
| destination_postal_code   | string        | Destination postal code      |
| shipment_status_clean     | string        | Standardized shipment status |
| service_type_clean        | string        | Standardized service type    |
| weight_kg                 | decimal(10,3) | Package weight in kg         |
| length_cm                 | decimal(8,2)  | Package length in cm         |
| width_cm                  | decimal(8,2)  | Package width in cm          |
| height_cm                 | decimal(8,2)  | Package height in cm         |
| volume_cm3                | decimal(12,2) | Package volume in cm³        |
| declared_value            | decimal(12,2) | Declared value               |
| shipping_cost             | decimal(10,2) | Shipping cost                |
| insurance_cost            | decimal(8,2)  | Insurance cost               |
| total_cost                | decimal(12,2) | Total shipping cost          |
| distance_km               | decimal(10,2) | Distance in kilometers       |
| transit_days              | int           | Transit time in days         |
| is_express                | boolean       | Express delivery flag        |
| is_international          | boolean       | International shipment flag  |
| created_timestamp         | timestamp     | Record creation timestamp    |
| updated_timestamp         | timestamp     | Record last update timestamp |

### silver_routes

| column_name              | type          | description                  |
| ------------------------ | ------------- | ---------------------------- |
| route_key                | bigint        | Surrogate key for route      |
| route_id                 | string        | Route identifier             |
| route_name_clean         | string        | Cleaned route name           |
| vehicle_id               | string        | Vehicle identifier           |
| driver_id                | string        | Driver identifier            |
| route_date               | date          | Route date                   |
| planned_start_time       | timestamp     | Planned start time           |
| actual_start_time        | timestamp     | Actual start time            |
| planned_end_time         | timestamp     | Planned end time             |
| actual_end_time          | timestamp     | Actual end time              |
| planned_duration_minutes | int           | Planned duration in minutes  |
| actual_duration_minutes  | int           | Actual duration in minutes   |
| total_distance_km        | decimal(10,2) | Total distance in km         |
| total_stops              | int           | Number of stops              |
| completed_stops          | int           | Completed stops              |
| fuel_consumption_liters  | decimal(8,2)  | Fuel consumption in liters   |
| route_status_clean       | string        | Standardized route status    |
| delay_minutes            | int           | Delay in minutes             |
| efficiency_score         | decimal(5,2)  | Route efficiency score       |
| created_timestamp        | timestamp     | Record creation timestamp    |
| updated_timestamp        | timestamp     | Record last update timestamp |

### silver_deliveries

| column_name             | type      | description                  |
| ----------------------- | --------- | ---------------------------- |
| delivery_key            | bigint    | Surrogate key for delivery   |
| delivery_id             | string    | Delivery identifier          |
| shipment_id             | string    | Shipment identifier          |
| route_id                | string    | Route identifier             |
| customer_id             | string    | Customer identifier          |
| delivery_address_clean  | string    | Cleaned delivery address     |
| delivery_city           | string    | Delivery city                |
| delivery_state          | string    | Delivery state               |
| delivery_country        | string    | Delivery country             |
| delivery_postal_code    | string    | Delivery postal code         |
| scheduled_delivery_time | timestamp | Scheduled delivery time      |
| actual_delivery_time    | timestamp | Actual delivery time         |
| delivery_status_clean   | string    | Standardized delivery status |
| delivery_attempt        | int       | Delivery attempt number      |
| is_signature_required   | boolean   | Signature required flag      |
| is_signature_obtained   | boolean   | Signature obtained flag      |
| delivery_notes_clean    | string    | Cleaned delivery notes       |
| driver_id               | string    | Driver identifier            |
| delivery_proof_type     | string    | Proof of delivery type       |
| exception_code_clean    | string    | Cleaned exception code       |
| delivery_window_minutes | int       | Delivery window in minutes   |
| time_variance_minutes   | int       | Time variance from scheduled |
| is_successful           | boolean   | Successful delivery flag     |
| is_first_attempt        | boolean   | First attempt flag           |
| created_timestamp       | timestamp | Record creation timestamp    |

### silver_vehicles

| column_name              | type          | description                  |
| ------------------------ | ------------- | ---------------------------- |
| vehicle_key              | bigint        | Surrogate key for vehicle    |
| vehicle_id               | string        | Vehicle identifier           |
| vehicle_type_clean       | string        | Standardized vehicle type    |
| make_clean               | string        | Vehicle make                 |
| model_clean              | string        | Vehicle model                |
| year                     | int           | Vehicle year                 |
| license_plate_clean      | string        | Cleaned license plate        |
| vin_clean                | string        | Cleaned VIN                  |
| capacity_weight_kg       | decimal(10,2) | Weight capacity in kg        |
| capacity_volume_m3       | decimal(8,2)  | Volume capacity in m³        |
| fuel_type_clean          | string        | Standardized fuel type       |
| current_mileage          | int           | Current mileage              |
| maintenance_due_date     | date          | Next maintenance due date    |
| insurance_expiry_date    | date          | Insurance expiry date        |
| registration_expiry_date | date          | Registration expiry date     |
| vehicle_status_clean     | string        | Standardized vehicle status  |
| assigned_depot           | string        | Assigned depot               |
| vehicle_age_years        | int           | Vehicle age in years         |
| is_active                | boolean       | Vehicle active status        |
| is_maintenance_due       | boolean       | Maintenance due flag         |
| days_to_maintenance      | int           | Days until maintenance       |
| created_timestamp        | timestamp     | Record creation timestamp    |
| updated_timestamp        | timestamp     | Record last update timestamp |

### silver_drivers

| column_name             | type         | description                    |
| ----------------------- | ------------ | ------------------------------ |
| driver_key              | bigint       | Surrogate key for driver       |
| driver_id               | string       | Driver identifier              |
| full_name               | string       | Driver full name               |
| license_number_clean    | string       | Cleaned license number         |
| license_class_clean     | string       | Standardized license class     |
| license_expiry_date     | date         | License expiry date            |
| phone_clean             | string       | Cleaned phone number           |
| email_clean             | string       | Cleaned email                  |
| hire_date               | date         | Hire date                      |
| employment_status_clean | string       | Standardized employment status |
| assigned_depot          | string       | Assigned depot                 |
| certification_clean     | string       | Cleaned certifications         |
| safety_score            | decimal(5,2) | Safety score                   |
| performance_rating      | string       | Performance rating             |
| years_of_service        | int          | Years of service               |
| is_active               | boolean      | Driver active status           |
| is_license_expiring     | boolean      | License expiring flag          |
| days_to_license_expiry  | int          | Days to license expiry         |
| created_timestamp       | timestamp    | Record creation timestamp      |
| updated_timestamp       | timestamp    | Record last update timestamp   |

### silver_warehouses

| column_name               | type          | description                    |
| ------------------------- | ------------- | ------------------------------ |
| warehouse_key             | bigint        | Surrogate key for warehouse    |
| warehouse_id              | string        | Warehouse identifier           |
| warehouse_name_clean      | string        | Cleaned warehouse name         |
| warehouse_type_clean      | string        | Standardized warehouse type    |
| address_clean             | string        | Cleaned address                |
| city                      | string        | Warehouse city                 |
| state                     | string        | Warehouse state                |
| country                   | string        | Warehouse country              |
| postal_code               | string        | Warehouse postal code          |
| manager_id                | string        | Manager identifier             |
| capacity_m3               | decimal(12,2) | Capacity in cubic meters       |
| current_utilization_pct   | decimal(5,2)  | Current utilization percentage |
| operating_hours_clean     | string        | Cleaned operating hours        |
| dock_doors_count          | int           | Number of dock doors           |
| storage_zones_count       | int           | Number of storage zones        |
| is_temperature_controlled | boolean       | Temperature controlled flag    |
| automation_level          | string        | Automation level               |
| is_active                 | boolean       | Warehouse active status        |
| created_timestamp         | timestamp     | Record creation timestamp      |
| updated_timestamp         | timestamp     | Record last update timestamp   |

## Gold Layer (Facts and Dimensions)

### dim_carrier

| column_name          | type          | description                           |
| -------------------- | ------------- | ------------------------------------- |
| carrier_key          | bigint        | Primary key (surrogate)               |
| carrier_id           | string        | Business key                          |
| carrier_name         | string        | Carrier name                          |
| carrier_type         | string        | Carrier type (LTL, FTL, Parcel, etc.) |
| service_levels       | string        | Available service levels              |
| coverage_area        | string        | Coverage area                         |
| headquarters_city    | string        | Headquarters city                     |
| headquarters_country | string        | Headquarters country                  |
| contact_phone        | string        | Contact phone                         |
| contact_email        | string        | Contact email                         |
| payment_terms        | string        | Payment terms                         |
| insurance_coverage   | decimal(15,2) | Insurance coverage amount             |
| tracking_capability  | string        | Tracking capability level             |
| is_preferred         | boolean       | Preferred carrier flag                |
| is_active            | boolean       | Active status                         |
| effective_date       | date          | SCD Type 2 effective date             |
| expiry_date          | date          | SCD Type 2 expiry date                |
| is_current           | boolean       | Current record indicator              |
| version              | int           | Record version number                 |

### dim_location

| column_name           | type         | description                                |
| --------------------- | ------------ | ------------------------------------------ |
| location_key          | bigint       | Primary key (surrogate)                    |
| address               | string       | Full address                               |
| city                  | string       | City                                       |
| state                 | string       | State/province                             |
| country               | string       | Country                                    |
| postal_code           | string       | Postal code                                |
| latitude              | decimal(9,6) | Latitude coordinate                        |
| longitude             | decimal(9,6) | Longitude coordinate                       |
| time_zone             | string       | Time zone                                  |
| region                | string       | Geographic region                          |
| location_type         | string       | Location type (warehouse, customer, depot) |
| is_residential        | boolean      | Residential address flag                   |
| is_business           | boolean      | Business address flag                      |
| delivery_restrictions | string       | Delivery restrictions                      |
| access_requirements   | string       | Access requirements                        |
| is_active             | boolean      | Active status                              |

### dim_vehicle

| column_name               | type          | description                    |
| ------------------------- | ------------- | ------------------------------ |
| vehicle_key               | bigint        | Primary key (surrogate)        |
| vehicle_id                | string        | Business key                   |
| vehicle_type              | string        | Vehicle type                   |
| make                      | string        | Vehicle make                   |
| model                     | string        | Vehicle model                  |
| year                      | int           | Vehicle year                   |
| license_plate             | string        | License plate                  |
| vin                       | string        | Vehicle identification number  |
| capacity_weight_kg        | decimal(10,2) | Weight capacity                |
| capacity_volume_m3        | decimal(8,2)  | Volume capacity                |
| fuel_type                 | string        | Fuel type                      |
| fuel_efficiency_kmpl      | decimal(6,2)  | Fuel efficiency                |
| vehicle_category          | string        | Vehicle category               |
| assigned_depot            | string        | Assigned depot                 |
| vehicle_age_category      | string        | Age category                   |
| maintenance_schedule      | string        | Maintenance schedule           |
| is_temperature_controlled | boolean       | Temperature control capability |
| is_gps_enabled            | boolean       | GPS tracking enabled           |
| is_active                 | boolean       | Active status                  |
| effective_date            | date          | SCD Type 2 effective date      |
| expiry_date               | date          | SCD Type 2 expiry date         |
| is_current                | boolean       | Current record indicator       |
| version                   | int           | Record version number          |

### dim_driver

| column_name         | type    | description               |
| ------------------- | ------- | ------------------------- |
| driver_key          | bigint  | Primary key (surrogate)   |
| driver_id           | string  | Business key              |
| full_name           | string  | Driver full name          |
| license_number      | string  | License number            |
| license_class       | string  | License class             |
| license_expiry_date | date    | License expiry date       |
| hire_date           | date    | Hire date                 |
| employment_status   | string  | Employment status         |
| assigned_depot      | string  | Assigned depot            |
| certification_level | string  | Certification level       |
| safety_rating       | string  | Safety rating category    |
| performance_rating  | string  | Performance rating        |
| experience_level    | string  | Experience level          |
| years_of_service    | int     | Years of service          |
| specializations     | string  | Special certifications    |
| is_cdl_holder       | boolean | CDL holder flag           |
| is_hazmat_certified | boolean | Hazmat certified          |
| is_active           | boolean | Active status             |
| effective_date      | date    | SCD Type 2 effective date |
| expiry_date         | date    | SCD Type 2 expiry date    |
| is_current          | boolean | Current record indicator  |
| version             | int     | Record version number     |

### dim_service_type

| column_name        | type    | description                |
| ------------------ | ------- | -------------------------- |
| service_type_key   | bigint  | Primary key (surrogate)    |
| service_type_code  | string  | Service type code          |
| service_type_name  | string  | Service type name          |
| service_category   | string  | Service category           |
| delivery_speed     | string  | Delivery speed category    |
| is_express         | boolean | Express service flag       |
| is_overnight       | boolean | Overnight service flag     |
| is_ground          | boolean | Ground service flag        |
| is_air             | boolean | Air service flag           |
| tracking_level     | string  | Tracking detail level      |
| insurance_included | boolean | Insurance included flag    |
| signature_required | boolean | Signature required flag    |
| business_days_only | boolean | Business days only flag    |
| weekend_delivery   | boolean | Weekend delivery available |
| is_active          | boolean | Active status              |

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
| season          | string  | Season                 |
| fiscal_year     | int     | Fiscal year            |
| fiscal_quarter  | int     | Fiscal quarter         |
| fiscal_month    | int     | Fiscal month           |

### fact_shipments

| column_name                | type          | description                     |
| -------------------------- | ------------- | ------------------------------- |
| shipment_fact_key          | bigint        | Primary key (surrogate)         |
| shipment_id                | string        | Business key                    |
| carrier_key                | bigint        | Foreign key to dim_carrier      |
| origin_location_key        | bigint        | Foreign key to dim_location     |
| destination_location_key   | bigint        | Foreign key to dim_location     |
| service_type_key           | bigint        | Foreign key to dim_service_type |
| ship_date_key              | int           | Foreign key to dim_date         |
| expected_delivery_date_key | int           | Foreign key to dim_date         |
| actual_delivery_date_key   | int           | Foreign key to dim_date         |
| order_id                   | string        | Associated order ID             |
| tracking_number            | string        | Tracking number                 |
| shipment_status            | string        | Current status                  |
| weight_kg                  | decimal(10,3) | Package weight                  |
| volume_cm3                 | decimal(12,2) | Package volume                  |
| declared_value             | decimal(12,2) | Declared value                  |
| shipping_cost              | decimal(10,2) | Shipping cost                   |
| insurance_cost             | decimal(8,2)  | Insurance cost                  |
| total_cost                 | decimal(12,2) | Total cost                      |
| distance_km                | decimal(10,2) | Distance                        |
| planned_transit_days       | int           | Planned transit days            |
| actual_transit_days        | int           | Actual transit days             |
| transit_time_variance      | int           | Transit time variance           |
| is_on_time                 | boolean       | On-time delivery indicator      |
| is_damaged                 | boolean       | Damage indicator                |
| is_lost                    | boolean       | Lost package indicator          |
| delivery_attempts          | int           | Number of delivery attempts     |
| created_timestamp          | timestamp     | Record creation timestamp       |

### fact_deliveries

| column_name                 | type         | description                 |
| --------------------------- | ------------ | --------------------------- |
| delivery_fact_key           | bigint       | Primary key (surrogate)     |
| delivery_id                 | string       | Business key                |
| shipment_id                 | string       | Shipment identifier         |
| carrier_key                 | bigint       | Foreign key to dim_carrier  |
| vehicle_key                 | bigint       | Foreign key to dim_vehicle  |
| driver_key                  | bigint       | Foreign key to dim_driver   |
| delivery_location_key       | bigint       | Foreign key to dim_location |
| scheduled_date_key          | int          | Foreign key to dim_date     |
| actual_date_key             | int          | Foreign key to dim_date     |
| customer_id                 | string       | Customer identifier         |
| scheduled_delivery_time     | timestamp    | Scheduled delivery time     |
| actual_delivery_time        | timestamp    | Actual delivery time        |
| delivery_status             | string       | Delivery status             |
| delivery_attempt_number     | int          | Delivery attempt number     |
| time_window_start           | time         | Delivery window start       |
| time_window_end             | time         | Delivery window end         |
| delivery_duration_minutes   | int          | Delivery duration           |
| wait_time_minutes           | int          | Wait time at location       |
| distance_from_depot_km      | decimal(8,2) | Distance from depot         |
| is_successful               | boolean      | Successful delivery         |
| is_first_attempt_success    | boolean      | First attempt success       |
| is_within_time_window       | boolean      | Within time window          |
| is_signature_obtained       | boolean      | Signature obtained          |
| exception_code              | string       | Exception code              |
| proof_of_delivery_type      | string       | Proof of delivery type      |
| customer_satisfaction_score | decimal(3,2) | Customer satisfaction score |
| created_timestamp           | timestamp    | Record creation timestamp   |

### fact_routes

| column_name               | type          | description                |
| ------------------------- | ------------- | -------------------------- |
| route_fact_key            | bigint        | Primary key (surrogate)    |
| route_id                  | string        | Business key               |
| vehicle_key               | bigint        | Foreign key to dim_vehicle |
| driver_key                | bigint        | Foreign key to dim_driver  |
| route_date_key            | int           | Foreign key to dim_date    |
| route_name                | string        | Route name                 |
| planned_start_time        | timestamp     | Planned start time         |
| actual_start_time         | timestamp     | Actual start time          |
| planned_end_time          | timestamp     | Planned end time           |
| actual_end_time           | timestamp     | Actual end time            |
| planned_duration_minutes  | int           | Planned duration           |
| actual_duration_minutes   | int           | Actual duration            |
| duration_variance_minutes | int           | Duration variance          |
| total_distance_km         | decimal(10,2) | Total distance             |
| planned_stops             | int           | Planned stops              |
| actual_stops              | int           | Actual stops               |
| completed_stops           | int           | Completed stops            |
| successful_deliveries     | int           | Successful deliveries      |
| failed_deliveries         | int           | Failed deliveries          |
| fuel_consumption_liters   | decimal(8,2)  | Fuel consumption           |
| fuel_efficiency_kmpl      | decimal(6,2)  | Fuel efficiency            |
| route_efficiency_score    | decimal(5,2)  | Route efficiency score     |
| on_time_performance       | decimal(5,4)  | On-time performance rate   |
| cost_per_km               | decimal(8,4)  | Cost per kilometer         |
| revenue_per_km            | decimal(8,4)  | Revenue per kilometer      |
| profitability_score       | decimal(8,4)  | Route profitability        |
| created_timestamp         | timestamp     | Record creation timestamp  |

### fact_vehicle_utilization

| column_name            | type          | description                   |
| ---------------------- | ------------- | ----------------------------- |
| utilization_fact_key   | bigint        | Primary key (surrogate)       |
| vehicle_key            | bigint        | Foreign key to dim_vehicle    |
| date_key               | int           | Foreign key to dim_date       |
| utilization_date       | date          | Utilization date              |
| total_hours_available  | decimal(6,2)  | Total available hours         |
| total_hours_used       | decimal(6,2)  | Total hours used              |
| idle_hours             | decimal(6,2)  | Idle hours                    |
| maintenance_hours      | decimal(6,2)  | Maintenance hours             |
| distance_traveled_km   | decimal(10,2) | Distance traveled             |
| fuel_consumed_liters   | decimal(8,2)  | Fuel consumed                 |
| number_of_trips        | int           | Number of trips               |
| number_of_deliveries   | int           | Number of deliveries          |
| cargo_weight_kg        | decimal(12,2) | Total cargo weight            |
| cargo_volume_m3        | decimal(10,2) | Total cargo volume            |
| weight_utilization_pct | decimal(5,2)  | Weight utilization percentage |
| volume_utilization_pct | decimal(5,2)  | Volume utilization percentage |
| time_utilization_pct   | decimal(5,2)  | Time utilization percentage   |
| operating_cost         | decimal(10,2) | Operating cost                |
| revenue_generated      | decimal(12,2) | Revenue generated             |
| profit_margin          | decimal(8,4)  | Profit margin                 |
| efficiency_rating      | string        | Efficiency rating             |
| created_timestamp      | timestamp     | Record creation timestamp     |
