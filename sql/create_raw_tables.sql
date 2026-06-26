-- BigQuery raw schema for CommerceFlow AI
-- Replace `your_project.commerce_raw.orders` with your real project/dataset/table.
-- Example final table id used by the API:
-- BIGQUERY_ORDERS_TABLE=your_project.commerce_raw.orders

CREATE SCHEMA IF NOT EXISTS `your_project.commerce_raw`;

CREATE TABLE IF NOT EXISTS `your_project.commerce_raw.orders` (
  order_id STRING,
  batch_id STRING,
  order_timestamp TIMESTAMP,
  ingestion_timestamp TIMESTAMP,

  customer_id STRING,
  customer_name STRING,
  customer_email STRING,
  city STRING,
  state STRING,
  region STRING,
  address STRING,
  latitude FLOAT64,
  longitude FLOAT64,

  product_id STRING,
  product_bought STRING,
  product_category STRING,
  order_value FLOAT64,
  order_weight_kg FLOAT64,
  product_fragility_score FLOAT64,

  dc_id STRING,
  dc_city STRING,
  dc_state STRING,
  dc_latitude FLOAT64,
  dc_longitude FLOAT64,
  distance_km FLOAT64,

  carrier STRING,
  carrier_base_delay_rate FLOAT64,
  promised_delivery_days INT64,
  estimated_delivery_days INT64,
  actual_delivery_days INT64,
  delivery_status STRING,
  delay_risk_label INT64,
  delay_probability_true FLOAT64,

  order_year INT64,
  order_month INT64,
  order_day INT64,
  order_day_of_week INT64,
  order_hour INT64,
  is_weekend INT64,
  is_peak_hour INT64,
  weather_condition STRING,
  traffic_condition STRING,
  route_type STRING
)
PARTITION BY DATE(order_timestamp)
CLUSTER BY state, carrier, product_category, delay_risk_label;


CREATE TABLE IF NOT EXISTS `your_project.commerce_raw.ingestion_logs` (
  id STRING,
  batch_id STRING,
  created_at TIMESTAMP,
  rows_generated INT64,
  rows_inserted_sqlite INT64,
  rows_inserted_bigquery INT64,
  status STRING,
  message STRING
)
PARTITION BY DATE(created_at)
CLUSTER BY status;
