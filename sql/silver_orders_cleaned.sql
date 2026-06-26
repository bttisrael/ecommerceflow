CREATE OR REPLACE TABLE `your_project.commerce_silver.orders_cleaned`
PARTITION BY DATE(order_timestamp)
CLUSTER BY state, carrier, product_category, delay_risk_label AS
SELECT
  *,
  DATE(order_timestamp) AS order_date
FROM `your_project.commerce_raw.orders`
WHERE order_id IS NOT NULL
  AND order_timestamp IS NOT NULL
  AND order_value > 0
  AND distance_km >= 0;
