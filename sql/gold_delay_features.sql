CREATE OR REPLACE TABLE `your_project.commerce_gold.delay_risk_features`
PARTITION BY DATE(order_timestamp)
CLUSTER BY state, carrier, product_category, delay_risk_label AS
WITH base AS (
  SELECT
    order_id,
    order_timestamp,
    delay_risk_label,
    order_value,
    order_weight_kg,
    product_fragility_score,
    distance_km,
    carrier_base_delay_rate,
    promised_delivery_days,
    estimated_delivery_days,
    city,
    state,
    region,
    product_category,
    carrier,
    dc_id,
    dc_state,
    traffic_condition,
    weather_condition,
    route_type,
    order_month,
    order_day,
    order_day_of_week,
    order_hour,
    is_weekend,
    is_peak_hour
  FROM `your_project.commerce_silver.orders_cleaned`
),
scored AS (
  SELECT
    *,
    estimated_delivery_days - promised_delivery_days AS estimated_minus_promised_days,
    SAFE_DIVIDE(estimated_delivery_days, NULLIF(promised_delivery_days, 0)) AS estimated_over_promised,
    LOG(1 + GREATEST(order_value, 0)) AS log_order_value,
    LOG(1 + GREATEST(distance_km, 0)) AS log_distance_km,
    LOG(1 + GREATEST(order_weight_kg, 0)) AS log_weight_kg,
    SAFE_DIVIDE(order_value, NULLIF(order_weight_kg, 0)) AS value_per_kg,
    SAFE_DIVIDE(order_value, NULLIF(distance_km, 0)) AS value_per_km,
    order_weight_kg * distance_km AS weight_distance_interaction,
    SIN(2 * ACOS(-1) * order_hour / 24) AS hour_sin,
    COS(2 * ACOS(-1) * order_hour / 24) AS hour_cos,
    SIN(2 * ACOS(-1) * order_day_of_week / 7) AS dow_sin,
    COS(2 * ACOS(-1) * order_day_of_week / 7) AS dow_cos,
    SIN(2 * ACOS(-1) * order_month / 12) AS month_sin,
    COS(2 * ACOS(-1) * order_month / 12) AS month_cos,
    CASE WHEN state = dc_state THEN 1 ELSE 0 END AS same_state_dc,
    CASE
      WHEN distance_km < 300 THEN 'short'
      WHEN distance_km < 900 THEN 'medium'
      WHEN distance_km < 1600 THEN 'long'
      ELSE 'very_long'
    END AS distance_band,
    CASE
      WHEN order_weight_kg < 1 THEN 'light'
      WHEN order_weight_kg < 5 THEN 'medium'
      WHEN order_weight_kg < 15 THEN 'heavy'
      ELSE 'bulky'
    END AS weight_band,
    CASE LOWER(COALESCE(traffic_condition, ''))
      WHEN 'light' THEN 0.15
      WHEN 'normal' THEN 0.30
      WHEN 'moderate' THEN 0.45
      WHEN 'heavy' THEN 0.75
      WHEN 'severe' THEN 0.90
      ELSE 0.35
    END AS traffic_risk_score,
    CASE LOWER(COALESCE(weather_condition, ''))
      WHEN 'clear' THEN 0.10
      WHEN 'cloudy' THEN 0.25
      WHEN 'rain' THEN 0.55
      WHEN 'storm' THEN 0.85
      WHEN 'fog' THEN 0.65
      ELSE 0.35
    END AS weather_risk_score,
    CASE LOWER(COALESCE(route_type, ''))
      WHEN 'urban' THEN 0.30
      WHEN 'regional' THEN 0.45
      WHEN 'highway' THEN 0.25
      WHEN 'remote' THEN 0.75
      ELSE 0.40
    END AS route_risk_score
  FROM base
)
SELECT
  *,
  (traffic_risk_score + weather_risk_score + route_risk_score) / 3 AS combined_operational_risk
FROM scored;
