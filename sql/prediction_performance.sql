-- BigQuery model monitoring schema for daily CommerceFlow predictions.
-- Replace `your_project.commerce_gold.delay_risk_prediction_performance`
-- with your deployed table id.

CREATE TABLE IF NOT EXISTS `your_project.commerce_gold.delay_risk_prediction_performance` (
  run_id STRING,
  run_timestamp TIMESTAMP,
  run_date DATE,
  model_version STRING,
  scoring_mode STRING,
  endpoint_id STRING,

  evaluated_rows INT64,
  positive_labels INT64,
  positive_predictions INT64,

  accuracy FLOAT64,
  precision FLOAT64,
  recall FLOAT64,
  f1 FLOAT64,
  roc_auc FLOAT64,
  average_precision FLOAT64,

  true_positives INT64,
  false_positives INT64,
  false_negatives INT64,
  true_negatives INT64,

  avg_delay_probability FLOAT64,
  high_risk_orders INT64,
  medium_risk_orders INT64,
  low_risk_orders INT64,

  baseline_run_timestamp TIMESTAMP,
  baseline_accuracy FLOAT64,
  baseline_precision FLOAT64,
  baseline_recall FLOAT64,
  baseline_f1 FLOAT64,
  baseline_roc_auc FLOAT64,
  baseline_average_precision FLOAT64,

  accuracy_delta FLOAT64,
  precision_delta FLOAT64,
  recall_delta FLOAT64,
  f1_delta FLOAT64,
  roc_auc_delta FLOAT64,
  average_precision_delta FLOAT64,

  accuracy_drop_alert BOOL,
  f1_drop_alert BOOL,
  comparison_source STRING,
  created_at TIMESTAMP
)
PARTITION BY run_date
CLUSTER BY model_version, scoring_mode;
