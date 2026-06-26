import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from google.cloud import bigquery
from google.cloud import storage


PROJECT_ID = os.getenv("PROJECT_ID", "otimizador-cargas")
REGION = os.getenv("REGION", "us-central1")
ENDPOINT_ID = os.getenv("VERTEX_ENDPOINT_ID", "2085985213879418880")

SCORING_MODE = os.getenv("SCORING_MODE", "local").strip().lower()
SCORING_ENABLED = os.getenv("SCORING_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
ALLOW_VERTEX_ENDPOINT = os.getenv("ALLOW_VERTEX_ENDPOINT", "false").strip().lower() in {"1", "true", "yes", "on"}

FEATURE_TABLE = os.getenv("FEATURE_TABLE", "otimizador-cargas.commerce_gold.delay_risk_features")
PREDICTIONS_TABLE = os.getenv("PREDICTIONS_TABLE", "otimizador-cargas.commerce_gold.delay_risk_predictions")
RAW_ORDERS_TABLE = os.getenv("RAW_ORDERS_TABLE", "otimizador-cargas.commerce_raw.orders")
SILVER_ORDERS_TABLE = os.getenv("SILVER_ORDERS_TABLE", "otimizador-cargas.commerce_silver.orders_cleaned")

FEATURE_COLUMNS_GCS = os.getenv(
    "FEATURE_COLUMNS_GCS",
    "gs://commerceflow-ml-artifacts-otimizador-cargas/commerceflow/vertex_custom_model/v1/feature_columns.json",
)
FEATURE_COLUMNS_PATH = os.getenv("FEATURE_COLUMNS_PATH", "vertex_custom_model/feature_columns.json")
MODEL_LOCAL_PATH = os.getenv("MODEL_LOCAL_PATH", "vertex_custom_model/model.joblib")
MODEL_GCS_URI = os.getenv(
    "MODEL_GCS_URI",
    "gs://commerceflow-ml-artifacts-otimizador-cargas/commerceflow/vertex_custom_model/v1",
)

MODEL_VERSION = os.getenv("MODEL_VERSION", "commerceflow-delay-risk-local-v1")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "50"))
DAILY_SCORING_LIMIT = int(os.getenv("DAILY_SCORING_LIMIT", "300"))
BIGQUERY_MAX_BYTES_BILLED = int(os.getenv("BIGQUERY_MAX_BYTES_BILLED", "50000000"))
FEATURE_REFRESH_MAX_BYTES_BILLED = int(os.getenv("FEATURE_REFRESH_MAX_BYTES_BILLED", "2000000000"))
REFRESH_FEATURES_BEFORE_SCORING = os.getenv("REFRESH_FEATURES_BEFORE_SCORING", "false").strip().lower() in {"1", "true", "yes", "on"}
SCORING_LOOKBACK_DAYS = int(os.getenv("SCORING_LOOKBACK_DAYS", "7"))
PREDICTIONS_INSERT_CHUNK_SIZE = int(os.getenv("PREDICTIONS_INSERT_CHUNK_SIZE", "500"))

LOCAL_MODEL_THRESHOLD = float(os.getenv("LOCAL_MODEL_THRESHOLD", "0.5"))
HIGH_RISK_THRESHOLD = float(os.getenv("HIGH_RISK_THRESHOLD", "0.7"))
MEDIUM_RISK_THRESHOLD = float(os.getenv("MEDIUM_RISK_THRESHOLD", "0.4"))

MODEL = None
FEATURE_COLUMNS = None


def bq_job_config():
    if BIGQUERY_MAX_BYTES_BILLED <= 0:
        return None
    return bigquery.QueryJobConfig(maximum_bytes_billed=BIGQUERY_MAX_BYTES_BILLED)


def bq_feature_refresh_job_config():
    if FEATURE_REFRESH_MAX_BYTES_BILLED <= 0:
        return None
    return bigquery.QueryJobConfig(maximum_bytes_billed=FEATURE_REFRESH_MAX_BYTES_BILLED)


def refresh_feature_tables() -> None:
    bq = bigquery.Client(project=PROJECT_ID)

    silver_sql = f"""
    CREATE OR REPLACE TABLE `{SILVER_ORDERS_TABLE}`
    PARTITION BY DATE(order_timestamp)
    CLUSTER BY state, carrier, product_category, delay_risk_label AS
    SELECT
      *,
      DATE(order_timestamp) AS order_date
    FROM `{RAW_ORDERS_TABLE}`
    WHERE order_id IS NOT NULL
      AND order_timestamp IS NOT NULL
      AND order_value > 0
      AND distance_km >= 0
    """

    gold_sql = f"""
    CREATE OR REPLACE TABLE `{FEATURE_TABLE}`
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
      FROM `{SILVER_ORDERS_TABLE}`
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
    FROM scored
    """

    print(f"Refreshing {SILVER_ORDERS_TABLE} from {RAW_ORDERS_TABLE}")
    bq.query(f"DROP TABLE IF EXISTS `{SILVER_ORDERS_TABLE}`", job_config=bq_feature_refresh_job_config()).result()
    bq.query(silver_sql, job_config=bq_feature_refresh_job_config()).result()
    print(f"Refreshing {FEATURE_TABLE} from {SILVER_ORDERS_TABLE}")
    bq.query(f"DROP TABLE IF EXISTS `{FEATURE_TABLE}`", job_config=bq_feature_refresh_job_config()).result()
    bq.query(gold_sql, job_config=bq_feature_refresh_job_config()).result()
    print("FEATURE_REFRESH_SUCCESS")


def read_gcs_text(gcs_uri: str) -> str:
    path = gcs_uri.replace("gs://", "")
    bucket_name, blob_name = path.split("/", 1)

    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    return blob.download_as_text(encoding="utf-8")


def download_gcs_file(gcs_uri: str, local_path: Path) -> None:
    path = gcs_uri.replace("gs://", "")
    bucket_name, blob_name = path.split("/", 1)

    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(local_path))


def download_gcs_folder(gcs_uri: str) -> Path:
    path = gcs_uri.replace("gs://", "")
    bucket_name, prefix = path.split("/", 1)

    local_dir = Path(tempfile.mkdtemp(prefix="commerceflow_model_"))
    client = storage.Client(project=PROJECT_ID)

    for blob in client.list_blobs(bucket_name, prefix=prefix.rstrip("/") + "/"):
        if blob.name.endswith("/"):
            continue
        rel_path = blob.name.replace(prefix.rstrip("/") + "/", "")
        local_path = local_dir / rel_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local_path))

    return local_dir


def load_feature_columns() -> list[str]:
    global FEATURE_COLUMNS
    if FEATURE_COLUMNS is not None:
        return FEATURE_COLUMNS

    local_path = Path(FEATURE_COLUMNS_PATH)
    if local_path.exists():
        FEATURE_COLUMNS = json.loads(local_path.read_text(encoding="utf-8"))
        return FEATURE_COLUMNS

    FEATURE_COLUMNS = json.loads(read_gcs_text(FEATURE_COLUMNS_GCS))
    return FEATURE_COLUMNS


def unwrap_model_artifact(obj):
    if not isinstance(obj, dict):
        return patch_sklearn_compatibility(obj)

    for key in ["model", "best_model", "pipeline", "best_pipeline", "estimator", "trained_model", "final_model"]:
        if key in obj and hasattr(obj[key], "predict"):
            return patch_sklearn_compatibility(obj[key])

    raise RuntimeError(f"No model object with .predict() found. Artifact keys: {list(obj.keys())}")


def patch_sklearn_compatibility(model):
    try:
        if hasattr(model, "steps"):
            for _, step in model.steps:
                if step.__class__.__name__ == "LogisticRegression" and not hasattr(step, "multi_class"):
                    step.multi_class = "auto"
        elif model.__class__.__name__ == "LogisticRegression" and not hasattr(model, "multi_class"):
            model.multi_class = "auto"
    except Exception as exc:
        print(f"Compatibility patch warning: {exc}")

    return model


def load_local_model():
    global MODEL
    if MODEL is not None:
        return MODEL

    local_path = Path(MODEL_LOCAL_PATH)
    if local_path.exists():
        MODEL = unwrap_model_artifact(joblib.load(local_path))
        return MODEL

    if not MODEL_GCS_URI:
        raise RuntimeError("MODEL_LOCAL_PATH does not exist and MODEL_GCS_URI is not set.")

    if MODEL_GCS_URI.endswith(".joblib") or MODEL_GCS_URI.endswith(".pkl"):
        tmp_path = Path(tempfile.mkdtemp(prefix="commerceflow_model_")) / Path(MODEL_GCS_URI).name
        download_gcs_file(MODEL_GCS_URI, tmp_path)
        MODEL = unwrap_model_artifact(joblib.load(tmp_path))
        return MODEL

    model_dir = download_gcs_folder(MODEL_GCS_URI)
    model_path = model_dir / "model.joblib"
    if not model_path.exists():
        raise RuntimeError(f"model.joblib not found in downloaded MODEL_GCS_URI: {MODEL_GCS_URI}")

    MODEL = unwrap_model_artifact(joblib.load(model_path))
    return MODEL


def fetch_predictions_count_today() -> int:
    bq = bigquery.Client(project=PROJECT_ID)
    query = f"""
    SELECT COUNT(*) AS n
    FROM `{PREDICTIONS_TABLE}`
    WHERE DATE(prediction_timestamp) = CURRENT_DATE()
    """
    row = list(bq.query(query, job_config=bq_job_config()).result())[0]
    return int(row["n"] or 0)


def remaining_daily_quota() -> int:
    if DAILY_SCORING_LIMIT <= 0:
        return BATCH_SIZE

    try:
        already_scored = fetch_predictions_count_today()
    except Exception as exc:
        print(f"Could not read today's prediction count; using one batch only. Error: {exc}")
        return BATCH_SIZE

    return max(0, min(BATCH_SIZE, DAILY_SCORING_LIMIT - already_scored))


def fetch_unscored_orders(limit: int) -> pd.DataFrame:
    bq = bigquery.Client(project=PROJECT_ID)
    lookback_filter = ""
    if SCORING_LOOKBACK_DAYS > 0:
        lookback_filter = f"""
          AND f.order_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {int(SCORING_LOOKBACK_DAYS)} DAY)
        """

    query = f"""
    SELECT
      f.*
    FROM `{FEATURE_TABLE}` f
    LEFT JOIN `{PREDICTIONS_TABLE}` p
      ON f.order_id = p.order_id
    WHERE p.order_id IS NULL
      {lookback_filter}
    ORDER BY f.order_timestamp DESC
    LIMIT {int(limit)}
    """

    return bq.query(query, job_config=bq_job_config()).to_dataframe(create_bqstorage_client=False)


def predict_vertex(instances: list[dict]) -> list[dict]:
    if not ALLOW_VERTEX_ENDPOINT:
        raise RuntimeError(
            "Vertex endpoint scoring is blocked by default for cost control. "
            "Set ALLOW_VERTEX_ENDPOINT=true and SCORING_MODE=vertex to enable it intentionally."
        )

    from google.cloud import aiplatform_v1

    client_options = {"api_endpoint": f"{REGION}-aiplatform.googleapis.com"}
    client = aiplatform_v1.PredictionServiceClient(client_options=client_options)

    endpoint = client.endpoint_path(project=PROJECT_ID, location=REGION, endpoint=ENDPOINT_ID)
    response = client.predict(endpoint=endpoint, instances=instances)

    return [dict(prediction) for prediction in response.predictions]


def predict_local(instances: list[dict]) -> list[dict]:
    feature_columns = load_feature_columns()
    model = load_local_model()

    df = pd.DataFrame(instances)
    for col in feature_columns:
        if col not in df.columns:
            df[col] = None

    df = df[feature_columns]
    df = df.where(pd.notnull(df), None)

    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(df))[:, 1]
    else:
        probabilities = np.asarray(model.predict(df), dtype=float)

    output = []
    for prob in probabilities:
        prob = float(prob)
        pred = int(prob >= LOCAL_MODEL_THRESHOLD)
        if prob >= HIGH_RISK_THRESHOLD:
            risk_band = "high"
        elif prob >= MEDIUM_RISK_THRESHOLD:
            risk_band = "medium"
        else:
            risk_band = "low"

        output.append({
            "delay_probability": prob,
            "delay_prediction": pred,
            "risk_band": risk_band,
        })

    return output


def predict(instances: list[dict]) -> list[dict]:
    if SCORING_MODE == "vertex":
        return predict_vertex(instances)
    if SCORING_MODE == "local":
        return predict_local(instances)
    raise RuntimeError(f"Unsupported SCORING_MODE={SCORING_MODE!r}. Use local, vertex, or off.")


def insert_predictions(order_ids: list[str], predictions: list[dict]) -> None:
    bq = bigquery.Client(project=PROJECT_ID)
    now = datetime.now(timezone.utc).isoformat()

    rows = []
    for order_id, pred in zip(order_ids, predictions):
        rows.append({
            "order_id": order_id,
            "prediction_timestamp": now,
            "delay_probability": float(pred.get("delay_probability", 0.0)),
            "delay_prediction": int(pred.get("delay_prediction", 0)),
            "risk_band": str(pred.get("risk_band", "unknown")),
            "endpoint_id": "local" if SCORING_MODE == "local" else ENDPOINT_ID,
            "model_version": MODEL_VERSION,
        })

    errors = []
    for start in range(0, len(rows), PREDICTIONS_INSERT_CHUNK_SIZE):
        chunk = rows[start:start + PREDICTIONS_INSERT_CHUNK_SIZE]
        errors.extend(bq.insert_rows_json(PREDICTIONS_TABLE, chunk))

    if errors:
        raise RuntimeError(f"BigQuery insert errors: {errors}")

    print(f"Inserted {len(rows)} predictions into {PREDICTIONS_TABLE}")


def main():
    print("Starting CommerceFlow scoring job")
    print(f"SCORING_MODE={SCORING_MODE} BATCH_SIZE={BATCH_SIZE} DAILY_SCORING_LIMIT={DAILY_SCORING_LIMIT}")

    if not SCORING_ENABLED or SCORING_MODE in {"off", "disabled", "false"}:
        print("Scoring skipped by SCORING_ENABLED=false or SCORING_MODE=off.")
        return

    if REFRESH_FEATURES_BEFORE_SCORING:
        refresh_feature_tables()

    feature_columns = load_feature_columns()
    print(f"Loaded {len(feature_columns)} feature columns")

    remaining = remaining_daily_quota()
    if remaining <= 0:
        print("Daily scoring limit reached; no predictions will be generated.")
        return

    df = fetch_unscored_orders(limit=remaining)
    print(f"Fetched {len(df)} unscored orders")

    if df.empty:
        print("No new orders to score.")
        return

    if "order_id" not in df.columns:
        raise RuntimeError("order_id column not found in feature table.")

    order_ids = df["order_id"].astype(str).tolist()
    X = df.drop(columns=["order_id"], errors="ignore")

    missing_cols = [col for col in feature_columns if col not in X.columns]
    if missing_cols:
        print(f"Missing columns in BigQuery feature table: {missing_cols}")
        for col in missing_cols:
            X[col] = None

    X = X[feature_columns]

    for col in X.columns:
        if str(X[col].dtype).startswith("datetime"):
            X[col] = X[col].astype(str)

    X = X.where(pd.notnull(X), None)
    instances = X.to_dict(orient="records")

    predictions = predict(instances)
    print(f"Received {len(predictions)} predictions using {SCORING_MODE} mode")

    insert_predictions(order_ids, predictions)
    print("SCORING_SUCCESS")


if __name__ == "__main__":
    main()
