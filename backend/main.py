from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .bigquery_client import backfill_sqlite_to_bigquery, bigquery_enabled, insert_orders_bigquery
from .database import DB_PATH, get_ingestion_logs, get_metrics, get_orders, init_db, insert_orders, log_ingestion
from .order_generator import generate_orders

DEFAULT_ORDERS_PER_BATCH = 3000
DEFAULT_SIMULATION_MINUTES = 240
DEFAULT_BACKFILL_BATCH_SIZE = 3000


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return max(minimum, parsed)


ORDERS_PER_BATCH = _env_int("ORDERS_PER_BATCH", DEFAULT_ORDERS_PER_BATCH)
SIMULATION_MINUTES = _env_int("SIMULATION_MINUTES", DEFAULT_SIMULATION_MINUTES)
MAX_ORDERS_PER_BATCH = max(ORDERS_PER_BATCH, _env_int("MAX_ORDERS_PER_BATCH", ORDERS_PER_BATCH))
BIGQUERY_BACKFILL_BATCH_SIZE = _env_int("BIGQUERY_BACKFILL_BATCH_SIZE", DEFAULT_BACKFILL_BATCH_SIZE)
MAX_BIGQUERY_BACKFILL_BATCH_SIZE = max(
    BIGQUERY_BACKFILL_BATCH_SIZE,
    _env_int("MAX_BIGQUERY_BACKFILL_BATCH_SIZE", BIGQUERY_BACKFILL_BATCH_SIZE),
)
AUTO_START_SIMULATION = _env_bool("AUTO_START_SIMULATION", True)
GENERATE_INITIAL_BATCH = _env_bool("GENERATE_INITIAL_BATCH", False)

app = FastAPI(title="CommerceFlow AI API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

scheduler = BackgroundScheduler(timezone="UTC")


class GenerateRequest(BaseModel):
    n: int = Field(default=ORDERS_PER_BATCH, ge=1, le=MAX_ORDERS_PER_BATCH)


def generate_and_ingest_batch(n: int = ORDERS_PER_BATCH) -> Dict[str, Any]:
    batch_id = str(uuid.uuid4())
    orders = generate_orders(n=n, batch_id=batch_id)

    sqlite_rows = insert_orders(orders)
    bq_rows, bq_message = insert_orders_bigquery(orders)

    if bq_rows == len(orders) or not bigquery_enabled():
        status = "success"
    else:
        status = "warning"

    log_ingestion(batch_id, len(orders), sqlite_rows, bq_rows, status, bq_message)

    return {
        "batch_id": batch_id,
        "rows_generated": len(orders),
        "rows_inserted_sqlite": sqlite_rows,
        "rows_inserted_bigquery": bq_rows,
        "bigquery_message": bq_message,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def scheduled_job() -> None:
    generate_and_ingest_batch(ORDERS_PER_BATCH)


def schedule_order_generation(first_run_seconds: int | None = None) -> None:
    """
    Schedules the real-time order generator.

    The previous version used next_run_time=None, which can leave the job
    without a visible next run. This version explicitly schedules the next run.
    """
    if not scheduler.running:
        scheduler.start()

    first_run = datetime.now(timezone.utc) + timedelta(
        seconds=first_run_seconds if first_run_seconds is not None else SIMULATION_MINUTES * 60
    )

    scheduler.add_job(
        scheduled_job,
        "interval",
        minutes=SIMULATION_MINUTES,
        id="order_generation_job",
        replace_existing=True,
        next_run_time=first_run,
        max_instances=1,
        coalesce=True,
        misfire_grace_time=300,
    )


@app.on_event("startup")
def startup_event() -> None:
    init_db()
    if AUTO_START_SIMULATION:
        schedule_order_generation()
    if GENERATE_INITIAL_BATCH:
        generate_and_ingest_batch(ORDERS_PER_BATCH)


@app.on_event("shutdown")
def shutdown_event() -> None:
    if scheduler.running:
        scheduler.shutdown(wait=False)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "scheduler_running": scheduler.running,
        "bigquery_enabled": bigquery_enabled(),
        "db_path": str(DB_PATH),
        "orders_per_batch": ORDERS_PER_BATCH,
        "interval_minutes": SIMULATION_MINUTES,
    }


@app.get("/orders")
def list_orders(limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0)):
    return {"orders": get_orders(limit=limit, offset=offset)}


@app.get("/metrics")
def metrics():
    return get_metrics()


@app.get("/ingestion/logs")
def ingestion_logs(limit: int = Query(50, ge=1, le=500)):
    return {"logs": get_ingestion_logs(limit=limit)}


@app.post("/orders/generate")
def generate_orders_endpoint(
    payload: GenerateRequest | None = None,
    n: int | None = Query(None, ge=1, le=MAX_ORDERS_PER_BATCH),
):
    batch_size = n or (payload.n if payload else ORDERS_PER_BATCH)
    return generate_and_ingest_batch(n=batch_size)


@app.post("/simulation/start")
def start_simulation(run_first_in_seconds: int = Query(10, ge=0, le=3600)):
    schedule_order_generation(first_run_seconds=run_first_in_seconds)
    return {
        "status": "started",
        "orders_per_batch": ORDERS_PER_BATCH,
        "interval_minutes": SIMULATION_MINUTES,
        "first_run_in_seconds": run_first_in_seconds,
    }


@app.post("/simulation/stop")
def stop_simulation():
    if scheduler.get_job("order_generation_job"):
        scheduler.remove_job("order_generation_job")
    return {"status": "stopped"}


@app.post("/simulation/run-now")
def run_now(n: int = Query(ORDERS_PER_BATCH, ge=1, le=MAX_ORDERS_PER_BATCH)):
    return generate_and_ingest_batch(n=n)


@app.get("/simulation/status")
def simulation_status():
    job = scheduler.get_job("order_generation_job")
    return {
        "scheduler_running": scheduler.running,
        "job_exists": job is not None,
        "next_run_time": job.next_run_time.isoformat() if job and job.next_run_time else None,
        "orders_per_batch": ORDERS_PER_BATCH,
        "interval_minutes": SIMULATION_MINUTES,
        "max_orders_per_batch": MAX_ORDERS_PER_BATCH,
        "bigquery_enabled": bigquery_enabled(),
    }


@app.post("/bigquery/backfill")
def bigquery_backfill(
    batch_size: int = Query(BIGQUERY_BACKFILL_BATCH_SIZE, ge=100, le=MAX_BIGQUERY_BACKFILL_BATCH_SIZE),
    max_rows: int | None = Query(None, ge=1),
):
    """
    Backfills SQLite orders into BigQuery.

    Start small:
        POST /bigquery/backfill?max_rows=3000&batch_size=3000
    """
    return backfill_sqlite_to_bigquery(
        db_path=str(DB_PATH),
        table_name="orders",
        batch_size=batch_size,
        max_rows=max_rows,
    )




# ============================================================
# Vertex AI prediction endpoints for frontend dashboard
# ============================================================

try:
    from google.cloud import bigquery as _bigquery
except Exception:
    _bigquery = None

BIGQUERY_PROJECT = os.getenv("BIGQUERY_PROJECT", "otimizador-cargas")
PREDICTIONS_TABLE = os.getenv("PREDICTIONS_TABLE", "otimizador-cargas.commerce_gold.delay_risk_predictions")
PREDICTION_PERFORMANCE_TABLE = os.getenv(
    "PREDICTION_PERFORMANCE_TABLE",
    "otimizador-cargas.commerce_gold.delay_risk_prediction_performance",
)
BIGQUERY_PREDICTIONS_ENABLED = _env_bool("BIGQUERY_PREDICTIONS_ENABLED", True)
PREDICTIONS_CACHE_SECONDS = _env_int("PREDICTIONS_CACHE_SECONDS", 900, minimum=0)
BIGQUERY_MAX_BYTES_BILLED = _env_int("BIGQUERY_MAX_BYTES_BILLED", 100_000_000, minimum=0)
_PREDICTIONS_CACHE: dict[str, tuple[datetime, Dict[str, Any]]] = {}


def _bq_client():
    if _bigquery is None:
        raise RuntimeError("google-cloud-bigquery is not installed.")
    return _bigquery.Client(project=BIGQUERY_PROJECT)


def _run_bq_query(client, query: str):
    job_config = None
    if BIGQUERY_MAX_BYTES_BILLED:
        job_config = _bigquery.QueryJobConfig(maximum_bytes_billed=BIGQUERY_MAX_BYTES_BILLED)
    return client.query(query, job_config=job_config).result()


def _cached_prediction_response(key: str, loader) -> Dict[str, Any]:
    if PREDICTIONS_CACHE_SECONDS <= 0:
        return loader()

    now = datetime.now(timezone.utc)
    cached = _PREDICTIONS_CACHE.get(key)
    if cached and (now - cached[0]).total_seconds() < PREDICTIONS_CACHE_SECONDS:
        return cached[1]

    data = loader()
    _PREDICTIONS_CACHE[key] = (now, data)
    return data


def _empty_predictions_summary() -> Dict[str, Any]:
    return {
        "total_predictions": 0,
        "avg_delay_probability": 0.0,
        "high_risk_orders": 0,
        "medium_risk_orders": 0,
        "low_risk_orders": 0,
        "last_prediction_timestamp": None,
        "risk_distribution": [],
    }


def _empty_prediction_performance() -> Dict[str, Any]:
    return {
        "latest": None,
        "history": [],
    }


@app.get("/predictions/summary")
def predictions_summary():
    if not BIGQUERY_PREDICTIONS_ENABLED:
        return _empty_predictions_summary()

    return _cached_prediction_response("summary", _load_predictions_summary)


def _load_predictions_summary() -> Dict[str, Any]:
    client = _bq_client()

    query = f"""
    SELECT
      COUNT(*) AS total_predictions,
      AVG(delay_probability) AS avg_delay_probability,
      SUM(CASE WHEN risk_band = 'high' THEN 1 ELSE 0 END) AS high_risk_orders,
      SUM(CASE WHEN risk_band = 'medium' THEN 1 ELSE 0 END) AS medium_risk_orders,
      SUM(CASE WHEN risk_band = 'low' THEN 1 ELSE 0 END) AS low_risk_orders,
      MAX(prediction_timestamp) AS last_prediction_timestamp
    FROM `{PREDICTIONS_TABLE}`
    """

    row = list(_run_bq_query(client, query))[0]

    risk_query = f"""
    SELECT
      risk_band,
      COUNT(*) AS orders,
      AVG(delay_probability) AS avg_delay_probability
    FROM `{PREDICTIONS_TABLE}`
    GROUP BY risk_band
    ORDER BY orders DESC
    """

    risk_rows = [
        {
            "risk_band": r["risk_band"],
            "orders": int(r["orders"]),
            "avg_delay_probability": float(r["avg_delay_probability"] or 0),
        }
        for r in _run_bq_query(client, risk_query)
    ]

    return {
        "total_predictions": int(row["total_predictions"] or 0),
        "avg_delay_probability": float(row["avg_delay_probability"] or 0),
        "high_risk_orders": int(row["high_risk_orders"] or 0),
        "medium_risk_orders": int(row["medium_risk_orders"] or 0),
        "low_risk_orders": int(row["low_risk_orders"] or 0),
        "last_prediction_timestamp": str(row["last_prediction_timestamp"]) if row["last_prediction_timestamp"] else None,
        "risk_distribution": risk_rows,
    }


@app.get("/predictions/latest")
def latest_predictions(limit: int = Query(100, ge=1, le=500)):
    if not BIGQUERY_PREDICTIONS_ENABLED:
        return {"predictions": []}

    return _cached_prediction_response(f"latest:{limit}", lambda: _load_latest_predictions(limit))


def _load_latest_predictions(limit: int) -> Dict[str, Any]:
    client = _bq_client()

    query = f"""
    SELECT
      order_id,
      prediction_timestamp,
      delay_probability,
      delay_prediction,
      risk_band,
      endpoint_id,
      model_version
    FROM `{PREDICTIONS_TABLE}`
    ORDER BY prediction_timestamp DESC
    LIMIT {limit}
    """

    rows = []
    for r in _run_bq_query(client, query):
        rows.append({
            "order_id": r["order_id"],
            "prediction_timestamp": str(r["prediction_timestamp"]),
            "delay_probability": float(r["delay_probability"] or 0),
            "delay_prediction": int(r["delay_prediction"] or 0),
            "risk_band": r["risk_band"],
            "endpoint_id": r["endpoint_id"],
            "model_version": r["model_version"],
        })

    return {"predictions": rows}


@app.get("/predictions/performance")
def prediction_performance(limit: int = Query(14, ge=1, le=90)):
    if not BIGQUERY_PREDICTIONS_ENABLED:
        return _empty_prediction_performance()

    return _cached_prediction_response(
        f"performance:{limit}",
        lambda: _load_prediction_performance(limit),
    )


def _load_prediction_performance(limit: int) -> Dict[str, Any]:
    client = _bq_client()

    query = f"""
    SELECT
      run_id,
      run_timestamp,
      run_date,
      model_version,
      scoring_mode,
      endpoint_id,
      evaluated_rows,
      positive_labels,
      positive_predictions,
      accuracy,
      precision,
      recall,
      f1,
      roc_auc,
      average_precision,
      true_positives,
      false_positives,
      false_negatives,
      true_negatives,
      avg_delay_probability,
      high_risk_orders,
      medium_risk_orders,
      low_risk_orders,
      baseline_run_timestamp,
      baseline_accuracy,
      baseline_precision,
      baseline_recall,
      baseline_f1,
      baseline_roc_auc,
      baseline_average_precision,
      accuracy_delta,
      precision_delta,
      recall_delta,
      f1_delta,
      roc_auc_delta,
      average_precision_delta,
      accuracy_drop_alert,
      f1_drop_alert,
      comparison_source,
      created_at
    FROM `{PREDICTION_PERFORMANCE_TABLE}`
    ORDER BY run_timestamp DESC
    LIMIT {limit}
    """

    try:
        result = _run_bq_query(client, query)
    except Exception as exc:
        if "Not found: Table" in str(exc) or "Not found: Dataset" in str(exc):
            return _empty_prediction_performance()
        raise

    rows = [_prediction_performance_row(r) for r in result]
    return {
        "latest": rows[0] if rows else None,
        "history": rows,
    }


def _prediction_performance_row(row) -> Dict[str, Any]:
    return {
        "run_id": row["run_id"],
        "run_timestamp": str(row["run_timestamp"]) if row["run_timestamp"] else None,
        "run_date": str(row["run_date"]) if row["run_date"] else None,
        "model_version": row["model_version"],
        "scoring_mode": row["scoring_mode"],
        "endpoint_id": row["endpoint_id"],
        "evaluated_rows": int(row["evaluated_rows"] or 0),
        "positive_labels": int(row["positive_labels"] or 0),
        "positive_predictions": int(row["positive_predictions"] or 0),
        "accuracy": float(row["accuracy"] or 0),
        "precision": float(row["precision"] or 0),
        "recall": float(row["recall"] or 0),
        "f1": float(row["f1"] or 0),
        "roc_auc": float(row["roc_auc"]) if row["roc_auc"] is not None else None,
        "average_precision": float(row["average_precision"]) if row["average_precision"] is not None else None,
        "true_positives": int(row["true_positives"] or 0),
        "false_positives": int(row["false_positives"] or 0),
        "false_negatives": int(row["false_negatives"] or 0),
        "true_negatives": int(row["true_negatives"] or 0),
        "avg_delay_probability": float(row["avg_delay_probability"] or 0),
        "high_risk_orders": int(row["high_risk_orders"] or 0),
        "medium_risk_orders": int(row["medium_risk_orders"] or 0),
        "low_risk_orders": int(row["low_risk_orders"] or 0),
        "baseline_run_timestamp": str(row["baseline_run_timestamp"]) if row["baseline_run_timestamp"] else None,
        "baseline_accuracy": float(row["baseline_accuracy"]) if row["baseline_accuracy"] is not None else None,
        "baseline_precision": float(row["baseline_precision"]) if row["baseline_precision"] is not None else None,
        "baseline_recall": float(row["baseline_recall"]) if row["baseline_recall"] is not None else None,
        "baseline_f1": float(row["baseline_f1"]) if row["baseline_f1"] is not None else None,
        "baseline_roc_auc": float(row["baseline_roc_auc"]) if row["baseline_roc_auc"] is not None else None,
        "baseline_average_precision": (
            float(row["baseline_average_precision"])
            if row["baseline_average_precision"] is not None
            else None
        ),
        "accuracy_delta": float(row["accuracy_delta"]) if row["accuracy_delta"] is not None else None,
        "precision_delta": float(row["precision_delta"]) if row["precision_delta"] is not None else None,
        "recall_delta": float(row["recall_delta"]) if row["recall_delta"] is not None else None,
        "f1_delta": float(row["f1_delta"]) if row["f1_delta"] is not None else None,
        "roc_auc_delta": float(row["roc_auc_delta"]) if row["roc_auc_delta"] is not None else None,
        "average_precision_delta": (
            float(row["average_precision_delta"])
            if row["average_precision_delta"] is not None
            else None
        ),
        "accuracy_drop_alert": bool(row["accuracy_drop_alert"]),
        "f1_drop_alert": bool(row["f1_drop_alert"]),
        "comparison_source": row["comparison_source"],
        "created_at": str(row["created_at"]) if row["created_at"] else None,
    }
