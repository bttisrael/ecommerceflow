from __future__ import annotations

import os
import sqlite3
from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Tuple

BIGQUERY_TABLE_ENV = "BIGQUERY_ORDERS_TABLE"  # format: project.dataset.table
BIGQUERY_INGESTION_ENABLED_ENV = "BIGQUERY_INGESTION_ENABLED"
BIGQUERY_INSERT_CHUNK_SIZE_ENV = "BIGQUERY_INSERT_CHUNK_SIZE"


def _env_bool(name: str, default: bool = True) -> bool:
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


def bigquery_enabled() -> bool:
    return _env_bool(BIGQUERY_INGESTION_ENABLED_ENV, True) and bool(os.getenv(BIGQUERY_TABLE_ENV))


def normalize_for_bigquery(order: Dict[str, Any]) -> Dict[str, Any]:
    normalized = {}
    for k, v in dict(order).items():
        if isinstance(v, (datetime, date)):
            normalized[k] = v.isoformat()
        else:
            normalized[k] = v
    return normalized


def insert_orders_bigquery(orders: Iterable[Dict[str, Any]]) -> Tuple[int, str]:
    rows = [normalize_for_bigquery(o) for o in orders]
    if not rows:
        return 0, "No rows to insert."

    table_id = os.getenv(BIGQUERY_TABLE_ENV)
    if not table_id:
        return 0, "BigQuery disabled: BIGQUERY_ORDERS_TABLE not set."
    if not _env_bool(BIGQUERY_INGESTION_ENABLED_ENV, True):
        return 0, "BigQuery disabled: BIGQUERY_INGESTION_ENABLED=false."

    try:
        from google.cloud import bigquery

        client = bigquery.Client()
        chunk_size = _env_int(BIGQUERY_INSERT_CHUNK_SIZE_ENV, 3000)
        total_inserted = 0

        for start in range(0, len(rows), chunk_size):
            chunk = rows[start:start + chunk_size]
            errors = client.insert_rows_json(table_id, chunk)
            if errors:
                return total_inserted, f"BigQuery insertion errors: {errors[:3]}"
            total_inserted += len(chunk)

        return total_inserted, f"BigQuery insert success ({total_inserted} rows)."
    except Exception as exc:
        return 0, f"BigQuery insert failed: {type(exc).__name__}: {exc}"


def backfill_sqlite_to_bigquery(
    db_path: str,
    table_name: str = "orders",
    batch_size: int = 3000,
    max_rows: int | None = None,
) -> Dict[str, Any]:
    """
    Backfills historical SQLite orders into BigQuery in batches.

    Use carefully with 2M rows. Start with max_rows=3000 to validate credentials/schema.
    """
    table_id = os.getenv(BIGQUERY_TABLE_ENV)
    if not table_id:
        return {"status": "skipped", "message": "BIGQUERY_ORDERS_TABLE not set.", "rows_inserted": 0}

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    total_inserted = 0
    offset = 0
    try:
        while True:
            limit = batch_size
            if max_rows is not None:
                remaining = max_rows - total_inserted
                if remaining <= 0:
                    break
                limit = min(limit, remaining)

            rows = conn.execute(
                f"SELECT * FROM {table_name} ORDER BY order_timestamp LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()

            if not rows:
                break

            payload = [dict(r) for r in rows]
            inserted, message = insert_orders_bigquery(payload)
            if inserted == 0 and "success" not in message.lower():
                return {
                    "status": "error",
                    "message": message,
                    "rows_inserted": total_inserted,
                    "offset": offset,
                }

            total_inserted += inserted
            offset += len(rows)

        return {"status": "success", "rows_inserted": total_inserted, "message": "Backfill completed."}
    finally:
        conn.close()
