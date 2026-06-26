from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List

DB_PATH = Path(__file__).resolve().parent.parent / "commerceflow.db"

ORDER_SCHEMA: Dict[str, str] = {
    "order_id": "TEXT PRIMARY KEY",
    "batch_id": "TEXT",
    "order_timestamp": "TEXT",
    "ingestion_timestamp": "TEXT",

    "customer_id": "TEXT",
    "customer_name": "TEXT",
    "customer_email": "TEXT",
    "city": "TEXT",
    "state": "TEXT",
    "region": "TEXT",
    "address": "TEXT",
    "latitude": "REAL",
    "longitude": "REAL",

    "product_id": "TEXT",
    "product_bought": "TEXT",
    "product_category": "TEXT",
    "order_value": "REAL",
    "order_weight_kg": "REAL",
    "product_fragility_score": "REAL",

    "dc_id": "TEXT",
    "dc_city": "TEXT",
    "dc_state": "TEXT",
    "dc_latitude": "REAL",
    "dc_longitude": "REAL",
    "distance_km": "REAL",

    "carrier": "TEXT",
    "carrier_base_delay_rate": "REAL",
    "promised_delivery_days": "INTEGER",
    "estimated_delivery_days": "INTEGER",
    "actual_delivery_days": "INTEGER",
    "delivery_status": "TEXT",
    "delay_risk_label": "INTEGER",
    "delay_probability_true": "REAL",

    "order_year": "INTEGER",
    "order_month": "INTEGER",
    "order_day": "INTEGER",
    "order_day_of_week": "INTEGER",
    "order_hour": "INTEGER",
    "is_weekend": "INTEGER",
    "is_peak_hour": "INTEGER",
    "weather_condition": "TEXT",
    "traffic_condition": "TEXT",
    "route_type": "TEXT",
}

ORDER_COLUMNS = list(ORDER_SCHEMA.keys())


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _existing_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {r["name"] for r in rows}


def init_db() -> None:
    conn = connect()
    try:
        col_defs = ",\n                ".join([f"{c} {t}" for c, t in ORDER_SCHEMA.items()])
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS orders (
                {col_defs}
            )
        """)

        # If the table already existed from an older app version, add missing columns.
        existing = _existing_columns(conn, "orders")
        for col, col_type in ORDER_SCHEMA.items():
            if col not in existing:
                # Cannot add PRIMARY KEY via ALTER. If order_id exists, this branch will not run.
                safe_type = col_type.replace(" PRIMARY KEY", "")
                conn.execute(f"ALTER TABLE orders ADD COLUMN {col} {safe_type}")

        conn.execute("CREATE INDEX IF NOT EXISTS idx_orders_ts ON orders(order_timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_orders_state ON orders(state)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_orders_delay ON orders(delay_risk_label)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_orders_carrier ON orders(carrier)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_orders_batch ON orders(batch_id)")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS ingestion_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_id TEXT,
                created_at TEXT,
                rows_generated INTEGER,
                rows_inserted_sqlite INTEGER,
                rows_inserted_bigquery INTEGER,
                status TEXT,
                message TEXT
            )
        """)
        conn.commit()
    finally:
        conn.close()


def insert_orders(orders: Iterable[Dict[str, Any]]) -> int:
    rows = list(orders)
    if not rows:
        return 0

    conn = connect()
    try:
        placeholders = ",".join(["?"] * len(ORDER_COLUMNS))
        sql = f"INSERT OR IGNORE INTO orders ({','.join(ORDER_COLUMNS)}) VALUES ({placeholders})"
        values = [[o.get(c) for c in ORDER_COLUMNS] for o in rows]
        cur = conn.executemany(sql, values)
        conn.commit()
        return cur.rowcount if cur.rowcount is not None else len(rows)
    finally:
        conn.close()


def log_ingestion(batch_id: str, rows_generated: int, rows_sqlite: int, rows_bigquery: int, status: str, message: str) -> None:
    from datetime import datetime, timezone

    conn = connect()
    try:
        conn.execute(
            """
            INSERT INTO ingestion_logs
            (batch_id, created_at, rows_generated, rows_inserted_sqlite, rows_inserted_bigquery, status, message)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (batch_id, datetime.now(timezone.utc).isoformat(), rows_generated, rows_sqlite, rows_bigquery, status, message),
        )
        conn.commit()
    finally:
        conn.close()


def get_orders(limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
    conn = connect()
    try:
        cur = conn.execute(
            "SELECT * FROM orders ORDER BY order_timestamp DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def get_ingestion_logs(limit: int = 50) -> List[Dict[str, Any]]:
    conn = connect()
    try:
        cur = conn.execute(
            "SELECT * FROM ingestion_logs ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def get_metrics() -> Dict[str, Any]:
    conn = connect()
    try:
        row = conn.execute(
            """
            SELECT
              COUNT(*) AS total_orders,
              COALESCE(SUM(order_value), 0) AS total_revenue,
              COALESCE(AVG(order_value), 0) AS avg_order_value,
              COALESCE(AVG(distance_km), 0) AS avg_distance_km,
              COALESCE(AVG(delay_risk_label), 0) AS delay_risk_rate
            FROM orders
            """
        ).fetchone()
        by_state = conn.execute(
            "SELECT state, COUNT(*) AS orders, SUM(order_value) AS revenue FROM orders GROUP BY state ORDER BY orders DESC"
        ).fetchall()
        by_carrier = conn.execute(
            "SELECT carrier, COUNT(*) AS orders, AVG(delay_risk_label) AS delay_risk_rate FROM orders GROUP BY carrier ORDER BY orders DESC"
        ).fetchall()
        by_traffic = conn.execute(
            "SELECT traffic_condition, COUNT(*) AS orders, AVG(delay_risk_label) AS delay_risk_rate FROM orders GROUP BY traffic_condition ORDER BY orders DESC"
        ).fetchall()
        return {
            **dict(row),
            "orders_by_state": [dict(r) for r in by_state],
            "orders_by_carrier": [dict(r) for r in by_carrier],
            "orders_by_traffic": [dict(r) for r in by_traffic],
        }
    finally:
        conn.close()


def export_orders_dataframe(limit: int | None = None):
    import pandas as pd
    conn = connect()
    try:
        query = "SELECT * FROM orders ORDER BY order_timestamp"
        if limit:
            query += f" LIMIT {int(limit)}"
        return pd.read_sql_query(query, conn)
    finally:
        conn.close()
