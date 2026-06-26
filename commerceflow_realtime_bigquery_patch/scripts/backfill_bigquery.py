from __future__ import annotations

import argparse
from pathlib import Path

from backend.bigquery_client import backfill_sqlite_to_bigquery


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", default="commerceflow.db")
    parser.add_argument("--table-name", default="orders")
    parser.add_argument("--batch-size", type=int, default=3000)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    result = backfill_sqlite_to_bigquery(
        db_path=args.db_path,
        table_name=args.table_name,
        batch_size=args.batch_size,
        max_rows=args.max_rows,
    )
    print(result)


if __name__ == "__main__":
    main()
