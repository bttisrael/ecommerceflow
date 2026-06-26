# CommerceFlow AI realtime + BigQuery patch

Files included:

```text
backend/main.py              # fixed APScheduler auto-ingestion and BigQuery backfill endpoints
backend/database.py          # full 43-column operational schema + migration support
backend/order_generator.py   # live orders now include traffic/weather/route/date features
backend/bigquery_client.py   # BigQuery streaming insert + SQLite historical backfill
sql/create_raw_tables.sql    # full BigQuery raw schema
scripts/backfill_bigquery.py # CLI backfill from SQLite to BigQuery
```

## Quick test auto-ingestion locally

```powershell
cd "C:\Users\israb\Documents\ML-production"
.\.venv\Scripts\activate

$env:ORDERS_PER_BATCH="3000"
$env:SIMULATION_MINUTES="240"
$env:MAX_ORDERS_PER_BATCH="3000"
$env:AUTO_START_SIMULATION="true"
$env:GENERATE_INITIAL_BATCH="false"

uvicorn backend.main:app --reload
```

Check:

```text
http://localhost:8000/simulation/status
```

For a fast test, call:

```powershell
Invoke-RestMethod -Method Post "http://localhost:8000/simulation/start?run_first_in_seconds=10"
```

Then wait 10 seconds and check total orders.

## BigQuery env

```env
BIGQUERY_INGESTION_ENABLED=true
BIGQUERY_INSERT_CHUNK_SIZE=3000
BIGQUERY_ORDERS_TABLE=your_project.commerce_raw.orders
GOOGLE_APPLICATION_CREDENTIALS=C:\path\to\service-account.json
```

## Backfill test

```powershell
python scripts\backfill_bigquery.py --max-rows 3000 --batch-size 3000
```
