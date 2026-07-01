# CommerceFlow AI - Current Project Status

CommerceFlow AI is a production-style portfolio project for simulated e-commerce operations, BigQuery data pipelines, and delay-risk model scoring.

The project is no longer just a local simulator. It now has a deployed FastAPI backend, scheduled order generation, BigQuery raw/silver/gold tables, a daily scoring Cloud Run Job, a React/Vite monitoring dashboard, and model-performance monitoring code.

## Current Status

Status date: 2026-07-01

Repository:

- GitHub repository: `https://github.com/bttisrael/ecommerceflow`
- Main branch latest local/deployed commit checked during this update: `25ae362 Add daily prediction performance monitoring`
- CI builds the frontend, checks Python syntax, and builds backend/scoring Docker images.
- CD deploys the backend to Cloud Run, the scoring job to Cloud Run Jobs, and the frontend to Vercel when the required secrets are present.

Live GCP project:

- Project: `otimizador-cargas`
- Region: `us-central1`
- Runtime service account: `commerceflow-runner@otimizador-cargas.iam.gserviceaccount.com`

Cloud Run services:

- `ecommerceflow-api`
  - Status: ready
  - URL: `https://ecommerceflow-api-lmcbphe4ha-uc.a.run.app`
  - Current image tag checked during this update: `25ae362cb7b0d17a97e3115fb2b7d7d333fa42fc`
  - Health status: `ok`
  - BigQuery ingestion: enabled
  - Internal APScheduler in Cloud Run: disabled intentionally; Cloud Scheduler triggers production order generation.
- `commerceflow-stop-billing`
  - Status: ready
  - Used by the optional budget-cap flow.

Cloud Scheduler:

- `ecommerceflow-generate-orders`
  - Status: enabled
  - Schedule: every 4 hours
  - Time zone: `America/Sao_Paulo`
  - Target: `POST /simulation/run-now?n=3000`
- `ecommerceflow-score-orders`
  - Status: enabled
  - Schedule: daily at 01:00
  - Time zone: `America/Sao_Paulo`
  - Target: Cloud Run Job `ecommerceflow-vertex-scoring`
  - The Scheduler target and `roles/run.invoker` permission were repaired on 2026-07-01 after the job had been blocked by permission errors.

Cloud Run Jobs:

- `ecommerceflow-vertex-scoring`
  - Status: ready
  - Current image tag checked during this update: `25ae362cb7b0d17a97e3115fb2b7d7d333fa42fc`
  - Mode: `SCORING_MODE=local`
  - It uses a saved sklearn model artifact from GCS/local files and does not call a persistent Vertex AI online endpoint by default.
- `ecommerceflow-crewai-training`
  - Status: ready
  - Exists as an on-demand training/orchestration job, not as the main daily scoring path.

Prediction status:

- Prediction table: `otimizador-cargas.commerce_gold.delay_risk_predictions`
- Current live prediction count from the API: `500`
- Latest live prediction timestamp from the API: `2026-06-24 01:03:21 UTC`
- In Sao Paulo time, that is `2026-06-23 22:03:21`, which explains why the dashboard appeared stuck on 23/06.
- Performance endpoint: `/predictions/performance`
- Current performance history: empty until the next successful scoring run writes the first monitoring row.
- Next expected daily scoring run after the repair: 2026-07-02 at 01:00 America/Sao_Paulo.

## What Is Implemented

- Synthetic e-commerce order generation.
- SQLite local operational database.
- FastAPI backend with endpoints for orders, metrics, simulation control, BigQuery backfill, prediction summary, latest predictions, and prediction performance.
- BigQuery ingestion for generated orders.
- BigQuery raw, silver, and gold SQL transformations.
- Delay-risk model training scripts and CrewAI orchestration code.
- Low-cost daily scoring job using a saved sklearn model artifact instead of a persistent online Vertex endpoint.
- BigQuery prediction output table.
- BigQuery prediction-performance monitoring table schema.
- Daily model-performance comparison logic:
  - accuracy
  - precision
  - recall
  - F1
  - ROC AUC
  - average precision
  - confusion-matrix counts
  - deltas versus the latest previous baseline
- React/Vite dashboard for operational metrics, prediction summary, latest predictions, and performance history.
- GitHub Actions CI/CD for frontend, backend, and scoring job deployment.
- Optional billing cap Cloud Run Function and setup script.

## Architecture

```text
Cloud Scheduler
  |-- every 4 hours --> FastAPI /simulation/run-now?n=3000
  |                     |
  |                     +--> synthetic orders
  |                     +--> SQLite local DB
  |                     +--> BigQuery commerce_raw.orders
  |
  +-- daily 01:00 --> Cloud Run Job ecommerceflow-vertex-scoring
                        |
                        +--> refresh commerce_silver.orders_cleaned
                        +--> refresh commerce_gold.delay_risk_features
                        +--> load saved sklearn model artifact
                        +--> score recent unscored orders
                        +--> write commerce_gold.delay_risk_predictions
                        +--> write commerce_gold.delay_risk_prediction_performance

React/Vite dashboard
  |
  +--> FastAPI backend
        |
        +--> operational order metrics
        +--> BigQuery prediction summary
        +--> BigQuery latest predictions
        +--> BigQuery performance history
```

## Local Quickstart

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
uvicorn backend.main:app --reload
```

Open API docs:

```text
http://localhost:8000/docs
```

Run frontend:

```bash
cd frontend
npm install
npm run dev
```

Open:

```text
http://localhost:5173
```

## Generate Orders Manually

```powershell
curl.exe -X POST "http://localhost:8000/orders/generate?n=3000"
```

Or use the simulation endpoint:

```powershell
curl.exe -X POST "http://localhost:8000/simulation/run-now?n=3000"
```

## Train A Delay-Risk Model Locally

After generating or loading orders:

```bash
python ml/train_delay_model.py
```

CrewAI orchestration is also available:

```bash
python agents/commerceflow_crew.py
```

Older notebook/static dashboard artifacts remain in `ml_outputs/`. The primary app dashboard is now the React/Vite frontend.

## BigQuery Setup

1. Create or select a GCP project.
2. Create a service account with BigQuery permissions.
3. Set `GOOGLE_APPLICATION_CREDENTIALS` in `.env`.
4. Run `sql/create_raw_tables.sql` in BigQuery.
5. Run or adapt:
   - `sql/silver_orders_cleaned.sql`
   - `sql/gold_delay_features.sql`
   - `sql/prediction_performance.sql`
6. Set:

```env
ORDERS_PER_BATCH=3000
SIMULATION_MINUTES=240
MAX_ORDERS_PER_BATCH=3000
BIGQUERY_INGESTION_ENABLED=true
BIGQUERY_INSERT_CHUNK_SIZE=3000
BIGQUERY_ORDERS_TABLE=your-gcp-project.commerce_raw.orders
PREDICTIONS_TABLE=your-gcp-project.commerce_gold.delay_risk_predictions
PREDICTION_PERFORMANCE_TABLE=your-gcp-project.commerce_gold.delay_risk_prediction_performance
```

To pause BigQuery ingestion without stopping the local simulator:

```env
BIGQUERY_INGESTION_ENABLED=false
```

Dashboard BigQuery queries are cached by default:

```env
BIGQUERY_PREDICTIONS_ENABLED=true
PREDICTIONS_CACHE_SECONDS=900
BIGQUERY_MAX_BYTES_BILLED=100000000
```

## Scoring And Cost Control

The production scoring job is named `ecommerceflow-vertex-scoring`, but the low-cost default is local model scoring:

```env
SCORING_MODE=local
SCORING_ENABLED=true
ALLOW_VERTEX_ENDPOINT=false
REFRESH_FEATURES_BEFORE_SCORING=true
BATCH_SIZE=300
DAILY_SCORING_LIMIT=300
SCORING_LOOKBACK_DAYS=7
PERFORMANCE_MONITORING_ENABLED=true
ACCURACY_DROP_ALERT_THRESHOLD=0.05
F1_DROP_ALERT_THRESHOLD=0.05
MODEL_GCS_URI=gs://commerceflow-ml-artifacts-otimizador-cargas/commerceflow/vertex_custom_model/v1
FEATURE_COLUMNS_GCS=gs://commerceflow-ml-artifacts-otimizador-cargas/commerceflow/vertex_custom_model/v1/feature_columns.json
```

`SCORING_MODE=vertex` is supported by the script, but online endpoint calls are blocked unless `ALLOW_VERTEX_ENDPOINT=true`. Keep this off unless you intentionally want endpoint costs.

## Useful API Endpoints

Local or Cloud Run backend:

```text
GET  /health
GET  /orders?limit=100&offset=0
GET  /metrics
POST /orders/generate?n=3000
POST /simulation/run-now?n=3000
GET  /simulation/status
GET  /predictions/summary
GET  /predictions/latest?limit=100
GET  /predictions/performance?limit=14
```

## CI/CD

GitHub Actions workflows:

- `.github/workflows/ci.yml`
  - Builds the React/Vite frontend.
  - Compiles Python sources.
  - Builds backend and scoring Docker images without pushing.
- `.github/workflows/cd.yml`
  - Deploys the frontend to Vercel.
  - Deploys the FastAPI backend to Cloud Run.
  - Creates or updates the Cloud Run scoring job.
  - Creates or updates the daily Cloud Scheduler scoring trigger.
  - Grants `roles/run.invoker` to the runtime service account on the scoring job.

Required GitHub repository secrets:

```text
VERCEL_TOKEN
VERCEL_ORG_ID
VERCEL_PROJECT_ID
GCP_WORKLOAD_IDENTITY_PROVIDER
GCP_SERVICE_ACCOUNT
```

The deployer service account needs:

```text
roles/run.admin
roles/cloudscheduler.admin
roles/cloudbuild.builds.editor
roles/artifactregistry.writer
roles/iam.serviceAccountUser on commerceflow-runner@otimizador-cargas.iam.gserviceaccount.com
```

## Disable Online Vertex Endpoint Costs

If an online Vertex endpoint has a deployed model, inspect it:

```powershell
.\scripts\vertex_disable_online_endpoint.ps1 `
  -ProjectId otimizador-cargas `
  -Region us-central1 `
  -EndpointId 2085985213879418880
```

Then undeploy the model:

```powershell
.\scripts\vertex_disable_online_endpoint.ps1 `
  -ProjectId otimizador-cargas `
  -Region us-central1 `
  -EndpointId 2085985213879418880 `
  -Apply
```

## Monthly Billing Cap

Google Cloud budgets send alerts; they do not stop billing by themselves. To stop spend automatically, deploy the budget cap function in `billing_cap_function/` and wire your monthly budget to Pub/Sub.

Dry run:

```powershell
.\scripts\setup_gcp_budget_cap.ps1 `
  -ProjectId otimizador-cargas `
  -BillingAccountId YOUR_BILLING_ACCOUNT_ID `
  -BudgetAmount 30 `
  -ThresholdRatio 1.0
```

Apply:

```powershell
.\scripts\setup_gcp_budget_cap.ps1 `
  -ProjectId otimizador-cargas `
  -BillingAccountId YOUR_BILLING_ACCOUNT_ID `
  -BudgetAmount 30 `
  -ThresholdRatio 1.0 `
  -Apply
```

Because budget notifications can be delayed, use a lower amount such as `25` or a lower threshold such as `0.9` if the real monthly ceiling is `30`.

## Current Gaps

- The performance monitoring endpoint is deployed, but it has no rows yet. It should populate after the next successful daily scoring run.
- The main production scoring path uses a saved sklearn artifact, not a live Vertex AI endpoint.
- The CrewAI training job exists, but retraining is still manual/on-demand rather than an automated daily production retrain.
- The frontend is deployed through Vercel in CD, but the canonical Vercel URL is managed outside this README.
- The dashboard depends on BigQuery availability and cached API responses, so prediction panels can lag behind newly inserted rows by `PREDICTIONS_CACHE_SECONDS`.
