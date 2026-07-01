# CommerceFlow AI — Real-Time E-commerce Order Simulator + BigQuery + ML

CommerceFlow AI is an end-to-end portfolio project that simulates a real-time e-commerce operation and prepares the data architecture for delay-risk prediction.

## Goal

Build a simulated real-time environment where:

- A website displays live e-commerce orders.
- The backend generates **3,000 new orders every 4 hours**.
- Each order includes customer, product, value, logistics and distance information.
- The backend can ingest the orders into a **BigQuery raw table**.
- A machine learning pipeline predicts the risk of delivery delay.
- A dashboard monitors order stats, delay risk and business insights.

## Architecture

```text
React/Vite Website
      │
      ▼
FastAPI Backend
      ├── Synthetic order generator
      ├── SQLite local operational database
      ├── APScheduler: 3,000 orders every 4 hours
      └── BigQuery raw ingestion
              │
              ▼
BigQuery raw/silver/gold tables
              │
              ▼
CrewAI orchestration
Data Quality → Feature Engineering → ML Training → Dashboard/GitOps
              │
              ▼
Delay risk model + Streamlit monitoring dashboard
```

## Local quickstart

```bash
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

## Generate orders manually

```bash
curl -X POST http://localhost:8000/orders/generate \
  -H "Content-Type: application/json" \
  -d "{\"n\": 3000}"
```

## Train first delay risk model

After generating some orders:

```bash
python ml/train_delay_model.py
```

## Run dashboard

```bash
streamlit run dashboard/streamlit_app.py
```

## Run CrewAI orchestration

```bash
python agents/commerceflow_crew.py
```

## BigQuery setup

1. Create a GCP project.
2. Create a service account with BigQuery permissions.
3. Set `GOOGLE_APPLICATION_CREDENTIALS` in `.env`.
4. Run `sql/create_raw_tables.sql` in BigQuery.
5. Set:

```env
ORDERS_PER_BATCH=3000
SIMULATION_MINUTES=240
MAX_ORDERS_PER_BATCH=3000
BIGQUERY_INGESTION_ENABLED=true
BIGQUERY_INSERT_CHUNK_SIZE=3000
BIGQUERY_ORDERS_TABLE=your-gcp-project.commerce_raw.orders
```

The backend will insert every generated batch into BigQuery if this variable is configured.

To pause BigQuery ingestion without stopping the local simulator, set:

```env
BIGQUERY_INGESTION_ENABLED=false
```

Dashboard BigQuery queries are cached by default. Tune these values to control query spend:

```env
BIGQUERY_PREDICTIONS_ENABLED=false
PREDICTIONS_CACHE_SECONDS=900
BIGQUERY_MAX_BYTES_BILLED=100000000
```

## Vertex AI cost control

For a very small budget, keep Vertex online endpoints turned off and use local scoring in the Cloud Run Job:

```env
SCORING_MODE=local
SCORING_ENABLED=true
ALLOW_VERTEX_ENDPOINT=false
REFRESH_FEATURES_BEFORE_SCORING=true
BATCH_SIZE=50
DAILY_SCORING_LIMIT=300
SCORING_LOOKBACK_DAYS=7
PERFORMANCE_MONITORING_ENABLED=true
PREDICTION_PERFORMANCE_TABLE=your-gcp-project.commerce_gold.delay_risk_prediction_performance
```

`score_recent_orders_vertex.py` defaults to `SCORING_MODE=local`, which loads `vertex_custom_model/model.joblib` or `MODEL_GCS_URI` and does not call a persistent Vertex AI endpoint.

The deployed low-cost schedule runs `ecommerceflow-vertex-scoring` once per day at 01:00 America/Sao_Paulo through the `ecommerceflow-score-orders` Cloud Scheduler job. It refreshes `commerce_silver.orders_cleaned` and `commerce_gold.delay_risk_features`, then scores a capped sample of recent unscored orders into `commerce_gold.delay_risk_predictions`. Each successful scoring run also writes accuracy, precision, recall, F1, ROC AUC, average precision and metric deltas into `commerce_gold.delay_risk_prediction_performance`. If the performance table has no prior rows, the first run compares itself to the latest previous prediction batch, such as the 2026-06-23 batch.

## CI/CD

This repo includes GitHub Actions workflows:

- `.github/workflows/ci.yml`: runs on pull requests and pushes to `main`.
  - Builds the React/Vite frontend.
  - Compiles Python sources.
  - Builds backend and scoring Docker images without pushing.
- `.github/workflows/cd.yml`: runs on pushes to `main` and can also be started manually.
  - Deploys the frontend to Vercel.
  - Builds and deploys the FastAPI backend to Cloud Run.
  - Builds and updates the daily scoring Cloud Run Job and Cloud Scheduler trigger.

Required GitHub repository secrets:

```text
VERCEL_TOKEN
VERCEL_ORG_ID
VERCEL_PROJECT_ID
GCP_WORKLOAD_IDENTITY_PROVIDER
GCP_SERVICE_ACCOUNT
```

`GCP_SERVICE_ACCOUNT` should be a deployer service account, not a local credential file. It needs permissions to run Cloud Build, deploy Cloud Run services/jobs, and act as the runtime service account:

```text
roles/run.admin
roles/cloudscheduler.admin
roles/cloudbuild.builds.editor
roles/artifactregistry.writer
roles/iam.serviceAccountUser on commerceflow-runner@otimizador-cargas.iam.gserviceaccount.com
```

The deployed runtime service account remains `commerceflow-runner@otimizador-cargas.iam.gserviceaccount.com`. The CD workflow grants this account `roles/run.invoker` on the scoring job so Cloud Scheduler can execute the daily run.

If Vercel Git integration is also enabled, keep only one production deploy path to avoid duplicate frontend deployments.

If an online Vertex endpoint is already deployed, first inspect it:

```powershell
.\scripts\vertex_disable_online_endpoint.ps1 `
  -ProjectId otimizador-cargas `
  -Region us-central1 `
  -EndpointId 2085985213879418880
```

Then undeploy the model from the endpoint:

```powershell
.\scripts\vertex_disable_online_endpoint.ps1 `
  -ProjectId otimizador-cargas `
  -Region us-central1 `
  -EndpointId 2085985213879418880 `
  -Apply
```

## Monthly billing cap

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

This deploys a Cloud Run Function triggered by budget Pub/Sub messages. When current cost reaches the threshold, it removes Cloud Billing from the project. Because budget notifications can be delayed, use a lower amount such as `25` or a lower threshold such as `0.9` if the real monthly ceiling is `30`.

## Next phases

- Add BigQuery silver/gold transformations.
- Train model from BigQuery feature table.
- Deploy model to Vertex AI.
- Add prediction API endpoint.
- Add Streamlit/HTML dashboard with real model risk scores.
- Add automated CrewAI data science cycle.
