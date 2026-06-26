# CommerceFlow CrewAI Cloud Run Job patch

Files:
- `commerceflow_delay_risk_crewai_v6.py`: supports `--source bigquery` and optional GCS artifact upload.
- `Dockerfile.crewai`: Dockerfile for a Cloud Run Job.
- `requirements_crewai.txt`: requirements with BigQuery + Cloud Storage.

Recommended local test:

```powershell
python commerceflow_delay_risk_crewai_v6.py --source bigquery --bq-table otimizador-cargas.commerce_gold.delay_risk_features --max-rows 100000 --direct --no-git
```

Recommended cloud job command:

```bash
python commerceflow_delay_risk_crewai_v6.py \
  --source bigquery \
  --bq-table otimizador-cargas.commerce_gold.delay_risk_features \
  --max-rows 500000 \
  --direct \
  --no-git \
  --gcs-bucket YOUR_BUCKET \
  --gcs-prefix commerceflow/ml_runs/latest
```
