import pandas as pd
import json
from pathlib import Path

df = pd.read_parquet("cloud_ml_outputs/df3_ml_ready.parquet")

if "delay_risk_label" in df.columns:
    X = df.drop(columns=["delay_risk_label"])
else:
    X = df.copy()

sample = X.head(3).copy()
sample = sample.where(pd.notnull(sample), None)

payload = {
    "instances": sample.to_dict(orient="records")
}

Path("vertex_sample_request.json").write_text(
    json.dumps(payload, indent=2, default=str),
    encoding="utf-8"
)

print("Saved vertex_sample_request.json")
print("Rows:", len(sample))
print("Columns:", len(sample.columns))
print(sample.head())
