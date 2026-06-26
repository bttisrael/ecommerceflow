import pandas as pd
import json
from pathlib import Path

df = pd.read_parquet("cloud_ml_outputs/df3_ml_ready.parquet")

X = df.drop(columns=["delay_risk_label"], errors="ignore")
feature_columns = list(X.columns)

Path("vertex_custom_model").mkdir(exist_ok=True)
Path("vertex_custom_model/feature_columns.json").write_text(
    json.dumps(feature_columns, indent=2),
    encoding="utf-8"
)

print("Saved feature_columns.json")
print("Columns:", len(feature_columns))
