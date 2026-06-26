import pickle
import joblib
import pandas as pd
from pathlib import Path

class CommerceFlowVertexWrapper:
    def __init__(self, model, feature_columns):
        self.model = model
        self.feature_columns = feature_columns

    def _to_dataframe(self, X):
        # Vertex may send list[dict] or list[list]
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        elif isinstance(X, list) and len(X) > 0 and isinstance(X[0], dict):
            df = pd.DataFrame(X)
        else:
            df = pd.DataFrame(X, columns=self.feature_columns)

        # Guarantee correct order and missing column handling
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = None

        return df[self.feature_columns]

    def predict(self, X):
        df = self._to_dataframe(X)
        return self.model.predict(df).tolist()

    def predict_proba(self, X):
        df = self._to_dataframe(X)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(df).tolist()
        preds = self.model.predict(df)
        return [[1 - int(p), int(p)] for p in preds]


# Load original artifact
with open("cloud_ml_outputs/model.pkl", "rb") as f:
    obj = pickle.load(f)

if isinstance(obj, dict):
    model = None
    for key in ["model", "best_model", "pipeline", "best_pipeline", "estimator", "trained_model", "final_model"]:
        if key in obj and hasattr(obj[key], "predict"):
            model = obj[key]
            print("Selected model key:", key)
            break
    if model is None:
        raise RuntimeError("No valid model found inside model.pkl")
else:
    model = obj

# Load exact ML-ready feature columns
df = pd.read_parquet("cloud_ml_outputs/df3_ml_ready.parquet")

if "delay_risk_label" in df.columns:
    feature_columns = list(df.drop(columns=["delay_risk_label"]).columns)
else:
    feature_columns = list(df.columns)

print("Feature columns:", len(feature_columns))

wrapper = CommerceFlowVertexWrapper(model, feature_columns)

# Quick local smoke test
sample = df.drop(columns=["delay_risk_label"], errors="ignore").head(3)
print("Local wrapper prediction:", wrapper.predict(sample))

dst_dir = Path("vertex_model_v3")
dst_dir.mkdir(exist_ok=True)
joblib.dump(wrapper, dst_dir / "model.joblib")

print("Saved wrapped Vertex model to vertex_model_v3/model.joblib")
