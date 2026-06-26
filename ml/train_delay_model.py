from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "artifacts"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "delay_risk_model.pkl"
METRICS_PATH = MODEL_DIR / "model_metrics.json"


def load_training_data() -> pd.DataFrame:
    # First local MVP: train from SQLite-generated orders.
    # Later, set TRAINING_SOURCE=bigquery and implement BigQuery read.
    source = os.getenv("TRAINING_SOURCE", "sqlite")
    if source == "bigquery":
        from google.cloud import bigquery
        table = os.getenv("BIGQUERY_FEATURE_TABLE")
        if not table:
            raise RuntimeError("BIGQUERY_FEATURE_TABLE is required when TRAINING_SOURCE=bigquery")
        client = bigquery.Client()
        return client.query(f"SELECT * FROM `{table}`").to_dataframe()

    from backend.database import export_orders_dataframe
    df = export_orders_dataframe()
    if df.empty:
        raise RuntimeError("No local orders found. Start backend and generate orders first.")
    return df


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    df["order_timestamp"] = pd.to_datetime(df["order_timestamp"], errors="coerce")
    df["day_of_week"] = df["order_timestamp"].dt.dayofweek.fillna(0).astype(int)
    df["order_hour"] = df["order_timestamp"].dt.hour.fillna(0).astype(int)
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    feature_cols = [
        "state", "city", "product_category", "order_value", "order_weight_kg", "distance_km",
        "carrier", "dc_id", "promised_delivery_days", "estimated_delivery_days", "day_of_week", "order_hour", "is_weekend",
    ]
    target_col = "delay_risk_label"
    X = df[feature_cols]
    y = df[target_col].astype(int)
    return X, y


def train() -> dict:
    df = load_training_data()
    X, y = build_features(df)
    if y.nunique() < 2:
        raise RuntimeError("Target has only one class. Generate more orders before training.")

    cat_cols = X.select_dtypes(include="object").columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced", n_jobs=2),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results = []
    fitted = {}
    for name, model in models.items():
        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe.named_steps["model"], "predict_proba") else pred
        metrics = {
            "model": name,
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, proba),
        }
        results.append(metrics)
        fitted[name] = pipe

    # In delay risk, prioritize recall and then F1.
    best = sorted(results, key=lambda r: (r["recall"], r["f1"], r["roc_auc"]), reverse=True)[0]
    best_pipe = fitted[best["model"]]

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_pipe, f)
    METRICS_PATH.write_text(json.dumps({"best_model": best, "all_models": results}, indent=2), encoding="utf-8")
    return {"best_model": best, "all_models": results, "model_path": str(MODEL_PATH)}


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
