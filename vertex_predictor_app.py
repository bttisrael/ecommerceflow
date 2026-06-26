import json
import os
import tempfile
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from google.cloud import storage

app = FastAPI()

MODEL = None
FEATURE_COLUMNS = None


def download_gcs_folder(gcs_uri: str, local_dir: str) -> None:
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"Invalid GCS URI: {gcs_uri}")

    path = gcs_uri.replace("gs://", "")
    bucket_name, prefix = path.split("/", 1)

    client = storage.Client()
    blobs = client.list_blobs(bucket_name, prefix=prefix)

    for blob in blobs:
        if blob.name.endswith("/"):
            continue

        rel_path = blob.name.replace(prefix.rstrip("/") + "/", "")
        local_path = Path(local_dir) / rel_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local_path))


def patch_sklearn_compatibility(model):
    """
    Compatibility patch for sklearn models serialized in a different version.
    Some LogisticRegression versions removed/changed multi_class handling.
    Older serving runtimes may still expect model.multi_class.
    """
    try:
        # Pipeline case
        if hasattr(model, "steps"):
            for _, step in model.steps:
                if step.__class__.__name__ == "LogisticRegression":
                    if not hasattr(step, "multi_class"):
                        step.multi_class = "auto"
                        print("Patched LogisticRegression.multi_class = auto")
        else:
            if model.__class__.__name__ == "LogisticRegression":
                if not hasattr(model, "multi_class"):
                    model.multi_class = "auto"
                    print("Patched LogisticRegression.multi_class = auto")
    except Exception as exc:
        print("Compatibility patch warning:", exc)

    return model


def load_model():
    global MODEL, FEATURE_COLUMNS

    if MODEL is not None:
        return

    storage_uri = os.getenv("AIP_STORAGE_URI") or os.getenv("MODEL_GCS_URI")
    if not storage_uri:
        raise RuntimeError("AIP_STORAGE_URI or MODEL_GCS_URI is not set.")

    local_dir = tempfile.mkdtemp()
    download_gcs_folder(storage_uri, local_dir)

    model_path = Path(local_dir) / "model.joblib"
    features_path = Path(local_dir) / "feature_columns.json"

    MODEL = joblib.load(model_path)
    MODEL = patch_sklearn_compatibility(MODEL)

    FEATURE_COLUMNS = json.loads(features_path.read_text(encoding="utf-8"))

    print(f"Loaded model from {model_path}")
    print(f"Loaded {len(FEATURE_COLUMNS)} feature columns")


@app.get("/health")
def health():
    load_model()
    return {
        "status": "ok",
        "features": len(FEATURE_COLUMNS),
        "model_type": str(type(MODEL)),
    }


@app.post("/predict")
async def predict(request: Request):
    try:
        load_model()

        body = await request.json()
        instances = body.get("instances", [])

        if not isinstance(instances, list):
            return {"error": "instances must be a list"}

        df = pd.DataFrame(instances)

        for col in FEATURE_COLUMNS:
            if col not in df.columns:
                df[col] = None

        df = df[FEATURE_COLUMNS]

        # Preferred: probability prediction
        try:
            if hasattr(MODEL, "predict_proba"):
                probabilities = MODEL.predict_proba(df)[:, 1]
            else:
                raise AttributeError("MODEL has no predict_proba")
        except Exception as proba_error:
            print("predict_proba failed, falling back to predict:", repr(proba_error))
            preds = MODEL.predict(df)
            probabilities = np.asarray(preds, dtype=float)

        predictions = (probabilities >= 0.5).astype(int)

        output = []
        for prob, pred in zip(probabilities, predictions):
            prob = float(prob)
            pred = int(pred)

            if prob >= 0.70:
                risk_band = "high"
            elif prob >= 0.40:
                risk_band = "medium"
            else:
                risk_band = "low"

            output.append({
                "delay_probability": prob,
                "delay_prediction": pred,
                "risk_band": risk_band
            })

        return {"predictions": output}

    except Exception as exc:
        print("PREDICT_ERROR:", repr(exc))
        print(traceback.format_exc())
        return {
            "error": str(exc),
            "traceback": traceback.format_exc()
        }
