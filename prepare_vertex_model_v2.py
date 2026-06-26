import pickle
import joblib
from pathlib import Path

src = Path("cloud_ml_outputs/model.pkl")
dst_dir = Path("vertex_model_v2")
dst_dir.mkdir(exist_ok=True)
dst = dst_dir / "model.joblib"

with open(src, "rb") as f:
    obj = pickle.load(f)

if isinstance(obj, dict):
    candidate_keys = [
        "model",
        "best_model",
        "pipeline",
        "best_pipeline",
        "estimator",
        "trained_model",
        "final_model",
    ]

    model = None
    selected_key = None

    for key in candidate_keys:
        if key in obj and hasattr(obj[key], "predict"):
            model = obj[key]
            selected_key = key
            break

    if model is None:
        print("Available keys and types:")
        for k, v in obj.items():
            print(k, type(v), "has_predict=", hasattr(v, "predict"))
        raise RuntimeError("Could not find a model object with .predict() inside model.pkl")

    print(f"Selected model key: {selected_key}")
else:
    model = obj
    print("model.pkl is already a model object")

print("Final model type:", type(model))
print("Has predict:", hasattr(model, "predict"))
print("Has predict_proba:", hasattr(model, "predict_proba"))

joblib.dump(model, dst)
print(f"Saved corrected Vertex model artifact: {dst}")
