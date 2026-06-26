import pickle
import joblib
from pathlib import Path

src = Path("cloud_ml_outputs/model.pkl")
dst_dir = Path("vertex_model")
dst_dir.mkdir(exist_ok=True)
dst = dst_dir / "model.joblib"

with open(src, "rb") as f:
    model = pickle.load(f)

joblib.dump(model, dst)

print(f"Saved Vertex artifact to: {dst}")
