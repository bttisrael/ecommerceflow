import pickle
from pathlib import Path

p = Path("cloud_ml_outputs/model.pkl")

with open(p, "rb") as f:
    obj = pickle.load(f)

print("Object type:", type(obj))

if isinstance(obj, dict):
    print("Dictionary keys:")
    for k, v in obj.items():
        print("-", k, "=>", type(v))
else:
    print("Object has predict:", hasattr(obj, "predict"))
    print("Object has predict_proba:", hasattr(obj, "predict_proba"))
