# src/key_utils.py
import os
from pathlib import Path
import cloudpickle as cp

def save_obj(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        cp.dump(obj, f)
    print(f"Saved {path}")

def load_obj(path):
    with open(path, "rb") as f:
        return cp.load(f)

def party_dir(base_dir, party_id):
    d = Path(base_dir) / f"party_{party_id}"
    d.mkdir(parents=True, exist_ok=True)
    return d
