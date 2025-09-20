"""
Generate reproducible random unit-normalized vectors for the
OpenFHE CKKS encrypted max-cosine prototype.

Outputs:
  - data/db_vectors.npz    -> contains 'db' (1000 x 512) float32
  - data/query_vector.npz  -> contains 'q' (512,) float32
  - data/metadata.json     -> small JSON with shapes/seed

Usage:
  python src/generate_vectors.py --n 1000 --dim 512 --seed 42 --out-dir data
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from utils import unit_normalize_rows

def generate_random_unit_vectors(n: int, dim: int, seed: int = None) -> np.ndarray:
    """
    Generate n random vectors of dimension dim, unit L2 normalized.
    Distribution: standard normal (Gaussian) then normalize per vector.
    Returns float32 array of shape (n, dim).
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, dim)).astype(np.float32)
    X = unit_normalize_rows(X)
    return X

def generate_random_unit_vector(dim: int, seed: int = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(dim,)).astype(np.float32)
    x = unit_normalize_rows(x)
    return x

def save_npz(output_path: Path, array: np.ndarray, name: str):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **{name: array})
    print(f"Saved {name} -> {output_path} (shape={array.shape}, dtype={array.dtype})")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000, help="Number of DB vectors")
    parser.add_argument("--dim", type=int, default=512, help="Vector dimension")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (reproducible)")
    parser.add_argument("--out-dir", type=str, default="data", help="Output directory")
    parser.add_argument("--query-seed", type=int, default=None, help="Optional separate seed for query")
    parser.add_argument("--save-batch-size", type=int, default=250, help="Optional: split DB into .npz shards (0 = single file)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.n} vectors of dim {args.dim} with seed={args.seed}")
    db = generate_random_unit_vectors(args.n, args.dim, seed=args.seed)

    q_seed = args.query_seed if args.query_seed is not None else (args.seed + 1)
    print(f"Generating query vector with seed={q_seed}")
    q = generate_random_unit_vector(args.dim, seed=q_seed)

    # Basic sanity checks
    db_norms = np.linalg.norm(db, axis=1)
    q_norm = np.linalg.norm(q)
    assert db.shape == (args.n, args.dim)
    assert np.allclose(db_norms, 1.0, atol=1e-6), "DB rows not unit-normalized"
    assert np.isclose(q_norm, 1.0, atol=1e-6), "Query not unit-normalized"

    # Save: either sharded or single file
    shard_size = args.save_batch_size
    if shard_size and shard_size > 0 and shard_size < args.n:
        num_shards = (args.n + shard_size - 1) // shard_size
        print(f"Saving DB in {num_shards} shards of up to {shard_size} vectors each.")
        for idx, start in enumerate(range(0, args.n, shard_size)):
            end = min(start + shard_size, args.n)
            shard = db[start:end]
            shard_path = out_dir / f"db_vectors_shard_{idx:03d}.npz"
            save_npz(shard_path, shard, name="db")
    else:
        db_path = out_dir / "db_vectors.npz"
        save_npz(db_path, db, name="db")

    q_path = out_dir / "query_vector.npz"
    save_npz(q_path, q, name="q")

    meta = {
        "n": args.n,
        "dim": args.dim,
        "seed_db": args.seed,
        "seed_query": q_seed,
        "db_files": [str(p.name) for p in sorted(out_dir.glob("db_vectors*.npz"))],
        "query_file": q_path.name,
    }
    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote metadata -> {meta_path}")

if __name__ == "__main__":
    main()
