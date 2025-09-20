#!/usr/bin/env python3
"""
tckks_keygen.py

Multiparty / Threshold CKKS key generation (prototype).
Generates per-party secret shares and collective public/eval keys.

Usage:
  python src/tckks_keygen.py --n-parties 3 --out-dir keys --seed 42

Notes:
  - Requires OpenFHE Python bindings (import openfhe).
  - This script follows the "party0 KeyGen; party_i MultipartyKeyGen(pk_prev)" pattern
    seen in OpenFHE threshold examples. If any function name differs in your version,
    consult OpenFHE Python examples: https://github.com/openfheorg/openfhe-python
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    # OpenFHE python binding (canonical import path used in examples)
    from openfhe import *
except Exception as e:
    print("Failed to import `openfhe` python binding. Make sure openfhe-python is installed.")
    print("You can find examples at https://github.com/openfheorg/openfhe-python and docs at https://openfhe-development.readthedocs.io")
    raise

from key_utils import save_obj, party_dir

def make_cc(params=None):
    """
    Create a CKKS CryptoContext for threshold use.
    We keep the parameters conservative for a prototype (you will tune ring dim & scale later).
    NOTE: function/class names vary between versions; adapt from OpenFHE examples if needed.
    """
    # Typical pattern from examples:
    # params = CCParamsCKKSRNS()   # if the binding exposes a params class
    # params.SetRingDimension(16384)
    # params.SetScalingModSize(50)
    # cc = GenCryptoContext(params)
    #
    # To be resilient across bindings, try a small set of canonical constructors:
    try:
        # Newer bindings: CCParamsCKKSRNS + GenCryptoContext
        params = CCParamsCKKSRNS()
        # Prototype defaults (tune later):
        try:
            params.SetRingDim(16384)
            params.SetScalingModSize(50)
            params.SetNumLargeDigits(3)
        except Exception:
            # alternate names
            params.SetRingDimension(16384)
            params.SetScalingModSize(50)
        cc = GenCryptoContext(params)
        print("Created CryptoContext (CCParamsCKKSRNS -> GenCryptoContext).")
        return cc
    except Exception:
        pass

    # Fallback: high-level helper (older bindings)
    try:
        cc = CryptoContextFactory.GenCryptoContextCKKS()
        print("Created CryptoContext using CryptoContextFactory.GenCryptoContextCKKS().")
        return cc
    except Exception as e:
        print("Could not create CryptoContext automatically. Please adapt make_cc() to your binding.")
        raise

def party0_keygen(cc):
    """
    Party 0 generates initial keypair.
    Returns (kp0, pk0, sk0) or (public_key, secret_key) depending on binding.
    """
    print("Party 0: running KeyGen()")
    kp0 = cc.KeyGen()
    # most bindings return a KeyPair object or tuple; attempt to extract pk/sk
    try:
        pk0 = kp0.public_key
        sk0 = kp0.secret_key
    except Exception:
        # maybe returned as tuple (pk, sk)
        try:
            pk0, sk0 = kp0
        except Exception:
            pk0 = kp0
            sk0 = None
    return kp0, pk0, sk0

def party_i_multiparty(cc, prev_pub, party_id):
    """
    Party i runs MultipartyKeyGen using prev_pub and generates its secret share and updated public key.
    Return (local_kp, local_pub, local_sk_share).
    """
    print(f"Party {party_id}: running MultipartyKeyGen(...)")
    # API is frequently: cc.MultipartyKeyGen(prev_pub) or cc.MultiKeyGen(prev_pub)
    try:
        kp_i = cc.MultipartyKeyGen(prev_pub)
    except Exception:
        kp_i = cc.MultiPartyKeyGen(prev_pub) if hasattr(cc, "MultiPartyKeyGen") else None

    if kp_i is None:
        raise RuntimeError("MultipartyKeyGen API not found in this binding. See OpenFHE examples for the exact call.")

    # extract public/secret if possible
    try:
        pk_i = kp_i.public_key
        sk_i = kp_i.secret_key
    except Exception:
        try:
            pk_i, sk_i = kp_i
        except Exception:
            pk_i = kp_i
            sk_i = None

    return kp_i, pk_i, sk_i

def generate_eval_keys_collective(cc, party_shares, collective_pub, out_dir):
    """
    Each party uses its secret share to generate their contribution to eval keys.
    Then we aggregate to obtain collective eval keys:
      - EvalMultKey
      - Rotation/EvalAtIndex Keys (for dot products reductions)
    The exact API differs by binding; we follow the high-level pattern from OpenFHE examples.
    """
    print("Generating / aggregating collective evaluation keys (per OpenFHE multiparty pattern).")
    # For each party: call cc.EvalMultKeyGen(sk_share) or cc.EvalMultKeyGen(kp_share)
    # and cc.EvalAtIndexKeyGen(sk_share, indices)
    # Then aggregate with cc.MultiEvalMultKeyGen or cc.AggregateEvalMultKeys(...)
    # We'll implement a generic pattern and serialize intermediate objects.
    eval_mult_contribs = []
    eval_rot_contribs = []
    for pid, share in party_shares.items():
        print(f"Party {pid}: generating local eval-key contributions")
        # typical calls:
        try:
            local_mult = cc.EvalMultKeyGen(share)
            local_rot  = cc.EvalAtIndexKeyGen(share, [1, -1])  # example rotations; adjust later
        except Exception:
            # some bindings require keypair rather than sk only
            local_mult = cc.EvalMultKeyGen(share)
            local_rot  = cc.EvalAtIndexKeyGen(share, [1, -1])

        eval_mult_contribs.append(local_mult)
        eval_rot_contribs.append(local_rot)

    # Combine/aggregate them into collective keys
    # Typical API: cc.MultiEvalMultKeyGen(eval_mult_contribs) or cc.AggregateEvalMultKeys(...)
    try:
        collective_eval_mult = cc.MultiEvalMultKeyGen(eval_mult_contribs)
        collective_eval_rot  = cc.MultiEvalAtIndexKeyGen(eval_rot_contribs)
    except Exception:
        # fallback: some examples show custom aggregation; if your binding doesn't have MultiEval...
        # For prototype, store per-party contributions and call aggregation in C++ example.
        print("Binding does not expose MultiEval* aggregation helper; saving per-party contributions for out-of-band aggregation.")
        collective_eval_mult = eval_mult_contribs
        collective_eval_rot = eval_rot_contribs

    # Serialize collective eval keys
    save_obj(collective_eval_mult, Path(out_dir) / "collective" / "eval_mult_key.pkl")
    save_obj(collective_eval_rot, Path(out_dir) / "collective" / "eval_rot_key.pkl")
    print("Saved collective eval keys (or per-party contributions) to disk.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-parties", type=int, default=3, help="Number of parties (>=2)")
    parser.add_argument("--out-dir", type=str, default="keys", help="Directory to write key shares and public keys")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed (if any randomness is used in context)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Make crypto context
    cc = make_cc()

    # 2) Party 0: KeyGen() -> initial kp0
    kp0, pk0, sk0 = party0_keygen(cc)
    # Persist party0 artifacts locally
    p0dir = party_dir(out_dir, 0)
    save_obj(pk0, p0dir / "pubkey.pkl")
    save_obj(sk0, p0dir / "sk_share.pkl")

    collective_pub = pk0
    party_shares = {0: sk0}

    # 3) For parties 1..n-1 run MultipartyKeyGen sequentially (simple ordered protocol)
    prev_pub = pk0
    for pid in range(1, args.n_parties):
        kp_i, pk_i, sk_i = party_i_multiparty(cc, prev_pub, pid)
        pdir = party_dir(out_dir, pid)
        save_obj(pk_i, pdir / "pubkey.pkl")
        save_obj(sk_i, pdir / "sk_share.pkl")
        # update prev
        prev_pub = pk_i
        collective_pub = pk_i
        party_shares[pid] = sk_i

    # 4) Save the collective public key for the server
    save_obj(collective_pub, out_dir + "/collective/public_key.pkl")
    print(f"Multi-party keygen complete. Collective public key saved to {out_dir}/collective/public_key.pkl")
    # 5) Generate/aggregate evaluation keys needed for homomorphic ops (rotations, mult)
    try:
        generate_eval_keys_collective(cc, party_shares, collective_pub, out_dir)
    except Exception as e:
        print("Failed to auto-generate collective eval keys. See OpenFHE examples for distributed generation.")
        print("Error:", e)
        # still save per-party shares so user can run distributed evalkeygen using C++ example or adapt code.
        for pid, sk in party_shares.items():
            save_obj(sk, Path(out_dir) / f"party_{pid}" / "sk_share.pkl")

    # 6) Write a small metadata file
    meta = {
        "n_parties": args.n_parties,
        "out_dir": str(out_dir),
        "collective_public_key": "keys/collective/public_key.pkl",
        "party_dirs": [str(p) for p in sorted(Path(out_dir).glob("party_*"))],
    }
    with open(Path(out_dir) / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("Wrote metadata.json")

if __name__ == "__main__":
    main()
