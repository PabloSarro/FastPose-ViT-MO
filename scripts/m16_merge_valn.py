"""M16: pool the per-object validation runs of single-object models into one score.

Each EVAL_DIR is a bop_evaluate.py output for one object (scripts/m16_single_valn.sh). The pooled
recalls are the instance-weighted means of the per-object recalls, i.e. what one evaluation of all
objects together would give; score = (AR_MSSD + AR_MSPD) / 2 as scripts/val_score.py.

Usage: .venv/bin/python scripts/m16_merge_valn.py [--split val_real] EVAL_DIR [EVAL_DIR ...]
"""
import argparse
import json

import numpy as np

KEYS = ("recall_MSSD", "recall_MSPD", "ADD@0.1d", "trans_mm_mean", "rot_deg_mean")


def pool(summaries: list, split: str) -> dict:
    rows = [s[split]["mean over instances"] for s in summaries]
    n = np.array([r["n"] for r in rows], dtype=np.float64)
    out = {k: float(np.average([r[k] for r in rows], weights=n)) for k in KEYS}
    out["n"] = int(n.sum())
    out["score"] = (out["recall_MSSD"] + out["recall_MSPD"]) / 2
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("eval_dirs", nargs="+")
    parser.add_argument("--split", default="val_real")
    args = parser.parse_args()
    summaries = []
    for d in args.eval_dirs:
        with open(f"{d}/summary.json") as f:
            summaries.append(json.load(f))
    p = pool(summaries, args.split)
    print(f"{len(summaries)} models, {args.split}: n={p['n']} score={p['score']:.4f} "
          f"AR_MSSD={p['recall_MSSD']:.4f} AR_MSPD={p['recall_MSPD']:.4f} "
          f"ADD@0.1d={p['ADD@0.1d']:.4f} trans={p['trans_mm_mean']:.2f}mm "
          f"rot={p['rot_deg_mean']:.2f}deg")


if __name__ == "__main__":
    main()
