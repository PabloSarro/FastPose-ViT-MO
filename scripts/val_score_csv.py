"""Validation score of a BOP results CSV against the GT of a split (e.g. an ensemble of two runs).

Same score as scripts/val_score.py: (AR_MSSD + AR_MSPD) / 2 with the own recalls, mean over
instances. The CSV must hold one estimate per (scene, image, object) of the split, as the CSVs of
src/bop_evaluate.py on val_real do.

Usage (repo root):
  PYTHONPATH=. .venv/bin/python scripts/val_score_csv.py --dataset lmo --split val_real A.csv [B.csv ...]
"""
import argparse
import csv

import numpy as np

from src.bop_dataset import SPLITS, default_bop_root, load_index, rows_for_objects, select_rows
from src.bop_evaluate import instance_errors, summarize


def read(path: str) -> dict:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    out = {
        (int(r["scene_id"]), int(r["im_id"]), int(r["obj_id"])): (
            np.array(r["R"].split(), float).reshape(3, 3),
            np.array(r["t"].split(), float),
        )
        for r in rows
    }
    if len(out) != len(rows):
        raise ValueError(f"{path}: several estimates for one (scene, image, object)")
    return out


def score_csv(path: str, dataset: str, split: str) -> dict:
    cfg = SPLITS[dataset]
    index = load_index(dataset, split)
    preds = read(path)
    n, mssd, mspd = [], [], []
    for obj_id in cfg["obj_ids"]:
        rows = rows_for_objects(index, [obj_id])
        if len(rows) == 0:
            continue
        sub = select_rows(index, rows)
        keys = [
            (int(s), int(i), obj_id) for s, i in zip(sub["scene_id"], sub["im_id"], strict=True)
        ]
        pred = {"R": np.stack([preds[k][0] for k in keys]), "t": np.stack([preds[k][1] for k in keys])}
        s = summarize(instance_errors(sub, pred, obj_id, default_bop_root(), cfg["models_dataset"]))
        n.append(s["n"])
        mssd.append(s["recall_MSSD"])
        mspd.append(s["recall_MSPD"])
    ar_mssd = float(np.average(mssd, weights=n))
    ar_mspd = float(np.average(mspd, weights=n))
    return {"n": int(sum(n)), "AR_MSSD": ar_mssd, "AR_MSPD": ar_mspd,
            "score": (ar_mssd + ar_mspd) / 2}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=sorted(SPLITS))
    p.add_argument("--split", default="val_real")
    p.add_argument("csvs", nargs="+")
    args = p.parse_args()
    for path in args.csvs:
        r = score_csv(path, args.dataset, args.split)
        print(f"{path} {args.split}: n={r['n']} score={r['score']:.4f} "
              f"AR_MSSD={r['AR_MSSD']:.4f} AR_MSPD={r['AR_MSPD']:.4f}")


if __name__ == "__main__":
    main()
