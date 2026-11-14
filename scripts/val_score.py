"""Validation score of a bop_evaluate.py run (GT boxes) from OUT_DIR/summary.json.

score = (AR_MSSD + AR_MSPD) / 2 with the own recalls (as the in-training selection metric and as
bop_metrics.py), the "mean over instances" entry of bop_evaluate's summary.json; also
ADD(-S)@0.1d and the mean translation / rotation errors. One block per split.
Used to choose between configurations on validation instead of on the LM-O test set (audit W2).

Usage: .venv/bin/python scripts/val_score.py EVAL_DIR [EVAL_DIR ...] [--json]
"""
import json
import sys


def score(summary: dict) -> dict:
    out = {}
    for split, per_obj in summary.items():
        if not isinstance(per_obj, dict):  # the "note" entry
            continue
        # per-object entries and two aggregates written by bop_evaluate.write_tables
        m = per_obj["mean over instances"]
        out[split] = {
            "n": m["n"],
            "AR_MSSD": m["recall_MSSD"],
            "AR_MSPD": m["recall_MSPD"],
            "score": (m["recall_MSSD"] + m["recall_MSPD"]) / 2,
            "ADD@0.1d": m["ADD@0.1d"],
            "trans_mm": m["trans_mm_mean"],
            "rot_deg": m["rot_deg_mean"],
        }
    return out


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    as_json = "--json" in sys.argv
    result = {}
    for d in args:
        with open(f"{d}/summary.json") as f:
            result[d] = score(json.load(f))
    if as_json:
        print(json.dumps(result, indent=1))
        return
    for d, r in result.items():
        for split, v in r.items():
            print(f"{d} {split}: n={v['n']} score={v['score']:.4f} AR_MSSD={v['AR_MSSD']:.4f} "
                  f"AR_MSPD={v['AR_MSPD']:.4f} ADD@0.1d={v['ADD@0.1d']:.4f} "
                  f"trans={v['trans_mm']:.2f}mm rot={v['rot_deg']:.2f}deg")


if __name__ == "__main__":
    main()
