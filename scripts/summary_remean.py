"""Recompute the two mean rows of bop_evaluate summary.json files with the current
per_object_table (objects without instances count in neither mean), and rewrite val_score.txt.
For evaluations of splits that miss some objects (YCB-V val_real) made before 2026-11-17, whose
means were nan. Usage: .venv/bin/python scripts/summary_remean.py EVAL_DIR [EVAL_DIR ...]"""
import json
import sys

from scripts.val_score import score
from src.bop_evaluate import per_object_table

MEANS = ("mean over objects", "mean over instances")

for d in sys.argv[1:]:
    path = f"{d}/summary.json"
    with open(path) as f:
        summary = json.load(f)
    for split, per_obj in summary.items():
        if not isinstance(per_obj, dict):
            continue
        objs = {o: v for o, v in per_obj.items() if o not in MEANS}
        summary[split].update(per_object_table(objs)[1])
    with open(path, "w") as f:
        json.dump(summary, f, indent=1)
    for split, v in score(summary).items():
        line = (f"{d} {split}: n={v['n']} score={v['score']:.4f} AR_MSSD={v['AR_MSSD']:.4f} "
                f"AR_MSPD={v['AR_MSPD']:.4f} ADD@0.1d={v['ADD@0.1d']:.4f} "
                f"trans={v['trans_mm']:.2f}mm rot={v['rot_deg']:.2f}deg")
        print(line)
        with open(f"{d}/val_score.txt", "w") as f:
            f.write(line + "\n")
