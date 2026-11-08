"""Per-object official BOP19 AR from an existing scripts/bop_eval.sh output (no re-evaluation).

eval_bop19_pose.py writes, for each error type (VSD for each tau, MSSD, MSPD) and threshold, a
scores_th=*.json with the recall over all instances and per object ("obj_recalls"). The BOP19 AR
is, per error type, the mean recall over its thresholds (and VSD taus), then the mean of the three
types. Doing the same with obj_recalls gives each object's AR. As a check, the same computation
with the overall "recall" must equal bop19_average_recall of scores_bop19.json.

Usage: .venv/bin/python scripts/bop_per_object_ar.py EVAL_DIR [--names 1=ape 5=can ...]
  EVAL_DIR = the folder with error=... subfolders (OUT/eval/<method>_<dataset>-<split>), or
  OUT/eval (its single subfolder is used). Prints a markdown table; --out FILE also writes it.
"""

import argparse
import glob
import json
import os

import numpy as np

LMO_NAMES = {1: "ape", 5: "can", 6: "cat", 8: "driller", 9: "duck", 10: "eggbox", 11: "glue",
             12: "holepuncher"}
YCBV_NAMES = dict(enumerate([
    "master_chef_can", "cracker_box", "sugar_box", "tomato_soup_can", "mustard_bottle",
    "tuna_fish_can", "pudding_box", "gelatin_box", "potted_meat_can", "banana", "pitcher_base",
    "bleach_cleanser", "bowl", "mug", "power_drill", "wood_block", "scissors", "large_marker",
    "large_clamp", "extra_large_clamp", "foam_brick"], start=1))


def per_type_recalls(eval_dir: str) -> dict:
    """{type: {"all": [recall per threshold], obj_id: [...]}} for vsd, mssd, mspd."""
    out = {t: {"all": []} for t in ("vsd", "mssd", "mspd")}
    for sub in sorted(glob.glob(os.path.join(eval_dir, "error=*"))):
        etype = os.path.basename(sub).split("=")[1].split("_")[0]
        for f in sorted(glob.glob(os.path.join(sub, "scores_th=*.json"))):
            with open(f) as fh:
                s = json.load(fh)
            out[etype]["all"].append(s["recall"])
            for obj, r in s["obj_recalls"].items():
                out[etype].setdefault(int(obj), []).append(r)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("eval_dir")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    eval_dir = args.eval_dir
    if not glob.glob(os.path.join(eval_dir, "error=*")):
        subs = [d for d in glob.glob(os.path.join(eval_dir, "*")) if os.path.isdir(d)]
        if len(subs) != 1:
            raise SystemExit(f"expected one result folder in {eval_dir}, found {subs}")
        eval_dir = subs[0]
    names = YCBV_NAMES if "_ycbv-" in os.path.basename(eval_dir) else LMO_NAMES
    rec = per_type_recalls(eval_dir)
    objs = sorted(k for k in rec["mssd"] if k != "all")
    rows = []
    for key in objs + ["all"]:
        per = {t: float(np.mean(rec[t][key])) for t in rec}
        per["ar"] = float(np.mean(list(per.values())))
        rows.append((key, per))
    official = None
    scores = os.path.join(eval_dir, "scores_bop19.json")
    if os.path.exists(scores):
        with open(scores) as fh:
            official = json.load(fh)["bop19_average_recall"]
    lines = [
        "| Object | AR | AR_VSD | AR_MSSD | AR_MSPD |",
        "|---|---|---|---|---|",
    ]
    for key, per in rows:
        name = "**all instances**" if key == "all" else f"{key} {names.get(key, '')}".strip()
        lines.append(
            f"| {name} | {per['ar']:.3f} | {per['vsd']:.3f} | {per['mssd']:.3f} | {per['mspd']:.3f} |"
        )
    mean_obj = float(np.mean([per["ar"] for key, per in rows if key != "all"]))
    lines.append(f"| mean over objects | {mean_obj:.3f} | | | |")
    text = "\n".join(lines) + "\n"
    if official is not None:
        diff = abs(rows[-1][1]["ar"] - official)
        text += f"\nCheck: recomputed AR {rows[-1][1]['ar']:.4f} vs official {official:.4f} (diff {diff:.1e})\n"
    print(text)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text)


if __name__ == "__main__":
    main()
