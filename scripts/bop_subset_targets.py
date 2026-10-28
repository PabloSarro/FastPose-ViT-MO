"""Write a BOP19 targets file restricted to some objects, next to the official one.

Used for the official evaluation on LM real val (= BOP's LM test targets of the 8 LM-O objects):
without it, eval_bop19_pose.py would count the 7 other LM objects as missed.

Usage (writes into the tree of scripts/make_bop_tree.sh, never into $DATA_DIR):
  .venv/bin/python scripts/bop_subset_targets.py --dataset lm --obj_ids 1 5 6 8 9 10 11 12 \
      --name test_targets_bop19_lmo8.json
then: bash scripts/bop_eval.sh DIR fastposevitmo_lm-test.csv --targets_filename=test_targets_bop19_lmo8.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bop_dataset import default_bop_root  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--dataset", default="lm")
parser.add_argument("--obj_ids", type=int, nargs="+", required=True)
parser.add_argument("--name", required=True, help="output file name, in the dataset folder")
parser.add_argument("--source", default="test_targets_bop19.json")
args = parser.parse_args()

folder = os.path.join(default_bop_root(), args.dataset)
out = os.path.join(folder, args.name)
if os.path.realpath(out).startswith(os.path.realpath(os.environ.get("DATA_DIR", "/nonexistent"))):
    sys.exit(f"{out} would be inside $DATA_DIR (read only)")
with open(os.path.join(folder, args.source)) as f:
    targets = [t for t in json.load(f) if t["obj_id"] in set(args.obj_ids)]
with open(out, "w") as f:
    json.dump(targets, f)
print(f"{len(targets)} targets of objects {sorted(set(args.obj_ids))} -> {out}")
