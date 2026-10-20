"""Write a BOP19 results CSV that uses the ground-truth poses as predictions.

Sanity check for the evaluation pipeline: scoring this file with
bop_toolkit's scripts/eval_bop19_pose.py must give AR close to 1.0.
For every target in test_targets_bop19.json, all GT instances of that object
in the image are written (score 1, time -1).

Usage (needs the tree from scripts/make_bop_tree.sh):
  BOP_PATH=$NAS_HOME/datasets/bop .venv/bin/python scripts/bop_gt_as_prediction.py \
      --dataset lmo --out_dir RESULTS_DIR
"""

import argparse
import os

import numpy as np
from bop_toolkit_lib import inout

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="lmo")
parser.add_argument("--split", default="test")
parser.add_argument("--targets_filename", default="test_targets_bop19.json")
parser.add_argument("--method", default="gtpred", help="Method name; must not contain '_'.")
parser.add_argument("--out_dir", required=True)
args = parser.parse_args()
assert "_" not in args.method, "BOP method names must not contain '_'"

ds_dir = os.path.join(os.environ["BOP_PATH"], args.dataset)
targets = inout.load_json(os.path.join(ds_dir, args.targets_filename))

scene_gts = {}
results = []
for target in targets:
    scene_id, im_id, obj_id = target["scene_id"], target["im_id"], target["obj_id"]
    if scene_id not in scene_gts:
        scene_gts[scene_id] = inout.load_scene_gt(
            os.path.join(ds_dir, args.split, f"{scene_id:06d}", "scene_gt.json")
        )
    for gt in scene_gts[scene_id][im_id]:
        if gt["obj_id"] == obj_id:
            results.append(
                {
                    "scene_id": scene_id,
                    "im_id": im_id,
                    "obj_id": obj_id,
                    "score": 1.0,
                    "R": np.asarray(gt["cam_R_m2c"]),
                    "t": np.asarray(gt["cam_t_m2c"]),  # mm
                }
            )

os.makedirs(args.out_dir, exist_ok=True)
path = os.path.join(args.out_dir, f"{args.method}_{args.dataset}-{args.split}.csv")
inout.save_bop_results(path, results)
n_insts = sum(t["inst_count"] for t in targets)
print(f"{len(targets)} targets ({n_insts} instances) -> {len(results)} estimates in {path}")
