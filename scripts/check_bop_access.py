"""Check that BOP data is readable (e.g. from inside a `train` job).

Reads one PBR RGB image, its scene_gt.json and one models_eval ply.
Usage: .venv/bin/python scripts/check_bop_access.py [--dataset lmo]
"""

import argparse
import json
import os

import cv2
from bop_toolkit_lib import inout

# Nested layout of $DATA_DIR/BOP_Datasets: (PBR scene dir, models_eval dir)
PATHS = {
    "lmo": ("lm/lm_train_pbr/train_pbr/000000", "lmo/lmo_models/models_eval"),
    "ycbv": ("ycbv/ycbv_train_pbr/train_pbr/000000", "ycbv/ycbv_models/models_eval"),
}

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="lmo", choices=sorted(PATHS))
args = parser.parse_args()

root = os.path.join(os.environ["DATA_DIR"], "BOP_Datasets")
scene_dir, models_dir = (os.path.join(root, p) for p in PATHS[args.dataset])

img = cv2.imread(os.path.join(scene_dir, "rgb", "000000.jpg"))
assert img is not None, f"cannot read RGB image in {scene_dir}"
print("rgb", img.shape)

with open(os.path.join(scene_dir, "scene_gt.json")) as f:
    gt = json.load(f)
print("scene_gt images", len(gt), "| instances in image 0:", len(gt["0"]))

ply = sorted(p for p in os.listdir(models_dir) if p.endswith(".ply"))[0]
model = inout.load_ply(os.path.join(models_dir, ply))
print(ply, "points", model["pts"].shape)

print(f"BOP ACCESS OK ({args.dataset})")
