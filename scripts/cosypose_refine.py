"""M19 R1: refine our poses with CosyPose's PBR-only render-and-compare refiner (happypose).

Reads a BOP results CSV of ours (R row-major, t in mm), runs the public CosyPose refiner
(trained on PBR images only: refiner-bop-<dataset>-pbr--*) for 1..N iterations starting at
those poses, and writes one BOP results CSV per iteration count, ready for scripts/bop_eval.sh.
The refiner renders the object at the current pose, compares it with the image crop and predicts
a pose update; it only needs the image, K, the initial pose and the object model.

Run with the happypose venv (separate from the project's .venv, TASKS.md M19; setup in
docs/RCP_WORKFLOW.md "CosyPose refiner"):
  HAPPYPOSE_DATA_DIR=$NAS_HOME/happypose_data $NAS_HOME/venvs/happypose/bin/python \
      scripts/cosypose_refine.py --dataset lmo --csv RUN/eval_det/fastposevitmo_lmo-test.csv \
      --out_dir $NAS_HOME/results/m19_lmo_r1 --iterations 4
Output: OUT_DIR/<method>r<n>_<dataset>-test.csv for n = 1..N (method = the CSV's method + "r<n>",
no "_"), and OUT_DIR/refine_time.json. The BOP `time` of an image = our time + the refiner's time
(measured on the whole image, scaled by n / N for the n-iteration file).
Credit: CosyPose (Labbé et al., ECCV 2020), via happypose (agimus-project); training data stays
PBR only.
"""

import argparse
import csv
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

# dataset -> (happypose object dataset name, label prefix, BOP test folder, model run ids)
DATASETS = {
    "lmo": ("lmo", "lm", "lmo/test"),
    "ycbv": ("ycbv", "ycbv", "ycbv/test"),
    "tless": ("tless.cad", "tless", "tless/test_primesense"),
}


def read_results(path: str) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def pose_matrix(row: dict) -> np.ndarray:
    """4x4 camera-from-model pose, translation in metres (CosyPose units)."""
    T = np.eye(4)
    T[:3, :3] = np.array(row["R"].split(), dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.array(row["t"].split(), dtype=np.float64) / 1000.0
    return T


def load_image(test_dir: Path, scene_id: int, im_id: int) -> np.ndarray:
    scene = test_dir / f"{scene_id:06d}"
    for ext in ("png", "jpg"):
        path = scene / "rgb" / f"{im_id:06d}.{ext}"
        if path.exists():
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    raise FileNotFoundError(f"no RGB image for scene {scene_id} image {im_id} in {scene}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--csv", required=True, help="our BOP results CSV (test split)")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--max_images", type=int, default=0, help="debug: first N images only")
    parser.add_argument("--bop_root", default=os.path.join(os.environ.get("NAS_HOME", "."), "datasets", "bop"))
    args = parser.parse_args()
    if "HAPPYPOSE_DATA_DIR" not in os.environ:
        parser.error("set HAPPYPOSE_DATA_DIR (bop_datasets/ and experiments/ inside)")
    # cosypose/config.py calls getpass.getuser(), which fails in a container without a passwd
    # entry for our uid; any name does
    os.environ.setdefault("USER", "rcp")

    # happypose reads HAPPYPOSE_DATA_DIR at import time
    from happypose.pose_estimators.cosypose.cosypose.bop_config import PBR_COARSE, PBR_REFINER
    from happypose.pose_estimators.cosypose.cosypose.integrated.pose_predictor import (
        CoarseRefinePosePredictor,
    )
    import happypose.pose_estimators.cosypose.cosypose.utils.tensor_collection as tc
    from happypose.pose_estimators.cosypose.cosypose.evaluation.evaluation import (
        load_pose_models_cosypose,
    )
    from happypose.toolbox.datasets.datasets_cfg import make_object_dataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_name, prefix, test_rel = DATASETS[args.dataset]
    key = "lmo" if args.dataset == "lmo" else args.dataset
    test_dir = Path(args.bop_root) / test_rel
    rows = read_results(args.csv)
    method = Path(args.csv).name.split("_")[0]

    object_dataset = make_object_dataset(ds_name)
    # the original CosyPose checkpoints (their own config format, the Panda3D renderer)
    _, refiner, _ = load_pose_models_cosypose(
        object_dataset, PBR_COARSE[key], PBR_REFINER[key], n_workers=4
    )
    predictor = CoarseRefinePosePredictor(coarse_model=None, refiner_model=refiner)

    by_image = defaultdict(list)
    for i, row in enumerate(rows):
        by_image[(int(row["scene_id"]), int(row["im_id"]))].append(i)
    keys = sorted(by_image)
    if args.max_images:
        keys = keys[: args.max_images]

    cameras: dict = {}
    refined = {n: {} for n in range(1, args.iterations + 1)}  # n -> row index -> (R, t mm)
    refine_seconds = {}
    for count, (scene_id, im_id) in enumerate(keys):
        if scene_id not in cameras:
            with open(test_dir / f"{scene_id:06d}" / "scene_camera.json") as f:
                cameras[scene_id] = json.load(f)
        K = np.array(cameras[scene_id][str(im_id)]["cam_K"], dtype=np.float32).reshape(3, 3)
        image = load_image(test_dir, scene_id, im_id)
        idx = by_image[(scene_id, im_id)]
        infos = pd.DataFrame(
            {
                "label": [f"{prefix}-obj_{int(rows[i]['obj_id']):06d}" for i in idx],
                "batch_im_id": 0,
                "row": idx,
            }
        )
        poses = torch.tensor(np.stack([pose_matrix(rows[i]) for i in idx]), dtype=torch.float32)
        init = tc.PandasTensorCollection(infos=infos, poses=poses.to(device))
        images = torch.from_numpy(image).to(device).float().permute(2, 0, 1)[None] / 255.0
        Ks = torch.from_numpy(K)[None].to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        _, preds = predictor.run_inference_pipeline(
            images, Ks, data_TCO_init=init, n_coarse_iterations=0,
            n_refiner_iterations=args.iterations,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        refine_seconds[(scene_id, im_id)] = time.perf_counter() - start
        for n in range(1, args.iterations + 1):
            out = preds[f"refiner/iteration={n}"]
            T = out.poses.detach().cpu().numpy().astype(np.float64)
            for j, i in enumerate(out.infos["row"].values):
                refined[n][int(i)] = (T[j, :3, :3], T[j, :3, 3] * 1000.0)
        if count % 20 == 0:
            print(f"{count + 1}/{len(keys)} images, {refine_seconds[(scene_id, im_id)]:.2f} s", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    fields = ["scene_id", "im_id", "obj_id", "score", "R", "t", "time"]
    for n in range(1, args.iterations + 1):
        path = os.path.join(args.out_dir, f"{method}r{n}_{args.dataset}-test.csv")
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for i in sorted(refined[n]):
                row = dict(rows[i])
                R, t = refined[n][i]
                row["R"] = " ".join(f"{v:.8f}" for v in R.reshape(-1))
                row["t"] = " ".join(f"{v:.6f}" for v in t)
                key_ = (int(row["scene_id"]), int(row["im_id"]))
                row["time"] = float(row["time"]) + refine_seconds[key_] * n / args.iterations
                writer.writerow(row)
        print(f"{len(refined[n])} poses -> {path}")
    with open(os.path.join(args.out_dir, "refine_time.json"), "w") as f:
        json.dump(
            {"images": len(refine_seconds), "mean_seconds_all_iterations": float(np.mean(list(refine_seconds.values()))),
             "iterations": args.iterations},
            f, indent=1,
        )


if __name__ == "__main__":
    main()
