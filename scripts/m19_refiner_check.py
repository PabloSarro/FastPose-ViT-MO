"""M19 R1 sanity check of the CosyPose refiner setup: GT poses + noise in, does the error go down?

The refiner must reduce both the rotation and the translation error of noisy GT poses. If it
makes the rotation worse (as the Panda3D renderer did, TASKS.md M19), the setup does not match
how the checkpoints were trained (PyBullet renderer) and refining our poses means nothing.

Run with the happypose venv (see scripts/cosypose_refine.py), LM-O test scene 2 or YCB-V test scene 50,
symmetric objects left out:
  HAPPYPOSE_DATA_DIR=$NAS_HOME/happypose_data PYTHONPATH=$NAS_HOME/bop_toolkit \
  HAPPYPOSE_EGL_DEVICE=0 CUDA_VISIBLE_DEVICES=0 \
  $NAS_HOME/venvs/happypose/bin/python scripts/m19_refiner_check.py --renderer bullet \
      --rot_noise 5 --trans_noise 10
"""

import argparse
import json
import os

import cv2
import numpy as np
import torch

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--dataset", default="lmo", choices=["lmo", "ycbv"])
parser.add_argument("--renderer", default="bullet", choices=["bullet", "panda3d"])
parser.add_argument("--rot_noise", type=float, default=5.0, help="std of the rotation noise, degrees")
parser.add_argument("--trans_noise", type=float, default=10.0, help="std of the translation noise, mm")
parser.add_argument("--n_images", type=int, default=30)
parser.add_argument("--iterations", type=int, default=2)
args = parser.parse_args()
os.environ.setdefault("USER", "rcp")  # cosypose/config.py calls getpass.getuser()

from happypose.pose_estimators.cosypose.cosypose.bop_config import PBR_COARSE, PBR_REFINER  # noqa: E402
from happypose.pose_estimators.cosypose.cosypose.evaluation.evaluation import (  # noqa: E402
    load_pose_models_cosypose,
)
from happypose.toolbox.datasets.datasets_cfg import make_object_dataset  # noqa: E402

# dataset -> (test scene, label prefix, happypose object dataset)
SCENE = {"lmo": ("lmo/test/000002", "lm"), "ycbv": ("ycbv/test/000050", "ycbv")}
test_dir = os.environ["NAS_HOME"] + "/datasets/bop/" + SCENE[args.dataset][0]
prefix = SCENE[args.dataset][1]
models_info = json.load(open(os.environ["NAS_HOME"] + f"/datasets/bop/{args.dataset}/models_eval/models_info.json"))
symmetric = {int(k) for k, v in models_info.items() if "symmetries_discrete" in v or "symmetries_continuous" in v}
_, refiner, _ = load_pose_models_cosypose(
    make_object_dataset(args.dataset), PBR_COARSE[args.dataset], PBR_REFINER[args.dataset],
    1 if args.renderer == "bullet" else 4,  # Bullet workers all rebuild urdfs/tmp: one worker, no race
    renderer_type=args.renderer,
)
rng = np.random.default_rng(0)
gt_all = json.load(open(f"{test_dir}/scene_gt.json"))
cams = json.load(open(f"{test_dir}/scene_camera.json"))


def rotation(axis, deg):
    axis = axis / np.linalg.norm(axis)
    a = np.radians(deg)
    A = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * A + (1 - np.cos(a)) * A @ A


def error(T, Tg):
    c = (np.trace(T[:3, :3].T @ Tg[:3, :3]) - 1) / 2
    return np.degrees(np.arccos(np.clip(c, -1, 1))), np.linalg.norm(T[:3, 3] - Tg[:3, 3]) * 1000


errors = {n: [] for n in range(args.iterations + 1)}  # 0 = the noisy input
for im in list(gt_all)[::5][: args.n_images]:
    img = cv2.cvtColor(cv2.imread(f"{test_dir}/rgb/{int(im):06d}.png"), cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(img).cuda().float().permute(2, 0, 1)[None] / 255
    K = torch.tensor(cams[im]["cam_K"], dtype=torch.float32).reshape(3, 3)[None].cuda()
    for g in gt_all[im]:
        if g["obj_id"] in symmetric:  # the rotation error is ill-defined
            continue
        Tg = np.eye(4)
        Tg[:3, :3] = np.array(g["cam_R_m2c"]).reshape(3, 3)
        Tg[:3, 3] = np.array(g["cam_t_m2c"]) / 1000
        T0 = Tg.copy()
        T0[:3, :3] = Tg[:3, :3] @ rotation(rng.normal(size=3), rng.normal() * args.rot_noise)
        T0[:3, 3] += rng.normal(size=3) * args.trans_noise / 1000
        with torch.no_grad():
            out = refiner(
                images=x, K=K, labels=[f"{prefix}-obj_{g['obj_id']:06d}"],
                TCO=torch.tensor(T0, dtype=torch.float32)[None].cuda(), n_iterations=args.iterations,
            )
        errors[0].append(error(T0, Tg))
        for n in range(1, args.iterations + 1):
            T = out[f"iteration={n}"].TCO_output[0].cpu().numpy().astype(np.float64)
            errors[n].append(error(T, Tg))
for n, e in errors.items():
    e = np.array(e)
    name = "input" if n == 0 else f"{n} iteration(s)"
    print(f"{args.dataset} {args.renderer} {name}: n={len(e)} rotation median {np.median(e[:, 0]):.2f} deg, "
          f"translation median {np.median(e[:, 1]):.1f} mm", flush=True)
