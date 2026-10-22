# This file evaluates a trained FastPoseViT on a BOP split with ground-truth boxes: per-instance
# rotation (deg), translation (mm), ADD, MSSD and MSPD errors, plus a BOP19-style results CSV.
#
# The metrics are computed per instance with bop_toolkit_lib.pose_error. They are not the BOP19
# AR (which needs all objects of the dataset in the results file); see scripts/bop_eval.sh.

import argparse
import json
import logging
import os

import numpy as np
import torch
from bop_toolkit_lib import inout, misc, pose_error
from torch.amp import autocast
from torch.utils.data import DataLoader

from src.bop_dataset import (
    LMO_OBJ_IDS,
    BOPDataset,
    default_bop_root,
    load_index,
    rows_for_objects,
    select_rows,
)
from src.engine import CoordinateTransformer, forward_model
from src.models import VIT_MODELS, FastPoseViT
from src.utils import process_rotation, quaternion_to_rotation_matrix

# BOP19 error thresholds: MSSD in fractions of the diameter, MSPD in pixels at 640 px width
MSSD_THRESHOLDS = np.arange(0.05, 0.51, 0.05)
MSPD_THRESHOLDS = np.arange(5, 51, 5)
VSD_SKIPPED = "VSD needs the renderer: use scripts/bop_eval.sh on the CSV with all objects"


def build_model(args: argparse.Namespace, device: torch.device) -> FastPoseViT:
    """The architecture of the best SPEED config on BOP (matrix rotation, merged head, no MLP)."""
    model = FastPoseViT(
        vit_model=args.vit_model,
        num_hidden_layers=0,
        hidden_layer_dim=0,
        out_dim_translation=3,
        out_dim_rotation=6,
        merge_outputs=True,
        nb_class_tokens=1,
        no_mlp=True,
        num_objects=len(LMO_OBJ_IDS),
    )
    # model.pth is a state dict; last.pt (train.py checkpoint) holds it under "model"
    state = torch.load(args.model_weights, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)
    return model.to(device).eval()


@torch.no_grad()
def predict(model, dataset, args, device) -> dict:
    """Absolute poses (R, t in mm) predicted for every row of `dataset`, in row order."""
    loader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False
    )
    transformer = CoordinateTransformer(
        rotation_format="matrix",
        val_camera=dataset.camera,
        no_rotation_compensation=False,
        depth_mode=args.depth_mode,
    )
    out = {"R": [], "t": []}
    for batch in loader:
        images = batch["images"].to(device)
        obj_idx = batch["obj_idx"].to(device)
        with autocast("cuda", enabled=device.type == "cuda"):
            pred_t, pred_R = forward_model(model, images, obj_idx, merge_outputs=True)
            pred_R = process_rotation(pred_R, "matrix")
        # The ground-truth entries are returned too and ignored: only the prediction is needed
        abs_t, abs_q, _, _ = transformer.transform_to_absolute(
            pred_t.float(),
            pred_R.float(),
            batch["translations"].to(device),
            batch["rotations"].to(device),
            batch["bboxes"],
            K=batch["K"],
            img_wh=batch["img_wh"],
            diameter=batch["diameter"],
        )
        out["R"].append(quaternion_to_rotation_matrix(abs_q).cpu().numpy())
        out["t"].append(abs_t.cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def rotation_error_deg(R_est: np.ndarray, R_gt: np.ndarray) -> float:
    cos = (np.trace(R_est.T @ R_gt) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def instance_errors(index: dict, pred: dict, obj_id: int, bop_root: str) -> dict:
    """Per-instance errors of one object.

    For a symmetric object (LM-O eggbox 10, glue 11), "add" is ADD-S, "rot_deg" is the smallest
    angle to any symmetric GT pose, and MSSD/MSPD use the BOP19 symmetry set (models_info.json,
    continuous symmetries discretised at 1 % of the diameter). "trans_mm" is always plain.
    """
    obj_idx = int(np.flatnonzero(index["obj_ids"] == obj_id)[0])
    diameter = float(index["diameters"][obj_idx])
    models_dir = os.path.join(bop_root, "lmo", "models_eval")
    pts = inout.load_ply(os.path.join(models_dir, f"obj_{obj_id:06d}.ply"))["pts"]
    symmetric = bool(index["symmetric"][obj_idx])
    if symmetric:
        info = inout.load_json(os.path.join(models_dir, "models_info.json"), keys_to_int=True)
        syms = misc.get_symmetry_transformations(info[obj_id], max_sym_disc_step=0.01)
    else:
        syms = [{"R": np.eye(3), "t": np.zeros((3, 1))}]

    errs = {k: [] for k in ("rot_deg", "trans_mm", "add", "mssd", "mspd")}
    for i in range(len(pred["R"])):
        R_est, t_est = pred["R"][i].astype(np.float64), pred["t"][i].astype(np.float64)[:, None]
        R_gt, t_gt = index["R"][i].astype(np.float64), index["t"][i].astype(np.float64)[:, None]
        K = index["K"][i].astype(np.float64)
        errs["rot_deg"].append(min(rotation_error_deg(R_est, R_gt @ sym["R"]) for sym in syms))
        errs["trans_mm"].append(float(np.linalg.norm(t_est - t_gt)))
        add = pose_error.adi if symmetric else pose_error.add
        errs["add"].append(add(R_est, t_est, R_gt, t_gt, pts))
        errs["mssd"].append(pose_error.mssd(R_est, t_est, R_gt, t_gt, pts, syms))
        errs["mspd"].append(pose_error.mspd(R_est, t_est, R_gt, t_gt, K, pts, syms))
    errs = {k: np.array(v) for k, v in errs.items()}
    errs["diameter"] = diameter
    errs["img_w"] = index["img_wh"][:, 0].astype(np.float64)
    return errs


def summarize(errs: dict) -> dict:
    """Mean/median errors and the recalls used by BOP and by the LM literature."""
    d = errs["diameter"]
    # MSPD thresholds scale with the image width relative to 640
    mspd_scaled = errs["mspd"] * 640.0 / errs["img_w"]
    return {
        "n": int(len(errs["rot_deg"])),
        "rot_deg_mean": float(errs["rot_deg"].mean()),
        "rot_deg_median": float(np.median(errs["rot_deg"])),
        "trans_mm_mean": float(errs["trans_mm"].mean()),
        "trans_mm_median": float(np.median(errs["trans_mm"])),
        "add_mm_mean": float(errs["add"].mean()),
        "add_mm_median": float(np.median(errs["add"])),
        "ADD@0.1d": float((errs["add"] < 0.1 * d).mean()),
        "ADD@0.05d": float((errs["add"] < 0.05 * d).mean()),
        "ADD@0.02d": float((errs["add"] < 0.02 * d).mean()),
        "recall_MSSD": float(np.mean([(errs["mssd"] < t * d).mean() for t in MSSD_THRESHOLDS])),
        "recall_MSPD": float(np.mean([(mspd_scaled < t).mean() for t in MSPD_THRESHOLDS])),
        "rot_lt_5deg": float((errs["rot_deg"] < 5).mean()),
        "trans_lt_5mm": float((errs["trans_mm"] < 5).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate FastPoseViT on BOP (GT boxes)")
    parser.add_argument("--model_weights", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--splits", nargs="+", default=["val_real", "test"])
    parser.add_argument("--obj_id", type=int, required=True)
    parser.add_argument("--vit_model", default="vit_b_16_384")
    parser.add_argument("--depth_mode", default="site")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--method", default="fpvit", help="BOP method name (no '_')")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args, device)
    bop_root = default_bop_root()
    os.makedirs(args.out_dir, exist_ok=True)

    summary = {}
    for split in args.splits:
        index = load_index("lmo", split)
        rows = rows_for_objects(index, [args.obj_id])
        sub = select_rows(index, rows)
        dataset = BOPDataset(
            index=sub,
            augment=False,
            bbox_source="visib",
            depth_mode=args.depth_mode,
            img_size=VIT_MODELS[args.vit_model][3][0],
        )
        pred = predict(model, dataset, args, device)
        errs = instance_errors(sub, pred, args.obj_id, bop_root)
        summary[split] = summarize(errs)
        logging.info(f"{split}: {json.dumps(summary[split])}")

        np.savez(
            os.path.join(args.out_dir, f"errors_{split}.npz"),
            scene_id=sub["scene_id"],
            im_id=sub["im_id"],
            gt_id=sub["gt_id"],
            R_est=pred["R"],
            t_est=pred["t"],
            **{k: v for k, v in errs.items() if k not in ("diameter", "img_w")},
        )
        results = [
            {
                "scene_id": int(sub["scene_id"][i]),
                "im_id": int(sub["im_id"][i]),
                "obj_id": args.obj_id,
                "score": 1.0,
                "R": pred["R"][i],
                "t": pred["t"][i],
            }
            for i in range(len(rows))
        ]
        inout.save_bop_results(
            os.path.join(args.out_dir, f"{args.method}_lmo-{split}.csv"), results
        )

    summary["note"] = VSD_SKIPPED
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
