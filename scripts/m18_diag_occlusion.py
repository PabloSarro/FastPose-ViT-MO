"""M18.7 diagnostic: what a dense model does with occluded pixels, and what that costs PnP.

Two parts, both on trained weights (no training):

real       LM-O test with the BOP default detections. For every detected instance the visible mask
           (mask_visib) and the full silhouette (mask) are sampled on the dense grid of its crop:
           share of the visible / occluded object pixels the model calls visible, and where its
           mask pixels lie. PnP with three masks, each written as a BOP CSV (for the official
           eval) and scored with the own MSSD / MSPD recalls:
             pred         the predicted mask, as in the normal evaluation
             no_occluded  without the predicted pixels on occluded parts of the object (oracle)
             in_visible   only the predicted pixels inside the true visible mask (oracle)
synthetic  LM real validation with the noisy boxes of eval_valn: the training occlusion
           rectangles (always applied, fixed seed) added to the crops at inference. Share of the
           erased object pixels still called visible, XYZ error there and elsewhere, and the own
           PnP score on the clean crops, on the occluded ones, and on the occluded ones without the
           erased pixels (oracle).

Usage (repo root, in dev):
  PYTHONPATH=. .venv/bin/python scripts/m18_diag_occlusion.py --out OUT VARIANT_DIR [VARIANT_DIR ...]
  bash scripts/bop_eval.sh OUT/<name>/<mask> fastposevitmo_lmo-test.csv     # official AR
"""
import argparse
import glob
import json
import os

import albumentations as A
import cv2
import numpy as np
import torch
from bop_toolkit_lib import inout
from torch.amp import autocast
from torch.utils.data import DataLoader

from src.bop_dataset import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    SPLITS,
    BOPDataset,
    crop_warp_matrix,
    default_bop_root,
    load_index,
    object_extents,
    rows_for_objects,
    select_rows,
)
from src.bop_evaluate import (
    assign_detections,
    build_model,
    instance_errors,
    noisy_boxes_for_rows,
    pose_transformer,
    summarize,
)
from src.engine import forward_dense
from src.pnp import PoseRefiner
from src.utils import process_rotation, quaternion_to_rotation_matrix

OBJ_IDS = SPLITS["lmo"]["obj_ids"]
BATCH = 32  # as bop_evaluate: the PnP seed is the position in the batch
DENSE = 64
OFF = -1e4  # mask logit of a pixel taken out of the correspondences


def model_args(variant_dir: str) -> argparse.Namespace:
    return argparse.Namespace(
        dataset="lmo",
        vit_model="vit_b_16_384",
        object_conditioning="token",
        model_weights=os.path.join(variant_dir, "last.pt"),
        dense_heads=True,
        dense_size=DENSE,
        depth_mode="site",
        no_rotation_compensation=False,
    )


def make_dataset(index: dict, split: str) -> BOPDataset:
    return BOPDataset(
        dataset="lmo", split=split, index=index, img_size=384, dense_size=DENSE,
        augment=False, depth_mode="site",
    )


def forward(model, dataset, args, device, rectangles=None) -> dict:
    """Dense outputs, DR poses and targets of every row; with `rectangles`, occluded crops."""
    transformer = pose_transformer(dataset, args)
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    keys = ("dense", "R", "t", "bboxes", "K", "obj_idx", "mask", "xyz", "xyz_valid", "erased")
    out = {k: [] for k in keys}
    row = 0
    for batch in DataLoader(dataset, batch_size=BATCH, num_workers=8, shuffle=False):
        images = batch["images"]
        erased = np.zeros((len(images), DENSE, DENSE), dtype=bool)
        if rectangles is not None:
            images = images.clone()
            for i in range(len(images)):
                img = images[i].permute(1, 2, 0).numpy() * std + mean
                res = rectangles(image=img.astype(np.float32),
                                 mask=np.zeros(img.shape[:2], dtype=np.uint8))
                images[i] = torch.from_numpy(((res["image"] - mean) / std).transpose(2, 0, 1))
                erased[i] = dataset.covered_dense(res["mask"] > 0, batch["bboxes"][i].tolist())
                row += 1
        with torch.no_grad(), autocast("cuda", enabled=device.type == "cuda"):
            pred_t, pred_R, dense = forward_dense(
                model, images.to(device), batch["obj_idx"].to(device), merge_outputs=True
            )
            pred_R = process_rotation(pred_R, "matrix")
        abs_t, abs_q, _, _ = transformer.transform_to_absolute(
            pred_t.float(), pred_R.float(), batch["translations"].to(device),
            batch["rotations"].to(device), batch["bboxes"], K=batch["K"],
            img_wh=batch["img_wh"], diameter=batch["diameter"],
        )
        out["dense"].append(dense.float().cpu().numpy())
        out["R"].append(quaternion_to_rotation_matrix(abs_q).cpu().numpy())
        out["t"].append(abs_t.cpu().numpy())
        for k in ("bboxes", "K", "obj_idx", "xyz"):
            out[k].append(batch[k].numpy())
        out["mask"].append(batch["mask"][:, 0].numpy() > 0)
        out["xyz_valid"].append(batch["xyz_valid"][:, 0].numpy() > 0)
        out["erased"].append(erased)
    return {k: np.concatenate(v) for k, v in out.items()}


def pnp(fw: dict, keep: np.ndarray | None, extents) -> tuple[np.ndarray, np.ndarray, int]:
    """PnP poses with the mask restricted to `keep` (N, S, S); also the number of DR fallbacks."""
    dense = fw["dense"].copy()
    if keep is not None:
        dense[:, 0] = np.where(keep, dense[:, 0], OFF)
    refiner = PoseRefiner("pnp", *extents)
    R, t, fallbacks = [], [], 0
    for s in range(0, len(dense), BATCH):
        part = slice(s, s + BATCH)
        r, tr, fb = refiner(dense[part], fw["bboxes"][part], fw["K"][part], fw["obj_idx"][part],
                            fw["R"][part], fw["t"][part])
        R.append(r)
        t.append(tr)
        fallbacks += int(fb.sum())
    return np.concatenate(R), np.concatenate(t), fallbacks


def own_score(split_index: dict, R, t, detected) -> dict:
    """Own MSSD / MSPD recalls (mean over instances); targets without a detection fail."""
    n = len(split_index["obj_id"])
    R_all = np.tile(np.eye(3), (n, 1, 1))
    t_all = np.tile(np.array([0.0, 0.0, 1e7]), (n, 1))
    R_all[detected], t_all[detected] = R, t
    counts, mssd, mspd = [], [], []
    for obj_id in OBJ_IDS:
        rows = rows_for_objects(split_index, [obj_id])
        errs = instance_errors(select_rows(split_index, rows),
                               {"R": R_all[rows], "t": t_all[rows]}, obj_id,
                               default_bop_root(), "lmo")
        s = summarize(errs)
        counts.append(s["n"])
        mssd.append(s["recall_MSSD"])
        mspd.append(s["recall_MSPD"])
    a = float(np.average(mssd, weights=counts))
    b = float(np.average(mspd, weights=counts))
    return {"AR_MSSD": a, "AR_MSPD": b, "score": (a + b) / 2}


def crop_sample(mask_img: np.ndarray, bbox) -> np.ndarray:
    """A full-image mask on the dense grid of a crop, sampled as BOPDataset.dense_targets."""
    M = crop_warp_matrix(bbox, DENSE)
    grid = np.arange(DENSE, dtype=np.float64)
    u = (grid - M[0, 2]) / M[0, 0] + 0.5
    v = (grid - M[1, 2]) / M[1, 1] + 0.5
    h, w = mask_img.shape
    ui = np.clip(np.floor(u).astype(int), 0, w - 1)
    vi = np.clip(np.floor(v).astype(int), 0, h - 1)
    x1, y1, x2, y2 = (float(c) for c in bbox)
    inside_x = (u >= 0) & (u < w) & (u >= x1) & (u <= x2)
    inside_y = (v >= 0) & (v < h) & (v >= y1) & (v <= y2)
    return (mask_img[np.ix_(vi, ui)] > 0) & inside_y[:, None] & inside_x[None, :]


def ratio(a, b) -> float:
    return float(a.sum() / max(b.sum(), 1))


def real_part(model, args, device, out_dir: str, extents) -> dict:
    index = load_index("lmo", "test")
    split_index = select_rows(index, rows_for_objects(index, OBJ_IDS))
    det_path = glob.glob(os.path.join(
        os.environ["NAS_HOME"], "datasets/bop_detections/default_task1",
        "bop23_default_detections_for_task1/gdrnppdet-pbr/gdrnppdet-pbr_lmo-test_*.json"))[0]
    det = assign_detections(split_index, det_path)
    detected = ~np.isnan(det["score"])
    pred_index = select_rows(split_index, np.flatnonzero(detected))
    pred_index["bbox_visib"] = det["bbox"][detected]
    dataset = make_dataset(pred_index, "test")
    fw = forward(model, dataset, args, device)

    full = []
    for i in range(len(dataset)):
        path = dataset.mask_path(i).replace(f"{os.sep}mask_visib{os.sep}", f"{os.sep}mask{os.sep}")
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(path)
        bbox = fw["bboxes"][i].tolist()
        full.append(crop_sample(img, bbox))
        if i < 50:  # the sampling must be the dataset's own
            vis = cv2.imread(dataset.mask_path(i), cv2.IMREAD_GRAYSCALE)
            assert (crop_sample(vis, bbox) == fw["mask"][i]).all()
    full = np.stack(full)
    visible, pred = fw["mask"], fw["dense"][:, 0] > 0
    occluded = full & ~visible
    result = {
        "instances": len(dataset),
        "occluded_share_of_object": ratio(occluded, full),
        "visible_called_visible": ratio(pred & visible, visible),
        "occluded_called_visible": ratio(pred & occluded, occluded),
        "mask_pixels_on_occluded": ratio(pred & occluded, pred),
        "mask_pixels_on_background": ratio(pred & ~full & ~visible, pred),
        "pnp": {},
    }
    for name, keep in (("pred", None), ("no_occluded", ~occluded), ("in_visible", visible)):
        R, t, fallbacks = pnp(fw, keep, extents)
        result["pnp"][name] = {**own_score(split_index, R, t, detected), "fallbacks": fallbacks}
        csv_dir = os.path.join(out_dir, name)
        os.makedirs(csv_dir, exist_ok=True)
        inout.save_bop_results(
            os.path.join(csv_dir, "fastposevitmo_lmo-test.csv"),
            [{"scene_id": int(pred_index["scene_id"][i]), "im_id": int(pred_index["im_id"][i]),
              "obj_id": int(pred_index["obj_id"][i]), "score": float(det["score"][detected][i]),
              "R": R[i], "t": t[i], "time": -1} for i in range(len(dataset))],
        )
    return result


def synthetic_part(model, args, device, extents, seed: int = 0) -> dict:
    index = load_index("lmo", "val_real")
    rows = rows_for_objects(index, OBJ_IDS)
    split_index = select_rows(index, rows)
    pred_index = dict(split_index)
    pred_index["bbox_visib"] = noisy_boxes_for_rows(index, rows, 0.1, 0)
    dataset = make_dataset(pred_index, "val_real")
    clean = forward(model, dataset, args, device)
    rectangles = A.Compose(
        [A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(0.1, 0.3),
                         hole_width_range=(0.1, 0.3), fill="random_uniform", fill_mask=1, p=1.0)],
        seed=seed,
    )
    occ = forward(model, dataset, args, device, rectangles=rectangles)
    visible, erased = occ["mask"], occ["erased"]
    hidden, shown = visible & erased, visible & ~erased
    pred_clean, pred_occ = clean["dense"][:, 0] > 0, occ["dense"][:, 0] > 0
    half = extents[1][occ["obj_idx"]]  # (N, 3) mm

    def xyz_error(fw, region):
        err = (fw["dense"][:, 1:4] - fw["xyz"]) * half[:, :, None, None]
        values = np.linalg.norm(err, axis=1)[region & fw["xyz_valid"] & (fw["dense"][:, 0] > 0)]
        return {"median_mm": float(np.median(values)), "p90_mm": float(np.percentile(values, 90))}

    result = {
        "instances": len(dataset),
        "erased_share_of_visible": ratio(hidden, visible),
        "called_visible": {
            "erased object pixels": ratio(pred_occ & hidden, hidden),
            "same pixels, clean crop": ratio(pred_clean & hidden, hidden),
            "other object pixels, occluded crop": ratio(pred_occ & shown, shown),
        },
        "mask_pixels_on_rectangles": ratio(pred_occ & erased, pred_occ),
        "xyz_error": {
            "erased object pixels": xyz_error(occ, hidden),
            "same pixels, clean crop": xyz_error(clean, hidden),
            "other object pixels, occluded crop": xyz_error(occ, shown),
        },
        "pnp": {},
    }
    detected = np.ones(len(dataset), dtype=bool)
    for name, fw, keep in (("clean", clean, None), ("occluded", occ, None),
                           ("occluded, erased pixels removed", occ, ~erased)):
        R, t, fallbacks = pnp(fw, keep, extents)
        result["pnp"][name] = {**own_score(split_index, R, t, detected), "fallbacks": fallbacks}
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("variant_dirs", nargs="+")
    a = p.parse_args()
    device = torch.device("cuda")
    extents = object_extents(default_bop_root(), "lmo", OBJ_IDS)
    for d in a.variant_dirs:
        name = os.path.basename(os.path.normpath(d))
        args = model_args(d)
        model = build_model(args, device)
        out_dir = os.path.join(a.out, name)
        os.makedirs(out_dir, exist_ok=True)
        result = {
            "weights": args.model_weights,
            "real": real_part(model, args, device, out_dir, extents),
            "synthetic": synthetic_part(model, args, device, extents),
        }
        with open(os.path.join(out_dir, "diag.json"), "w") as f:
            json.dump(result, f, indent=2)
        print(name, json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
