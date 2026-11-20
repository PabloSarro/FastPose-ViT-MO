# This file evaluates a trained FastPoseViT on a BOP split with ground-truth boxes: per-instance
# rotation (deg), translation (mm), ADD, MSSD and MSPD errors, plus a BOP19-style results CSV.
#
# The metrics are computed per instance with bop_toolkit_lib.pose_error, per object, and gathered
# in a per-object table (table_<split>.md). They are not the BOP19 AR, which needs VSD and the
# renderer: run scripts/bop_eval.sh on the CSV, which holds every evaluated object.
#
# BOP19 results CSV (M6): OUT_DIR/<method>_<dataset>-<split>.csv, method "fastposevitmo" by default (BOP
# method names have no "_"), columns scene_id,im_id,obj_id,score,R,t,time with R row-major and t in
# mm. `time` is BOP's per-image time (all objects of the image), measured in a separate timed pass
# (see time_per_image); the poses come from the normal batched pass.
#
# Real detections (M7): --detections FILE (BOP default detections, COCO-style JSON) crops around
# the top inst_count detections by score of each target (image, object) instead of the GT box;
# the CSV then has the detection scores, no estimate where nothing was detected (a miss for BOP),
# and time = ours + the detector's time of the image. In the per-object tables a missing
# detection counts as a failure.
#
# --dataset lmo (default) or ycbv (M8): objects, models and index from src/bop_dataset.py SPLITS.
#
# --backend trt --trt_engine FILE (M9): the network runs as a TensorRT engine built by
# optimization/bop_trt.py instead of PyTorch; everything else (crops, R, t recovery, timing) is
# the same. --model_weights is then not needed.

import argparse
import json
import logging
import os
import time

import numpy as np
import torch
from bop_toolkit_lib import inout, misc, pose_error
from torch.amp import autocast
from torch.utils.data import DataLoader

from src.safe_load import load_safe
from src.bop_dataset import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    SPLITS,
    BOPDataset,
    bop_bbox_to_xyxy,
    crop_image,
    default_bop_root,
    default_cache_dir,
    load_index,
    load_models_info,
    object_extents,
    rows_for_objects,
    select_rows,
)
from src.engine import CoordinateTransformer, forward_dense, forward_model
from src.models import OBJECT_CONDITIONINGS, VIT_MODELS, FastPoseViT
from src.pnp import POSE_MODES, PoseRefiner
from src.refiner import RefinerNet, refine_instances
from src.render import PointRenderer, load_render_assets
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
        num_objects=len(SPLITS[args.dataset]["obj_ids"]),  # as trained: one per dataset object
        object_conditioning=args.object_conditioning,
        dense_heads=getattr(args, "dense_heads", False),  # M18; the decoder is in the weights
        dense_size=getattr(args, "dense_size", 64),
    )
    # model.pth is a state dict; last.pt (train.py checkpoint) holds it under "model"
    state = load_safe(args.model_weights)  # weights_only + data-only allowlist (audit W9)
    model.load_state_dict(state["model"] if "model" in state else state)
    return model.to(device).eval()


_REFINE_CACHE: dict = {}


def make_refine_fn(args, dataset, device):
    """The M19 render-and-compare refinement of the arguments as a function
    f(image, K, obj_idx, R, t) -> (R, t) for the instances of one image (None without
    --refine_iterations). The network, its weights and the renderer are built once."""
    n = getattr(args, "refine_iterations", 0)
    if n <= 0:
        return None
    key = (args.refine_weights, str(device))
    if key not in _REFINE_CACHE:
        cfg = SPLITS[args.dataset]
        obj_ids = list(cfg["obj_ids"])
        bop_root = default_bop_root()
        model = RefinerNet(args.refine_vit_model, num_objects=len(obj_ids))
        state = load_safe(args.refine_weights)
        model.load_state_dict(state["model"] if "model" in state else state)
        model = model.to(device).eval()
        assets = load_render_assets(bop_root, args.dataset, obj_ids, args.refine_n_points, default_cache_dir())
        centre, half = object_extents(bop_root, cfg["models_dataset"], obj_ids)
        _REFINE_CACHE[key] = (model, PointRenderer(assets, centre, half, device), assets["points"][:, :2000])
    model, renderer, points_box = _REFINE_CACHE[key]

    def refine(image, K, obj_idx, R, t):
        steps = refine_instances(
            model, renderer, points_box, image, K, obj_idx, R, t, n,
            args.refine_size, args.refine_margin, device,
        )
        return steps[-1]

    return refine


def make_refiner(args, dataset) -> PoseRefiner | None:
    """The M18 pose mode of the arguments as a PoseRefiner (None for "dr", the plain model)."""
    mode = getattr(args, "pose_mode", "dr")
    if mode == "dr":
        return None
    centre, half = object_extents(
        default_bop_root(),
        SPLITS[args.dataset]["models_dataset"],
        [int(o) for o in dataset.index["obj_ids"]],
    )
    return PoseRefiner(mode, centre, half, getattr(args, "crop_scale", 1.0))


@torch.no_grad()
def predict(model, dataset, args, device) -> dict:
    """Absolute poses (R, t in mm) predicted for every row of `dataset`, in row order.

    With a dense pose mode (--pose_mode other than dr, M18) the poses are the refiner's, and the
    result has "fallback" too: where the mode failed and the DR pose was kept."""
    refiner = make_refiner(args, dataset)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False
    )
    transformer = CoordinateTransformer(
        rotation_format="matrix",
        val_camera=dataset.camera,
        no_rotation_compensation=getattr(args, "no_rotation_compensation", False),
        depth_mode=args.depth_mode,
    )
    out = {"R": [], "t": [], "fallback": []}
    for batch in loader:
        images = batch["images"].to(device)
        obj_idx = batch["obj_idx"].to(device)
        with autocast("cuda", enabled=device.type == "cuda"):
            if refiner is not None:
                pred_t, pred_R, dense = forward_dense(model, images, obj_idx, merge_outputs=True)
            else:
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
        R, t = quaternion_to_rotation_matrix(abs_q).cpu().numpy(), abs_t.cpu().numpy()
        fallback = np.zeros(len(R), dtype=bool)
        if refiner is not None:
            R, t, fallback = refiner(
                dense.float().cpu().numpy(),
                batch["bboxes"].numpy(),
                batch["K"].numpy(),
                batch["obj_idx"].numpy(),
                R,
                t,
            )
        out["R"].append(R)
        out["t"].append(t)
        out["fallback"].append(fallback)
    result = {k: np.concatenate(v) for k, v in out.items()}
    refine = make_refine_fn(args, dataset, device)
    if refine is not None:  # M19: one image at a time, all its instances together
        index = dataset.index
        groups: dict[tuple, list[int]] = {}
        for row in range(len(dataset)):
            groups.setdefault((int(index["scene_id"][row]), int(index["im_id"][row])), []).append(row)
        for rows in groups.values():
            result["R"][rows], result["t"][rows] = refine(
                dataset.load_image(rows[0]), index["K"][rows[0]].astype(np.float64),
                index["obj_idx"][rows], result["R"][rows], result["t"][rows],
            )
    return result


def noisy_boxes(bbox_xywh: np.ndarray, img_wh: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    """BOP boxes [x, y, w, h] with every side moved by up to `fraction` of the box size
    (uniform, as the training jitter, deterministic per seed), clamped to the image."""
    rng = np.random.default_rng(seed)
    b = np.asarray(bbox_xywh, dtype=np.float64)
    n = len(b)
    u = rng.uniform(-1, 1, (n, 4))
    x1, y1 = b[:, 0] + u[:, 0] * fraction * b[:, 2], b[:, 1] + u[:, 2] * fraction * b[:, 3]
    x2 = b[:, 0] + b[:, 2] - u[:, 1] * fraction * b[:, 2]
    y2 = b[:, 1] + b[:, 3] - u[:, 3] * fraction * b[:, 3]
    W, H = np.asarray(img_wh, dtype=np.float64)[:, 0], np.asarray(img_wh, dtype=np.float64)[:, 1]
    x1, x2 = np.clip(x1, 0, W), np.clip(x2, 0, W)
    y1, y2 = np.clip(y1, 0, H), np.clip(y2, 0, H)
    out = np.stack([x1, y1, np.maximum(x2 - x1, 1.0), np.maximum(y2 - y1, 1.0)], axis=1)
    return out.astype(np.asarray(bbox_xywh).dtype)


def noisy_boxes_for_rows(index: dict, rows, fraction: float, seed: int) -> np.ndarray:
    """noisy_boxes drawn over the whole split, then taken at `rows`: an instance gets the same
    box whatever subset of objects is evaluated (M16: single-object models vs the shared one).
    With all objects (rows = every row) this equals noisy_boxes on the split."""
    return noisy_boxes(index["bbox_visib"], index["img_wh"], fraction, seed)[np.asarray(rows)]


def pose_transformer(dataset, args) -> CoordinateTransformer:
    """R, t recovery from the network outputs, as in predict()."""
    return CoordinateTransformer(
        rotation_format="matrix",
        val_camera=dataset.camera,
        no_rotation_compensation=getattr(args, "no_rotation_compensation", False),
        depth_mode=args.depth_mode,
    )


def normalize_crops(crops: np.ndarray, device) -> torch.Tensor:
    """uint8 crops (N, S, S, 3) -> network input (N, 3, S, S) on `device`, normalised as
    pixel_transform("none") (/255, ImageNet mean/std), but computed on the device."""
    images = torch.from_numpy(crops).to(device).permute(0, 3, 1, 2).float().div_(255.0)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return ((images - mean) / std).contiguous()


def estimate_image(model, dataset, transformer, rows, image, device, refiner=None, refine=None):
    """Poses of the instances `rows` of one decoded image: (R (N, 3, 3), t (N, 3) mm) on the CPU.

    The whole per-image path: crop and normalisation of every instance, the network on that
    batch, and R, t recovery. Used for the BOP `time` (time_per_image) and the M9 latency
    benchmark (optimization/bop_latency.py). With a `refiner` (M18, make_refiner) the network
    also returns its dense outputs and the poses are the refiner's, PnP time included.
    """
    if dataset.augment:
        raise ValueError("estimate_image needs a dataset without augmentation")
    index = dataset.index
    bboxes = [dataset.crop_bbox(r) for r in rows]
    # Crops (uint8) on the CPU, normalisation on the GPU: the same ImageNet normalisation as
    # BOPDataset's transform, but its CPU version cost ~2 ms per crop (M9 latency)
    crops = np.stack(
        [
            crop_image(image, b, dataset.img_size, dataset.crop_context, dataset.crop_scale)
            for b in bboxes
        ]
    )
    images = normalize_crops(crops, device)
    obj_idx = torch.as_tensor(index["obj_idx"][rows], device=device)
    dense = None
    with autocast("cuda", enabled=device.type == "cuda"):
        if refiner is not None:
            pred_t, pred_R, dense = forward_dense(model, images, obj_idx, merge_outputs=True)
        else:
            pred_t, pred_R = forward_model(model, images, obj_idx, merge_outputs=True)
    # R, t recovery on the CPU, in fp32: N x 9 values, and on the GPU its masked ops synced it
    # ~66 times per image (~9 ms, M9 latency)
    pred_t, pred_R = pred_t.float().cpu(), pred_R.float().cpu()
    pred_R = process_rotation(pred_R, "matrix")
    n = len(rows)
    abs_t, abs_q, _, _ = transformer.transform_to_absolute(
        pred_t,
        pred_R,
        torch.zeros(n, 3),
        torch.zeros(n, 3, 3),
        torch.tensor(np.array(bboxes), dtype=torch.float32),
        K=torch.from_numpy(index["K"][rows]).float(),
        img_wh=torch.from_numpy(index["img_wh"][rows]).float(),
        diameter=torch.from_numpy(index["diameters"][index["obj_idx"][rows]]).float(),
    )
    R, t = quaternion_to_rotation_matrix(abs_q).numpy(), abs_t.numpy()
    if refiner is not None:
        R, t, _ = refiner(
            dense.float().cpu().numpy(),
            np.array(bboxes),
            index["K"][rows],
            index["obj_idx"][rows],
            R,
            t,
        )
    if refine is not None:  # M19 render-and-compare refinement, its time included
        R, t = refine(image, index["K"][rows[0]].astype(np.float64), index["obj_idx"][rows], R, t)
        R, t = np.asarray(R), np.asarray(t)
    return R, t


def time_per_image(model, dataset, args, device, warmup: int = 3) -> dict:
    """BOP19 `time` per image: seconds to estimate the poses of all its objects (GT boxes).

    Timed, for each image, from its decoded RGB in memory to the poses on the CPU
    (estimate_image, with a CUDA sync). Reading and decoding the image file is not timed (it
    measures the NAS, not the method). The first `warmup` images are run once untimed first
    (cuDNN autotuning, allocations).

    Returns:
        {(scene_id, im_id): seconds}
    """
    index = dataset.index
    groups: dict[tuple, list[int]] = {}
    for row in range(len(dataset)):
        groups.setdefault((int(index["scene_id"][row]), int(index["im_id"][row])), []).append(row)
    transformer = pose_transformer(dataset, args)
    refiner = make_refiner(args, dataset)
    refine = make_refine_fn(args, dataset, device)

    def estimate(rows, image):
        return estimate_image(model, dataset, transformer, rows, image, device, refiner, refine)

    times = {}
    model.eval()
    with torch.no_grad():
        keys = list(groups)
        for key in keys[:warmup]:
            estimate(groups[key], dataset.load_image(groups[key][0]))
        for key in keys:
            rows = groups[key]
            image = dataset.load_image(rows[0])
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            estimate(rows, image)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times[key] = time.perf_counter() - start
    return times


def bop_iou(a, b) -> float:
    """IoU of two [x1, y1, x2, y2] boxes."""
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h
    return w * h / union if union > 0 else 0.0


def assign_detections(index: dict, detections_path: str) -> dict:
    """Detection boxes for the rows (GT instances = targets) of `index`.

    For each (scene, image, object), the top inst_count detections by score (inst_count = its
    number of rows), matched to the instances greedily by IoU with bbox_visib (only for our
    per-instance errors: BOP matches by itself). Rows without a detection get NaN.

    Returns:
        bbox: (n, 4) BOP-format [x, y, w, h] floats (bop_bbox_to_xyxy gives the detection box
        back exactly), score: (n,), det_time: {(scene_id, im_id): detector seconds}
    """
    with open(detections_path) as f:
        dets = json.load(f)
    by_key: dict[tuple, list] = {}
    det_time: dict[tuple, float] = {}
    for d in dets:
        x, y, w, h = d["bbox"]
        by_key.setdefault((d["scene_id"], d["image_id"], d["category_id"]), []).append(
            (d["score"], [x, y, x + w, y + h])
        )
        key = (d["scene_id"], d["image_id"])
        det_time[key] = max(det_time.get(key, 0.0), float(d.get("time", 0.0)))
    n = len(index["obj_id"])
    bbox = np.full((n, 4), np.nan)
    score = np.full(n, np.nan)
    groups: dict[tuple, list[int]] = {}
    for row in range(n):
        key = (int(index["scene_id"][row]), int(index["im_id"][row]), int(index["obj_id"][row]))
        groups.setdefault(key, []).append(row)
    for key, rows in groups.items():
        kept = sorted(by_key.get(key, []), key=lambda c: -c[0])[: len(rows)]
        for row in sorted(rows, key=lambda r: -float(index["visib_fract"][r])):
            if not kept:
                break
            gt = bop_bbox_to_xyxy(index["bbox_visib"][row])
            j = max(range(len(kept)), key=lambda j: bop_iou(kept[j][1], gt))
            sc, (x1, y1, x2, y2) = kept.pop(j)
            bbox[row] = [x1, y1, x2 - x1 - 1.0, y2 - y1 - 1.0]  # inverse of bop_bbox_to_xyxy
            score[row] = sc
    return {"bbox": bbox, "score": score, "det_time": det_time}


def rotation_error_deg(R_est: np.ndarray, R_gt: np.ndarray) -> float:
    cos = (np.trace(R_est.T @ R_gt) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def instance_errors(
    index: dict, pred: dict, obj_id: int, bop_root: str, models_dataset: str = "lmo"
) -> dict:
    """Per-instance errors of one object (models from <bop_root>/<models_dataset>/models_eval).

    For a symmetric object (LM-O eggbox 10, glue 11), "add" is ADD-S, "rot_deg" is the smallest
    angle to any symmetric GT pose, and MSSD/MSPD use the BOP19 symmetry set (models_info.json,
    continuous symmetries discretised at 1 % of the diameter). "trans_mm" is always plain.
    """
    obj_idx = int(np.flatnonzero(index["obj_ids"] == obj_id)[0])
    diameter = float(index["diameters"][obj_idx])
    models_dir = os.path.join(bop_root, models_dataset, "models_eval")
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


# LM-O objects evaluated with symmetries (ADD-S): eggbox and glue (models_info.json)
LMO_SYMMETRIC_IDS = (10, 11)

# Per-object table: (summary key, column title, format)
TABLE_COLUMNS = (
    ("n", "n", "{:d}"),
    ("ADD@0.1d", "ADD(-S)@0.1d", "{:.3f}"),
    ("ADD@0.05d", "ADD(-S)@0.05d", "{:.3f}"),
    ("recall_MSSD", "MSSD recall", "{:.3f}"),
    ("recall_MSPD", "MSPD recall", "{:.3f}"),
    ("trans_mm_mean", "trans mean (mm)", "{:.1f}"),
    ("trans_mm_median", "trans median (mm)", "{:.1f}"),
    ("rot_deg_mean", "rot mean (°)", "{:.2f}"),
    ("rot_lt_5deg", "rot < 5°", "{:.3f}"),
)


def per_object_table(per_obj: dict, symmetric_ids=()) -> tuple[str, dict]:
    """Markdown table of per-object summaries, plus the mean over objects and over instances.

    The mean over objects is the LM literature's average; the mean over instances weights each
    object by its number of instances (as BOP19 recalls do). Returns (markdown, the two means).
    """
    keys = [k for k, _, _ in TABLE_COLUMNS if k != "n"]
    n = np.array([per_obj[o]["n"] for o in per_obj], dtype=np.float64)
    # Objects without instances in the split (e.g. the held-out YCB-V real videos) have nan
    # entries: they count in neither mean.
    present = [o for o in per_obj if per_obj[o]["n"] > 0] or list(per_obj)
    weights = np.array([per_obj[o]["n"] for o in present], dtype=np.float64)
    means = {
        "mean over objects": {k: float(np.mean([per_obj[o][k] for o in present])) for k in keys},
        "mean over instances": {
            k: float(np.average([per_obj[o][k] for o in present], weights=weights if weights.sum() else None))
            for k in keys
        },
    }
    for m in means.values():
        m["n"] = int(n.sum())
    lines = [
        "| object | " + " | ".join(t for _, t, _ in TABLE_COLUMNS) + " |",
        "|---" * (len(TABLE_COLUMNS) + 1) + "|",
    ]
    rows = [(f"{o}{' (sym.)' if int(o) in symmetric_ids else ''}", per_obj[o]) for o in per_obj]
    rows += [(f"**{name}**", m) for name, m in means.items()]
    for name, d in rows:
        lines.append(
            f"| {name} | " + " | ".join(f.format(d[k]) for k, _, f in TABLE_COLUMNS) + " |"
        )
    return "\n".join(lines) + "\n", means


def boxes_label(args) -> str:
    """Table title part: where the crops came from."""
    if args.detections:
        return f"detections {os.path.basename(args.detections)}"
    return "GT boxes"


def write_tables(
    out_dir: str, splits, symmetric_ids=LMO_SYMMETRIC_IDS, boxes: str = "GT boxes"
) -> dict:
    """table_<split>.md and summary.json from the per-object obj_<id>/summary.json files.

    symmetric_ids: objects marked "(sym.)" (evaluated with ADD-S), default LM-O's.
    boxes: what the crops came from, for the table title ("GT boxes" or the detections file).
    """
    obj_dirs = sorted(
        (d for d in os.listdir(out_dir) if d.startswith("obj_")), key=lambda d: int(d[4:])
    )
    per_obj = {}
    for d in obj_dirs:
        with open(os.path.join(out_dir, d, "summary.json")) as f:
            per_obj[d[4:]] = json.load(f)
    summary = {}
    for split in splits:
        objs = {o: v[split] for o, v in per_obj.items() if split in v}
        table, means = per_object_table(objs, set(symmetric_ids))
        with open(os.path.join(out_dir, f"table_{split}.md"), "w") as f:
            f.write(f"{split}, {boxes}; ADD-S for symmetric objects\n\n{table}")
        logging.info(f"{split}:\n{table}")
        summary[split] = {**objs, **means}
    summary["note"] = VSD_SKIPPED
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate FastPoseViT on BOP (GT boxes)")
    parser.add_argument("--model_weights", help="model.pth or last.pt (not with --table_only)")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--dataset", default="lmo", choices=sorted(SPLITS))
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Default: val_real test (lmo), val_pbr test (ycbv, which has no real val)",
    )
    parser.add_argument(
        "--obj_ids",
        "--obj_id",
        type=int,
        nargs="+",
        default=None,
        help="BOP object ids (default: all objects of the dataset); results in OUT_DIR/obj_<id>/",
    )
    parser.add_argument(
        "--table_only",
        action="store_true",
        help="Only rebuild table_<split>.md and summary.json from existing OUT_DIR/obj_<id>/",
    )
    parser.add_argument("--vit_model", default="vit_b_16_384")
    parser.add_argument("--depth_mode", default="site")
    parser.add_argument(
        "--object_conditioning", default="heads", choices=OBJECT_CONDITIONINGS, help="as train.py"
    )
    parser.add_argument(
        "--dense_heads",
        action="store_true",
        help="M18: the weights come from train.py --dense_heads (needed for --pose_mode != dr)",
    )
    parser.add_argument("--dense_size", type=int, default=64, help="as train.py")
    parser.add_argument(
        "--pose_mode",
        default="dr",
        choices=POSE_MODES,
        help="M18: dr = direct regression; pnp = PnP-RANSAC on the predicted mask + XYZ; "
        "pnp_init = LM from the DR pose; dr_rot_pnp_t = DR rotation + PnP translation. A mode "
        "that fails for an instance keeps its DR pose (counted in pose_mode.json)",
    )
    parser.add_argument("--refine_weights", default=None, help="M19: the refiner (src/train_refiner.py last.pt / model.pth)")
    parser.add_argument("--refine_iterations", type=int, default=0, help="M19: refinement iterations after the pose (0 = none)")
    parser.add_argument("--refine_vit_model", default="convnext_base")
    parser.add_argument("--refine_size", type=int, default=256)
    parser.add_argument("--refine_margin", type=float, default=1.4)
    parser.add_argument("--refine_n_points", type=int, default=65536)
    parser.add_argument(
        "--bbox_source",
        default="visib",
        choices=["visib", "obj"],
        help="GT box to crop around: bbox_visib (as in training) or bbox_obj (amodal, M7)",
    )
    parser.add_argument(
        "--detections",
        default=None,
        help="Crop around these detections (BOP default detections JSON) instead of GT boxes; "
        "one split only (M7)",
    )
    parser.add_argument(
        "--no_rotation_compensation",
        action="store_true",
        help="as train.py: the model regresses the absolute (egocentric) rotation, not the apparent one",
    )
    parser.add_argument(
        "--crop_context", action="store_true", help="as train.py: image around the box, not black"
    )
    parser.add_argument(
        "--crop_scale", type=float, default=1.0, help="as train.py: crop side / max(w, h) (M11)"
    )
    parser.add_argument(
        "--box_noise", type=float, default=0.0,
        help="Audit W2: perturb the GT crop boxes like a detector would (each side by up to this "
        "fraction of the box size, uniform, fixed seed) so that validation sees box errors; "
        "GT boxes only",
    )
    parser.add_argument("--box_noise_seed", type=int, default=0)
    parser.add_argument(
        "--backend", default="torch", choices=["torch", "trt"], help="trt: TensorRT engine (M9)"
    )
    parser.add_argument("--trt_engine", default=None, help="engine file for --backend trt")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--method", default="fastposevitmo", help="BOP method name (no '_')")
    args = parser.parse_args()
    cfg = SPLITS[args.dataset]
    if args.splits is None:
        args.splits = ["val_real", "test"] if "val_real" in cfg else ["val_pbr", "test"]
    if args.obj_ids is None:
        args.obj_ids = list(cfg["obj_ids"])
    bad_splits = [s for s in args.splits if s not in cfg or s in ("obj_ids", "models_dataset")]
    if bad_splits:
        parser.error(f"--splits {bad_splits} do not exist for {args.dataset}")
    bad_objs = sorted(set(args.obj_ids) - set(cfg["obj_ids"]))
    if bad_objs:
        parser.error(f"--obj_ids {bad_objs} are not objects of {args.dataset}")
    if args.detections and len(args.splits) != 1:
        parser.error("--detections needs exactly one split (the detections file's)")
    if "_" in args.method:
        parser.error("--method must not contain '_' (BOP file names are METHOD_DATASET-SPLIT.csv)")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True
    )  # force: an imported module may have configured logging already
    models = load_models_info(default_bop_root(), cfg["models_dataset"], cfg["obj_ids"])
    symmetric_ids = [o for o, sym in zip(cfg["obj_ids"], models["symmetric"]) if sym]
    if args.table_only:
        write_tables(args.out_dir, args.splits, symmetric_ids, boxes_label(args))
        return
    if args.refine_iterations > 0 and not args.refine_weights:
        parser.error("--refine_iterations needs --refine_weights")
    if args.pose_mode != "dr" and not args.dense_heads:
        parser.error("--pose_mode other than dr needs --dense_heads")
    if args.dense_heads and args.backend == "trt":
        parser.error("--dense_heads is not supported by the TensorRT backend")
    if args.backend == "trt" and not args.trt_engine:
        parser.error("--backend trt needs --trt_engine")
    if args.backend == "torch" and not args.model_weights:
        parser.error("--model_weights is required (unless --table_only or --backend trt)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.backend == "trt":
        from optimization.bop_trt import TRTPoseModel  # TensorRT only needed here

        model = TRTPoseModel(args.trt_engine)  # larger batches run in chunks of its profile
    else:
        model = build_model(args, device)
    bop_root = default_bop_root()
    os.makedirs(args.out_dir, exist_ok=True)

    per_obj = {o: {} for o in args.obj_ids}
    fallbacks = {}
    for split in args.splits:
        index = load_index(args.dataset, split)
        # One pass over all requested objects, then the errors object by object
        all_rows = rows_for_objects(index, args.obj_ids)
        split_index = select_rows(index, all_rows)
        n_rows = len(split_index["obj_id"])
        detected = np.ones(n_rows, dtype=bool)
        pred_index, bbox_source, score = split_index, args.bbox_source, np.ones(n_rows)
        if args.detections:
            det = assign_detections(split_index, args.detections)
            detected = ~np.isnan(det["score"])
            score = det["score"]
            pred_index = select_rows(split_index, np.flatnonzero(detected))
            pred_index["bbox_visib"] = det["bbox"][detected]  # crop around the detections
            bbox_source = "visib"
            logging.info(
                f"{split}: detections for {detected.sum()} of {n_rows} targets "
                f"({args.detections})"
            )
        if args.box_noise > 0:
            if args.detections or bbox_source != "visib":
                parser.error("--box_noise needs GT boxes (bbox_source visib, no --detections)")
            pred_index = dict(pred_index)
            pred_index["bbox_visib"] = noisy_boxes_for_rows(
                index, all_rows, args.box_noise, args.box_noise_seed
            )
        dataset = BOPDataset(
            index=pred_index,
            augment=False,
            bbox_source=bbox_source,
            depth_mode=args.depth_mode,
            no_rotation_compensation=args.no_rotation_compensation,
            img_size=VIT_MODELS[args.vit_model][3][0],
            crop_context=getattr(args, "crop_context", False),
            crop_scale=getattr(args, "crop_scale", 1.0),
        )
        pred_det = predict(model, dataset, args, device)
        if args.pose_mode != "dr":
            n_fail = int(pred_det["fallback"].sum())
            fallbacks[split] = {"instances": len(pred_det["R"]), "fallback_to_dr": n_fail}
            logging.info(
                f"{split}: pose mode {args.pose_mode}: {n_fail} of {len(pred_det['R'])} "
                "instances fell back to the DR pose"
            )
        # Targets without a detection: a pose 10 km away, so every error counts as a failure
        pred_all = {
            "R": np.tile(np.eye(3, dtype=pred_det["R"].dtype), (n_rows, 1, 1)),
            "t": np.tile(np.array([0.0, 0.0, 1e7], dtype=pred_det["t"].dtype), (n_rows, 1)),
        }
        for k in pred_all:
            pred_all[k][detected] = pred_det[k]
        image_times = time_per_image(model, dataset, args, device)
        if args.detections:
            image_times = {k: v + det["det_time"].get(k, 0.0) for k, v in image_times.items()}
        logging.info(
            f"{split}: {len(image_times)} images, time per image (BOP 'time', see time_per_image) "
            f"mean {np.mean(list(image_times.values())) * 1000:.1f} ms, "
            f"median {np.median(list(image_times.values())) * 1000:.1f} ms"
        )
        results = []
        for obj_id in args.obj_ids:
            rows = rows_for_objects(split_index, [obj_id])
            sub = select_rows(split_index, rows)
            pred = {k: v[rows] for k, v in pred_all.items()}
            errs = instance_errors(sub, pred, obj_id, bop_root, cfg["models_dataset"])
            per_obj[obj_id][split] = summarize(errs)

            obj_dir = os.path.join(args.out_dir, f"obj_{obj_id}")
            os.makedirs(obj_dir, exist_ok=True)
            np.savez(
                os.path.join(obj_dir, f"errors_{split}.npz"),
                scene_id=sub["scene_id"],
                im_id=sub["im_id"],
                gt_id=sub["gt_id"],
                R_est=pred["R"],
                t_est=pred["t"],
                **{k: v for k, v in errs.items() if k not in ("diameter", "img_w")},
            )
            results += [
                {
                    "scene_id": int(sub["scene_id"][i]),
                    "im_id": int(sub["im_id"][i]),
                    "obj_id": obj_id,
                    "score": float(score[rows[i]]),
                    "R": pred["R"][i],
                    "t": pred["t"][i],
                    "time": image_times[(int(sub["scene_id"][i]), int(sub["im_id"][i]))],
                }
                for i in range(len(rows))
                if detected[rows[i]]
            ]
        # One BOP results file per split with every evaluated object (scripts/bop_eval.sh)
        inout.save_bop_results(
            os.path.join(args.out_dir, f"{args.method}_{args.dataset}-{split}.csv"), results
        )

    for obj_id, d in per_obj.items():
        with open(os.path.join(args.out_dir, f"obj_{obj_id}", "summary.json"), "w") as f:
            json.dump(d, f, indent=2)
    with open(os.path.join(args.out_dir, "pose_mode.json"), "w") as f:
        json.dump({"pose_mode": args.pose_mode, "splits": fallbacks}, f, indent=2)
    write_tables(args.out_dir, args.splits, symmetric_ids, boxes_label(args))


if __name__ == "__main__":
    main()
