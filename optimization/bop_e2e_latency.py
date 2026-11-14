# Audit W5 (TASKS.md): true end-to-end latency, from the encoded image file in memory to the poses.
#
# The earlier latency tables (optimization/bop_latency.py) time the pose pipeline on one repeated
# image from the decoded array and add the detector's time measured separately. Here a real loop
# runs over every image of a BOP test set, with the same protocol as the AR evaluation:
#   1. decode the PNG / JPEG (bytes already read: reading the NAS is not part of the method),
#   2. detector (our multi-class LW-DETR TensorRT engine: resize, normalise, network, top-k),
#   3. per target object, its best detection (BOP "targets" protocol, as src/bop_evaluate.py),
#   4. crops, pose network (TensorRT engine or PyTorch), R, t recovery (estimate_image),
#   5. optionally the RGB-D depth correction (decode the 16-bit depth PNG + scripts/depth_correction.py).
# The total per image is timed with one CUDA sync at the end; a second pass with a sync after
# every stage gives the breakdown. The estimates are written as a BOP CSV with this time, so the
# official evaluation (scripts/bop_eval.sh) reports the AR of exactly this pipeline.
#
#   PYTHONPATH=. .venv/bin/python optimization/bop_e2e_latency.py --dataset lmo \
#       --pose_engine RUN/model_fp16.engine --det_engine OUT/lwdetr_small_fp16.engine \
#       --out_dir OUT/e2e [--rgbd] [--n_images 200]

import argparse
import json
import logging
import os
import time

import cv2
import numpy as np
import torch

from object_detector.bop_detector import BOPLWDETR
from optimization.bop_det_trt import TRTDetector
from optimization.bop_trt import TRTPoseModel
from scripts.depth_correction import SplatCorrector
from src.bop_dataset import SPLITS, BOPDataset, default_bop_root, image_path, load_index
from src.bop_evaluate import build_model, estimate_image, pose_transformer
from src.models import VIT_MODELS


def image_groups(index) -> dict:
    groups: dict = {}
    for row in range(len(index["obj_id"])):
        groups.setdefault((int(index["scene_id"][row]), int(index["im_id"][row])), []).append(row)
    return groups


def assign(rows, obj_ids, dets):
    """Per row (target instance), the best detection of its object, BOP targets protocol: the
    k-th best of an object serves its k-th target. Rows without a detection are dropped."""
    by_obj: dict = {}
    for d in sorted(dets, key=lambda d: -d["score"]):
        by_obj.setdefault(d["category_id"], []).append(d)
    used: dict = {}
    keep_rows, boxes, scores = [], [], []
    for r, o in zip(rows, obj_ids):
        k = used.get(o, 0)
        if k < len(by_obj.get(o, [])):
            d = by_obj[o][k]
            used[o] = k + 1
            keep_rows.append(r)
            boxes.append(d["bbox"])
            scores.append(d["score"])
    return keep_rows, boxes, scores


def depth_path(rgb_path: str) -> str:
    folder, name = os.path.split(rgb_path)
    return os.path.join(os.path.dirname(folder), "depth", os.path.splitext(name)[0] + ".png")


def stats_ms(values) -> dict:
    v = 1000 * np.asarray(values)
    return {"mean": float(v.mean()), "median": float(np.median(v)),
            "p90": float(np.percentile(v, 90)), "max": float(v.max())}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="lmo", choices=sorted(SPLITS))
    p.add_argument("--pose_engine", default=None, help="TensorRT engine of the pose network")
    p.add_argument("--model_weights", default=None, help="PyTorch pose weights instead of an engine")
    p.add_argument("--vit_model", default="vit_b_16_384")
    p.add_argument("--depth_mode", default="site")
    p.add_argument("--object_conditioning", default="token")
    p.add_argument("--det_engine", required=True, help="TensorRT engine of the detector")
    p.add_argument("--det_variant", default="small")
    p.add_argument("--det_img_size", type=int, default=640)
    p.add_argument("--rgbd", action="store_true", help="add the depth correction")
    p.add_argument("--n_images", type=int, default=0, help="0 = all images of the test set")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    os.makedirs(args.out_dir, exist_ok=True)
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    ids = SPLITS[args.dataset]["obj_ids"]
    index = load_index(args.dataset, "test")
    if args.pose_engine:
        pose_model = TRTPoseModel(args.pose_engine)
    elif args.model_weights:
        args.dataset_for_build = args.dataset
        pose_model = build_model(args, dev)
    else:
        p.error("--pose_engine or --model_weights is required")
    dataset = BOPDataset(index=index, augment=False, bbox_source="visib",
                         depth_mode=args.depth_mode, img_size=VIT_MODELS[args.vit_model][3][0])
    transformer = pose_transformer(dataset, args)
    post = BOPLWDETR(variant=args.det_variant, num_classes=len(ids))._lwdetr_post["bbox"]
    detector = TRTDetector(args.det_engine, post, args.det_img_size, ids, 100)
    root = default_bop_root()
    corrector = None
    if args.rgbd:
        models_ds = SPLITS[args.dataset].get("models_dataset", args.dataset)
        corrector = SplatCorrector(root, models_ds, ids, dev)
    cams = {}

    groups = image_groups(index)
    keys = sorted(groups)[: args.n_images or None]
    # Encoded files in memory (reading the NAS is not timed); depth scale per image
    data = {}
    for key in keys:
        row = groups[key][0]
        path = image_path(index, row)
        with open(path, "rb") as f:
            rgb = np.frombuffer(f.read(), np.uint8)
        depth, scale = None, 1.0
        if args.rgbd:
            with open(depth_path(path), "rb") as f:
                depth = np.frombuffer(f.read(), np.uint8)
            sc = key[0]
            if sc not in cams:
                sdir = os.path.dirname(os.path.dirname(path))
                with open(os.path.join(sdir, "scene_camera.json")) as f:
                    cams[sc] = json.load(f)
            scale = float(cams[sc][str(key[1])]["depth_scale"])
        data[key] = (rgb, depth, scale)

    def run(key, stages=None):
        """One image, from its encoded bytes to the poses; stages: dict to fill with times."""
        rows = groups[key]
        rgb_bytes, depth_bytes, scale = data[key]

        def mark(name, t0):
            if stages is not None:
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                stages[name] = stages.get(name, 0.0) + (t1 - t0)
                return t1
            return t0

        t = time.perf_counter()
        image = cv2.cvtColor(cv2.imdecode(rgb_bytes, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        t = mark("decode", t)
        dets = detector(image)
        t = mark("detector", t)
        keep, boxes, _ = assign(rows, [int(index["obj_id"][r]) for r in rows], dets)
        R = tt = None
        if keep:
            index["bbox_visib"][keep] = np.asarray(boxes, dtype=index["bbox_visib"].dtype)
            R, tt = estimate_image(pose_model, dataset, transformer, keep, image, dev)
        t = mark("crops + pose network + R, t", t)
        if corrector is not None and keep:
            d = cv2.imdecode(depth_bytes, cv2.IMREAD_UNCHANGED).astype(np.float32) * scale
            t = mark("decode depth", t)
            dt = torch.as_tensor(d, device=dev)
            K = index["K"][keep[0]]
            tt, _, _ = corrector(
                [int(index["obj_id"][r]) for r in keep], list(R), [x.reshape(3, 1) for x in tt],
                dt, K, [0.2, 0.1, 0.05], 50,
            )
            mark("depth correction", t)
        return keep, R, tt

    for key in keys[: args.warmup]:
        run(key)
    torch.cuda.synchronize()
    total, n_obj, rows_out = [], [], []
    for key in keys:
        torch.cuda.synchronize()
        start = time.perf_counter()
        keep, R, tt = run(key)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        total.append(elapsed)
        n_obj.append(len(keep))
        for r, Ri, ti in zip(keep, R if R is not None else [], tt if tt is not None else []):
            rows_out.append((key, int(index["obj_id"][r]), np.asarray(Ri), np.asarray(ti).reshape(3),
                             elapsed))
    breakdown = {}
    for key in keys:
        run(key, breakdown)
    breakdown = {k: 1000 * v / len(keys) for k, v in breakdown.items()}

    name = "fastposevitmodepth" if args.rgbd else "fastposevitmo"
    csv_path = os.path.join(args.out_dir, f"{name}_{args.dataset}-test.csv")
    with open(csv_path, "w") as f:
        f.write("scene_id,im_id,obj_id,score,R,t,time\n")
        for (sc, im), obj, R, t, el in rows_out:
            f.write(f"{sc},{im},{obj},1.0,{' '.join(map(str, R.flatten()))},"
                    f"{' '.join(map(str, t))},{el}\n")
    n = np.asarray(n_obj)
    tm = np.asarray(total)
    by_n = {int(k): stats_ms(tm[n == k]) | {"images": int((n == k).sum())} for k in sorted(set(n))}
    slope = float(np.polyfit(n, 1000 * tm, 1)[0]) if len(set(n)) > 1 else None
    report = {
        "gpu": torch.cuda.get_device_name(), "cpus": len(os.sched_getaffinity(0)),
        "torch": torch.__version__, "dataset": args.dataset, "rgbd": args.rgbd,
        "images": len(keys), "objects_mean": float(n.mean()), "objects_max": int(n.max()),
        "total_ms": stats_ms(total), "fps_median": 1000 / stats_ms(total)["median"],
        "ms_per_extra_object": slope, "by_objects": by_n,
        "breakdown_ms_mean": breakdown, "pose_engine": args.pose_engine,
        "det_engine": args.det_engine, "csv": csv_path,
    }
    with open(os.path.join(args.out_dir, "e2e_latency.json"), "w") as f:
        json.dump(report, f, indent=2)
    logging.info(json.dumps({k: report[k] for k in ("images", "objects_mean", "total_ms",
                                                    "fps_median", "breakdown_ms_mean")}, indent=1))


if __name__ == "__main__":
    main()
