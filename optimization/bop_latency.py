# M9 bullet 3: latency per image vs number of objects N, PyTorch vs TensorRT engines.
#
# Two measurements, for each N in --n_objects:
#   network:  the network alone on N crops already on the GPU (CUDA events, per call).
#   pipeline: one image, from its decoded RGB in memory to the N poses on the CPU, exactly the
#             evaluator's path (src/bop_evaluate.estimate_image: crop + normalise each instance on
#             the CPU, network, R, t recovery), timed with perf_counter and a CUDA sync. This is
#             the BOP `time` without the detector.
# The detector is not run here: its time is the mean per-image `time` of the BOP default
# detections file (GDRNPPDet, measured by its authors on their GPU), reported apart.
#
# Backends: torch_fp32 (eager, TF32 off), torch_amp (eager, fp16 autocast = how bop_evaluate.py
# runs PyTorch), and each --engine (optimization/bop_trt.py). The crops of the pipeline come from
# one LM-O test image with the most instances (GT boxes), its instances repeated to reach N.
#
# Usage (from the repo root):
#   PYTHONPATH=. .venv/bin/python optimization/bop_latency.py --model_weights RUN/last.pt \
#       --engine OUT/model_fp16.engine OUT/model_fp32.engine --out_dir OUT/latency

import argparse
import glob
import json
import os
import statistics
import time

import numpy as np
import torch
from torch.amp import autocast

from optimization.bop_trt import TRTPoseModel
from src.bop_dataset import BOPDataset, load_index, select_rows
from src.bop_evaluate import build_model, estimate_image, pose_transformer
from src.models import VIT_MODELS

DET_GLOB = (
    "datasets/bop_detections/default_task1/bop23_default_detections_for_task1/gdrnppdet-pbr/"
    "gdrnppdet-pbr_{ds}-test_*.json"
)


class AmpModel:
    """PyTorch model under fp16 autocast (network-only timing; the pipeline adds it itself)."""

    def __init__(self, model):
        self.model = model

    def eval(self):
        return self

    def __call__(self, images, obj_idx):
        with autocast("cuda"):
            return self.model(images, obj_idx)


class Fp32Model(AmpModel):
    """PyTorch model with autocast forced off, also inside estimate_image."""

    def __call__(self, images, obj_idx):
        with autocast("cuda", enabled=False):
            return self.model(images.float(), obj_idx)


def stats_ms(times_s: list[float]) -> dict:
    ms = np.array(times_s) * 1000
    return {
        "median_ms": round(float(np.median(ms)), 3),
        "p90_ms": round(float(np.percentile(ms, 90)), 3),
        "mean_ms": round(float(ms.mean()), 3),
        "min_ms": round(float(ms.min()), 3),
    }


@torch.no_grad()
def time_network(model, n: int, img_size: int, num_objects: int, warmup: int, iters: int):
    images = torch.randn(n, 3, img_size, img_size, device="cuda")
    obj_idx = torch.arange(n, device="cuda") % num_objects
    for _ in range(warmup):
        model(images, obj_idx)
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        model(images, obj_idx)
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / 1000)
    return stats_ms(times)


@torch.no_grad()
def time_pipeline(model, dataset, transformer, rows, image, warmup: int, iters: int):
    device = torch.device("cuda")
    for _ in range(warmup):
        estimate_image(model, dataset, transformer, rows, image, device)
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        estimate_image(model, dataset, transformer, rows, image, device)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return stats_ms(times)


def busiest_image(index) -> list[int]:
    """Rows of the image with the most instances (first one on ties)."""
    keys = list(zip(index["scene_id"].tolist(), index["im_id"].tolist()))
    counts: dict = {}
    for row, key in enumerate(keys):
        counts.setdefault(key, []).append(row)
    return max(counts.values(), key=len)


def markdown(report: dict) -> str:
    backends = list(report["network"])
    ns = [str(n) for n in report["n_objects"]]
    lines = [
        f"Latency per image on {report['gpu']} ({report['cpus']} CPUs), median ms (p90). "
        f"Network = N crops on the GPU; pipeline = decoded image -> N poses "
        f"(crop + network + R, t). Detector (BOP default detections, GDRNPPDet): "
        f"{report['detector_ms']:.1f} ms per image on its authors' GPU.",
        "",
        "| Backend | Part | " + " | ".join(f"N = {n}" for n in ns) + " |",
        "|---|---|" + "---|" * len(ns),
    ]
    for part in ("network", "pipeline"):
        for b in backends:
            if b not in report[part]:
                continue
            cells = [
                f"{report[part][b][n]['median_ms']:.1f} ({report[part][b][n]['p90_ms']:.1f})"
                for n in ns
            ]
            lines.append(f"| {b} | {part} | " + " | ".join(cells) + " |")
    for b in report["pipeline"]:
        cells = [
            f"{1000 / (report['pipeline'][b][n]['median_ms'] + report['detector_ms']):.1f}"
            for n in ns
        ]
        lines.append(f"| {b} | FPS incl. detector | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="M9 latency vs number of objects")
    parser.add_argument("--model_weights", required=True)
    parser.add_argument("--engine", nargs="*", default=[], help="TensorRT engines to time")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--dataset", default="lmo")
    parser.add_argument("--vit_model", default="vit_b_16_384")
    parser.add_argument("--depth_mode", default="site")
    parser.add_argument("--object_conditioning", default="token")
    parser.add_argument("--n_objects", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=200)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    os.makedirs(args.out_dir, exist_ok=True)

    model = build_model(args, torch.device("cuda"))
    backends = {"torch_fp32": Fp32Model(model), "torch_amp": AmpModel(model)}
    for e in args.engine:
        backends["trt_" + os.path.basename(e).replace(".engine", "").replace("model_", "")] = (
            TRTPoseModel(e)
        )
    img_size = VIT_MODELS[args.vit_model][3][0]
    num_objects = model.num_objects

    index = load_index(args.dataset, "test")
    image_rows = busiest_image(index)
    dataset = BOPDataset(
        index=index, augment=False, bbox_source="visib", depth_mode=args.depth_mode,
        img_size=img_size,
    )
    image = dataset.load_image(image_rows[0])
    transformer = pose_transformer(dataset, args)

    det_file = glob.glob(os.path.join(os.environ["NAS_HOME"], DET_GLOB.format(ds=args.dataset)))[0]
    det_times = {(d["scene_id"], d["image_id"]): d["time"] for d in json.load(open(det_file))}

    report = {
        "gpu": torch.cuda.get_device_name(),
        "cpus": len(os.sched_getaffinity(0)),
        "torch": torch.__version__,
        "n_objects": args.n_objects,
        "image": [int(index["scene_id"][image_rows[0]]), int(index["im_id"][image_rows[0]])],
        "image_instances": len(image_rows),
        "detector_ms": 1000 * statistics.mean(det_times.values()),
        "detector_file": det_file,
        "network": {},
        "pipeline": {},
    }
    for name, m in backends.items():
        # The PyTorch amp pipeline is estimate_image as is (it applies autocast itself)
        pipe_model = model if name == "torch_amp" else m
        report["network"][name], report["pipeline"][name] = {}, {}
        for n in args.n_objects:
            # Engines only go up to their profile's max batch in one call (larger: chunks)
            report["network"][name][str(n)] = time_network(
                m, n, img_size, num_objects, args.warmup, args.iters
            )
            rows = [image_rows[i % len(image_rows)] for i in range(n)]
            report["pipeline"][name][str(n)] = time_pipeline(
                pipe_model, dataset, transformer, rows, image, args.warmup, args.iters
            )
            print(name, n, report["network"][name][str(n)], report["pipeline"][name][str(n)])

    with open(os.path.join(args.out_dir, "latency.json"), "w") as f:
        json.dump(report, f, indent=2)
    table = markdown(report)
    with open(os.path.join(args.out_dir, "latency.md"), "w") as f:
        f.write(table)
    print(table)


if __name__ == "__main__":
    main()
