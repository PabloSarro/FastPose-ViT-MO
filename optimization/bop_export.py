# M9: export a multi-object BOP FastPoseViT to ONNX with a dynamic batch (one crop per object of
# an image), and check ONNX Runtime against PyTorch in fp32.
#
# The graph is the network only: inputs `images` (N, 3, S, S), normalised crops as in
# BOPDataset, and `obj_idx` (N,) int64, the dataset object index of each crop; output `pose`
# (N, 9) = the merged head (3 translation values, then the 6D rotation), exactly what
# forward_model() splits. Cropping, normalisation and the R, t recovery (CoordinateTransformer)
# stay outside the graph, as in src/bop_evaluate.py.
#
# Check: the same real crops (first --check_crops instances of the split, GT boxes) through
# PyTorch fp32 (TF32 off) and ONNX Runtime (CUDA EP with TF32 off, and CPU EP), for each batch
# size in --batch_sizes. Pass if every max |diff| < --tolerance (1e-3, M9 bullet 1). The report
# goes to <onnx path>.check.json.
#
# Usage (from the repo root):
#   PYTHONPATH=. .venv/bin/python optimization/bop_export.py --dataset lmo \
#       --model_weights RUN/tok4x/model.pth --object_conditioning token --onnx_path OUT/model.onnx

import argparse
import json
import logging
import os
import time

import numpy as np
import onnx
import torch
from torch import nn

from src.bop_dataset import SPLITS, BOPDataset, load_index, select_rows
from src.bop_evaluate import build_model
from src.models import OBJECT_CONDITIONINGS, VIT_MODELS

MAX_BATCH = 64  # upper bound of the dynamic batch dimension

# torch.export asks getpass.getuser() for its cache folder, which fails in RCP jobs (uid not in
# /etc/passwd, no USER): give it a name
os.environ.setdefault("USER", os.environ.get("LOGNAME") or f"uid{os.getuid()}")


class PoseGraph(nn.Module):
    """FastPoseViT with the exported signature: (images, obj_idx) -> pose (N, 9)."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor, obj_idx: torch.Tensor) -> torch.Tensor:
        return self.model(images, obj_idx)


def export_onnx(model: nn.Module, path: str, img_size: int, opset: int = 22) -> None:
    """Export `model` (eval mode, any device) to `path` with a dynamic batch dimension."""
    graph = PoseGraph(model).eval()
    device = next(model.parameters()).device
    # Batch 2 as the example: with 1 the exporter may specialise the batch dimension
    images = torch.zeros(2, 3, img_size, img_size, device=device)
    obj_idx = torch.zeros(2, dtype=torch.int64, device=device)
    batch = torch.export.Dim("batch", min=1, max=MAX_BATCH)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with torch.no_grad():
        program = torch.onnx.export(
            graph,
            (images, obj_idx),
            dynamo=True,
            opset_version=opset,
            input_names=["images", "obj_idx"],
            output_names=["pose"],
            dynamic_shapes={"images": {0: batch}, "obj_idx": {0: batch}},
            optimize=True,
            external_data=False,  # ViT-B fits in one file (< 2 GB)
        )
    program.save(path, external_data=False)
    onnx.checker.check_model(path)


def real_crops(dataset_name: str, split: str, n: int, img_size: int, depth_mode: str):
    """The first n instances of a split (GT boxes, no augmentation): images and obj_idx."""
    index = load_index(dataset_name, split)
    # Spread the n crops over the split so that several objects appear
    rows = np.linspace(0, len(index["obj_id"]) - 1, n).round().astype(int)
    ds = BOPDataset(
        index=select_rows(index, rows),
        augment=False,
        bbox_source="visib",
        depth_mode=depth_mode,
        img_size=img_size,
    )
    items = [ds[i] for i in range(len(ds))]
    images = torch.stack([it["images"] for it in items]).float()
    obj_idx = torch.as_tensor([int(it["obj_idx"]) for it in items], dtype=torch.int64)
    return images, obj_idx


def ort_sessions(path: str) -> dict:
    import onnxruntime as ort

    sessions = {}
    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" in providers:
        sess = ort.InferenceSession(
            path, providers=[("CUDAExecutionProvider", {"use_tf32": 0}), "CPUExecutionProvider"]
        )
        # Listed is not loadable: onnxruntime-gpu 1.23 needs CUDA 12 libraries, and image 1.60
        # has CUDA 13, so it silently falls back to the CPU (M9, 2026-11-08)
        if "CUDAExecutionProvider" in sess.get_providers():
            sessions["ort_cuda"] = sess
        else:
            logging.warning("ONNX Runtime CUDA provider not loadable: checking on the CPU only")
    sessions["ort_cpu"] = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return sessions


def check(model, path, images, obj_idx, batch_sizes, tolerance) -> dict:
    """Max |ORT - PyTorch| on the pose output, per execution provider and batch size."""
    device = next(model.parameters()).device
    with torch.no_grad():
        ref = model(images.to(device), obj_idx.to(device)).float().cpu().numpy()
    report = {"tolerance": tolerance, "n_crops": len(images), "results": {}}
    for name, sess in ort_sessions(path).items():
        res = {}
        for bs in batch_sizes:
            outs = []
            for start in range(0, len(images), bs):
                feed = {
                    "images": images[start : start + bs].numpy(),
                    "obj_idx": obj_idx[start : start + bs].numpy(),
                }
                outs.append(sess.run(["pose"], feed)[0])
            diff = np.abs(np.concatenate(outs) - ref)
            res[str(bs)] = {"max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean())}
            logging.info(
                f"{name} batch {bs}: max |diff| {diff.max():.2e}, mean {diff.mean():.2e}"
            )
        report["results"][name] = res
    report["max_abs_diff"] = max(
        r["max_abs_diff"] for res in report["results"].values() for r in res.values()
    )
    report["passed"] = report["max_abs_diff"] < tolerance
    report["output_abs_max"] = float(np.abs(ref).max())
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a BOP FastPoseViT to ONNX (M9)")
    parser.add_argument("--model_weights", required=True, help="model.pth or last.pt")
    parser.add_argument("--onnx_path", required=True)
    parser.add_argument("--dataset", default="lmo", choices=sorted(SPLITS))
    parser.add_argument("--vit_model", default="vit_b_16_384")
    parser.add_argument("--depth_mode", default="site")
    parser.add_argument("--object_conditioning", default="token", choices=OBJECT_CONDITIONINGS)
    parser.add_argument("--opset", type=int, default=22)
    parser.add_argument("--check_split", default="test")
    parser.add_argument("--check_crops", type=int, default=48)
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--tolerance", type=float, default=1e-3)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True
    )
    # fp32 reference: no TF32 in matmuls/convolutions
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args, device)
    img_size = VIT_MODELS[args.vit_model][3][0]

    start = time.perf_counter()
    export_onnx(model, args.onnx_path, img_size, args.opset)
    logging.info(
        f"Exported {args.onnx_path} ({os.path.getsize(args.onnx_path) / 2**20:.0f} MB) "
        f"in {time.perf_counter() - start:.0f} s"
    )

    images, obj_idx = real_crops(
        args.dataset, args.check_split, args.check_crops, img_size, args.depth_mode
    )
    report = check(model, args.onnx_path, images, obj_idx, args.batch_sizes, args.tolerance)
    report.update(
        model_weights=os.path.abspath(args.model_weights),
        dataset=args.dataset,
        check_split=args.check_split,
        object_conditioning=args.object_conditioning,
        opset=args.opset,
        torch=torch.__version__,
    )
    with open(args.onnx_path + ".check.json", "w") as f:
        json.dump(report, f, indent=2)
    logging.info(
        f"max |ORT - PyTorch| = {report['max_abs_diff']:.2e} (tolerance {args.tolerance}, "
        f"|output| up to {report['output_abs_max']:.2f}): "
        + ("PASSED" if report["passed"] else "FAILED")
    )


if __name__ == "__main__":
    main()
