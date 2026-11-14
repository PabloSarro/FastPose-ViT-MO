"""Audit W8: compact, verified copies of the final weights (and the base of the release).

The final models only exist inside run folders on the scratch NAS (training checkpoints of ~690 MB
with optimizer and RNG state). This exports each one as a weights-only file (the state dict, FP32,
~345 MB, loadable with torch.load(weights_only=True)) next to a JSON with everything needed to
rebuild it (architecture flags, object list, input size, depth mode) and the results it gave, and
writes MANIFEST.json (sha256 of source and export). Every export is verified: the exported file is
loaded into a fresh model and must give bit-identical outputs to the source checkpoint on a fixed
batch. Detector weights (best.pth, EMA state dict) are copied with their JSON.

Re-run it when a new final model exists (T-LESS, the improved models of M11, the YCB-V detector):
sources that do not exist yet are skipped, existing exports are not redone (--force).

Usage (repo root, dev or a GPU job):
  PYTHONPATH=. .venv/bin/python scripts/export_release_weights.py [--out $NAS_HOME/weights/final]
An off-NAS copy or a Hugging Face upload is a decision for the user (needs a token).
"""

import argparse
import hashlib
import json
import os
import shutil
import types

import torch

from src.bop_dataset import SPLITS
from src.bop_evaluate import build_model
from src.models import VIT_MODELS
from src.safe_load import load_safe

NAS = os.environ["NAS_HOME"]
RUNS = f"{NAS}/outputs/FastPose-ViT/runs"

# name, dataset, source, vit_model, results (test AR of the BOP19 protocol with the default detections)
POSE = [
    ("lmo_tok4x_384", "lmo", f"{RUNS}/lmo-m4-tok4x-001/tok4x/last.pt", "vit_b_16_384",
     {"test AR, default detections (official BOP19)": 0.668, "test AR, GT boxes": 0.689,
      "seed": 0, "role": "final LM-O model"}),
    ("lmo_tok4x2_384", "lmo", f"{RUNS}/lmo-m4-tok4x2-001/tok4x2/last.pt", "vit_b_16_384",
     {"test AR, default detections (official BOP19)": 0.664, "seed": 1}),
    ("ycbv_tok4x2_384", "ycbv", f"{RUNS}/ycbv-m8-tok4x2-001/tok4x2/last.pt", "vit_b_16_384",
     {"test AR, default detections (official BOP19)": 0.702, "test AR, GT boxes": 0.707,
      "seed": 1, "role": "final YCB-V model (selected on PBR validation)"}),
    ("ycbv_tok4x_384", "ycbv", f"{RUNS}/ycbv-m8-tok4x-001/tok4x/last.pt", "vit_b_16_384",
     {"test AR, default detections (official BOP19)": 0.694, "seed": 0}),
    ("lmo_tok4x_224", "lmo", f"{RUNS}/lmo-m10-r224tok4x-001/tok4x/last.pt", "vit_b_16",
     {"test AR, default detections (official BOP19)": 0.643, "seed": 0, "role": "224 px model"}),
    ("lmo_tok4x2_224", "lmo", f"{RUNS}/lmo-m10-r224tok4x2-001/tok4x2/last.pt", "vit_b_16",
     {"test AR, default detections (official BOP19)": 0.643, "seed": 1}),
    ("tless_tok4x_384", "tless", f"{RUNS}/tless-m10-tok4x-002/tok4x/last.pt", "vit_b_16_384",
     {"seed": 0, "role": "T-LESS model (training done?)"}),
    ("tless_tok4x2_384", "tless", f"{RUNS}/tless-m10-tok4x2-002/tok4x2/last.pt", "vit_b_16_384",
     {"seed": 1}),
]
DETECTORS = [
    ("lmo_lwdetr_small", "lmo", "small", f"{RUNS}/lmo-m9-detsmall-001/small/best.pth",
     {"test mAP": 0.656, "pose AR with tok4x": 0.654}),
    ("lmo_lwdetr_tiny", "lmo", "tiny", f"{RUNS}/lmo-m9-dettiny-001/tiny/best.pth",
     {"test mAP": 0.628, "pose AR with tok4x": 0.645}),
    ("ycbv_lwdetr_small", "ycbv", "small", f"{RUNS}/ycbv-m10-detsmall-001/small/best.pth", {}),
]


def finished(source: str) -> bool:
    """A run is final when its job log ends with "exit code 0" (the reliable signal, AGENTS.md) or
    its training script wrote the TRAINED marker next to the weights. A run that is still training
    must not be exported as a final model."""
    variant_dir = os.path.dirname(source)
    if os.path.exists(os.path.join(variant_dir, "TRAINED")):
        return True
    log = os.path.join(os.path.dirname(variant_dir), "log.txt")
    if not os.path.exists(log):
        return False
    with open(log, errors="ignore") as f:
        return any("exit code 0" in line for line in f.readlines()[-30:])


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(run_dir: str) -> str | None:
    info = os.path.join(os.path.dirname(os.path.dirname(run_dir)), "info.txt")
    if os.path.exists(info):
        for line in open(info):
            if line.strip().startswith("git:"):
                return line.split(":", 1)[1].strip()
    return None


def verify(pose_args, source: str, exported: str) -> float:
    """Max |diff| of the network outputs of the source checkpoint and the exported file."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    size = VIT_MODELS[pose_args.vit_model][3][0]
    n = len(SPLITS[pose_args.dataset]["obj_ids"])
    images = torch.randn(4, 3, size, size, device=dev)
    obj_idx = torch.arange(4, device=dev) % n
    outs = []
    for path in (source, exported):
        pose_args.model_weights = path
        model = build_model(pose_args, dev)
        with torch.no_grad():
            out = model(images, obj_idx)
        outs.append(torch.cat([o.flatten() for o in out]) if isinstance(out, (tuple, list)) else out)
        del model
    return float((outs[0] - outs[1]).abs().max())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=f"{NAS}/weights/final")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    manifest_path = os.path.join(args.out, "MANIFEST.json")
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {}

    for name, dataset, source, vit, results in POSE:
        dst = os.path.join(args.out, f"{name}.pt")
        if not os.path.exists(source) or not finished(source):
            print(f"{name}: not finished (no exit code 0 / TRAINED marker), skipped")
            continue
        if os.path.exists(dst) and not args.force and name in manifest:
            print(f"{name}: already exported")
            continue
        ckpt = load_safe(source)
        state = ckpt["model"] if "model" in ckpt else ckpt
        epoch = ckpt.get("epoch") if isinstance(ckpt, dict) else None
        torch.save({"model": {k: v.detach().clone().contiguous() for k, v in state.items()}}, dst)
        pose_args = types.SimpleNamespace(vit_model=vit, dataset=dataset, object_conditioning="token",
                                          model_weights=None)
        diff = verify(pose_args, source, dst)
        cfg = {
            "kind": "pose", "dataset": dataset, "vit_model": vit,
            "input_size": VIT_MODELS[vit][3][0], "object_conditioning": "token",
            "obj_ids": list(SPLITS[dataset]["obj_ids"]), "depth_mode": "site",
            "rotation": "apparent (allocentric), 6D, merged linear head, 1 class token per object",
            "load": "torch.load(path, weights_only=True)['model']; src.bop_evaluate.build_model",
            "source_checkpoint": source, "source_epoch": epoch, "source_git": git_commit(source),
            "sha256_source": sha256(source), "sha256_export": sha256(dst),
            "size_mb": round(os.path.getsize(dst) / 1e6, 1),
            "verify_max_abs_diff_vs_source": diff, "results": results,
        }
        json.dump(cfg, open(os.path.join(args.out, f"{name}.json"), "w"), indent=1)
        manifest[name] = cfg
        print(f"{name}: exported {cfg['size_mb']} MB, verify max|diff| = {diff:g}")
        assert diff == 0.0, f"{name}: exported weights differ from the checkpoint"

    for name, dataset, variant, source, results in DETECTORS:
        dst = os.path.join(args.out, f"{name}.pth")
        if not os.path.exists(source) or not finished(source):
            print(f"{name}: not finished (no exit code 0 / TRAINED marker), skipped")
            continue
        if os.path.exists(dst) and not args.force and name in manifest:
            print(f"{name}: already exported")
            continue
        state = load_safe(source)  # weights_only: refuses anything but tensors / plain containers
        shutil.copyfile(source, dst)
        cfg = {
            "kind": "detector", "dataset": dataset, "variant": variant, "img_size": 640,
            "num_classes": len(SPLITS[dataset]["obj_ids"]), "obj_ids": list(SPLITS[dataset]["obj_ids"]),
            "load": "torch.load(path, weights_only=True); object_detector.bop_detector.BOPLWDETR",
            "tensors": len(state), "source_checkpoint": source, "source_git": git_commit(source),
            "sha256_export": sha256(dst), "size_mb": round(os.path.getsize(dst) / 1e6, 1),
            "results": results,
        }
        json.dump(cfg, open(os.path.join(args.out, f"{name}.json"), "w"), indent=1)
        manifest[name] = cfg
        print(f"{name}: copied {cfg['size_mb']} MB")

    # the third-party starting point of the detector: record its hash (provenance)
    coco = f"{NAS}/weights/lwdetr/pretrain_weights"
    manifest["_third_party"] = {
        f: {"sha256": sha256(os.path.join(coco, f)),
            "source": "Hugging Face xbsu/LW-DETR (official LW-DETR COCO release)"}
        for f in sorted(os.listdir(coco))
    }
    json.dump(manifest, open(manifest_path, "w"), indent=1)
    with open(os.path.join(args.out, "README.md"), "w") as f:
        f.write("# Final weights (FastPose-ViT-MO)\n\nWeights-only exports of the final models, "
                "verified against their training checkpoints (bit-identical outputs); see "
                "MANIFEST.json for the configuration, results, source run and sha256 of each "
                "file. Load with `torch.load(path, weights_only=True)`.\n\nProduced by "
                "`scripts/export_release_weights.py`. This folder is on the same scratch NAS as the "
                "runs: copy it off the NAS (or upload it) for a real backup.\n")
    print(f"manifest: {manifest_path} ({len([k for k in manifest if not k.startswith('_')])} entries)")


if __name__ == "__main__":
    main()
