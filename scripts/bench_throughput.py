# This file measures training throughput: how many BOP samples/s the DataLoader delivers, and
# how many the GPU can train on, to see which one limits a run (TASKS.md M2)
#
#   PYTHONPATH=. .venv/bin/python scripts/bench_throughput.py loader --workers 16 --batches 300
#   PYTHONPATH=. .venv/bin/python scripts/bench_throughput.py gpu --batch_size 32 --compile

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bop_dataset import BOPDataset, crop_image, load_index  # noqa: E402


def bench_loader(args):
    index = load_index("lmo", "train")
    if args.image_root:
        # Same rows, images read from a local copy (e.g. staged to the job's disk)
        index = dict(index)
        index["source_dirs"] = np.array([args.image_root])
    ds = BOPDataset(index=index, img_size=args.img_size, pixel_aug=args.pixel_aug)
    generator = torch.Generator().manual_seed(args.seed)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=4 if args.workers else None,
        generator=generator,
        drop_last=True,
    )
    times = []
    start = time.perf_counter()
    last = start
    for i, batch in enumerate(loader):
        _ = (
            batch["images"].cuda(non_blocking=True)
            if torch.cuda.is_available()
            else batch["images"]
        )
        now = time.perf_counter()
        times.append(now - last)
        last = now
        if i + 1 == args.batches:
            break
    total = time.perf_counter() - start
    steady = np.array(times[10:]) if len(times) > 20 else np.array(times)
    print(
        f"loader workers={args.workers} batch={args.batch_size} img={args.img_size} "
        f"aug={args.pixel_aug} root={'local' if args.image_root else 'NAS'}: "
        f"{len(times) * args.batch_size / total:.0f} samples/s overall, "
        f"{args.batch_size / steady.mean():.0f} samples/s after the first 10 batches "
        f"(first batch {times[0]:.1f} s, {len(times)} batches in {total:.0f} s)"
    )


def bench_single_sample(args):
    """Where one sample's time goes, in one process: image read+decode vs crop+labels+augs."""
    ds = BOPDataset("lmo", "train", img_size=args.img_size, pixel_aug=args.pixel_aug)
    rows = np.random.default_rng(args.seed).choice(len(ds), args.batches, replace=False)
    t_read = t_rest = 0.0
    for row in rows:
        t0 = time.perf_counter()
        image = ds.load_image(int(row))
        t1 = time.perf_counter()
        bbox = ds.crop_bbox(int(row))
        crop = crop_image(image, bbox, args.img_size, ds.crop_context)
        ds.transform(image=crop.astype(np.float32) / 255.0)
        ds.labels(int(row), bbox)
        t_read += t1 - t0
        t_rest += time.perf_counter() - t1
    n = len(rows)
    print(
        f"one process, {n} random samples: read+decode {1000 * t_read / n:.1f} ms, "
        f"crop+augs+labels {1000 * t_rest / n:.1f} ms per sample"
    )


def bench_gpu(args):
    from torch.amp import GradScaler, autocast

    from src.models import FastPoseViT
    from src.optimizer import SingleDeviceMuonWithAuxAdam

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    model = FastPoseViT(
        "vit_b_16_384_no_weights" if args.img_size == 384 else "vit_b_16_no_weights",
        num_hidden_layers=0,
        hidden_layer_dim=0,
        out_dim_translation=3,
        out_dim_rotation=6,
        merge_outputs=True,
        no_mlp=True,
        num_objects=8,
    ).cuda()
    params = list(model.parameters())
    optimizer = SingleDeviceMuonWithAuxAdam(
        [
            {"params": [p for p in params if p.ndim >= 2], "use_muon": True, "lr": 1e-4},
            {
                "params": [p for p in params if p.ndim < 2],
                "use_muon": False,
                "lr": 1e-5,
                "betas": (0.9, 0.95),
            },
        ]
    )
    if args.compile:
        os.environ.setdefault("USER", "bench")
        model = torch.compile(model, mode="max-autotune")
    scaler = GradScaler(device="cuda")
    b = args.batch_size
    x = torch.randn(b, 3, args.img_size, args.img_size, device="cuda")
    obj_idx = torch.randint(0, 8, (b,), device="cuda")
    target = torch.randn(b, 9, device="cuda")

    def step():
        with autocast("cuda"):
            loss = (model(x, obj_idx).float() - target).pow(2).mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    t0 = time.perf_counter()
    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    warm = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(args.steps):
        step()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / args.steps
    print(
        f"gpu {torch.cuda.get_device_name()} batch={b} img={args.img_size} compile={args.compile}: "
        f"{b / dt:.0f} samples/s ({1000 * dt:.0f} ms/step; warmup {warm:.0f} s), "
        f"peak memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BOP training throughput benchmark")
    parser.add_argument("what", choices=["loader", "sample", "gpu"])
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--batches", type=int, default=300, help="loader batches / samples to time"
    )
    parser.add_argument("--img_size", type=int, default=384)
    parser.add_argument("--pixel_aug", default="standard")
    parser.add_argument("--image_root", default=None, help="read train_pbr images from this copy")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()
    {"loader": bench_loader, "sample": bench_single_sample, "gpu": bench_gpu}[args.what](args)
