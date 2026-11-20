"""Train the render-and-compare refiner (M19 R2, src/refiner.py) on BOP images.

Each sample is an object instance with its GT pose perturbed (NOISE_LEVELS); the image crop is cut
around the object at the perturbed pose, the model is rendered there (src/render.py, on the GPU),
and the network predicts the update that brings the pose back, trained with the symmetric
disentangled point-matching loss. Validation reports the loss of the starting poses and after one
update ("gain"), on fixed perturbations of the validation instances.

  PYTHONPATH=. .venv/bin/python src/train_refiner.py --dataset LMO --out_dir $RUN_DIR --resume auto

Resumable (last.pt every epoch, as src/train.py); model.pth = the best validation loss.
"""

import argparse
import json
import logging
import math
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.bop_dataset import (
    SPLITS,
    ObjectBalancedSampler,
    default_bop_root,
    default_cache_dir,
    load_index,
    load_models_info,
    object_extents,
    rows_for_objects,
    select_rows,
    symmetry_table,
)
from src.checkpoint import save_checkpoint
from src.refiner import NOISE_LEVELS, RefineDataset, RefineLoss, RefinerNet, apply_update, make_input
from src.render import PointRenderer, load_render_assets
from src.safe_load import load_safe

logger = logging.getLogger("refiner")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", default="LMO", choices=["LMO", "YCBV", "TLESS"])
    p.add_argument("--out_dir", required=True)
    p.add_argument("--resume", default="auto", choices=["auto", "no"])
    p.add_argument("--vit_model", default="convnext_base", help="convnext_base (a CNN learns the comparison; the ViT variants collapsed to the identity, TASKS.md M19.3) or a ViT name")
    p.add_argument("--render_init", default="zero", choices=["zero", "neg", "copy"], help="init of the render colour filters of the first layer")
    p.add_argument("--size", type=int, default=256, help="crop side = ViT input = render side")
    p.add_argument("--margin", type=float, default=1.4, help="crop side / the object's box (CosyPose: 1.4)")
    p.add_argument("--n_points", type=int, default=65536, help="render points per object")
    p.add_argument("--loss_points", type=int, default=1024, help="model points of the loss")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--epoch_samples", type=int, default=84416, help="samples per virtual epoch (object-balanced)")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--head_lr_mult", type=float, default=3.0, help="lr of the new layers (head, object token, first conv)")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_iters", type=int, default=500)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=12)
    p.add_argument("--pixel_aug", default="domain_gap")
    p.add_argument("--noise", default="mix", choices=["mix", "small", "medium", "large"], help="noise level(s) of the starting poses")
    p.add_argument("--occlusion_aug", action="store_true")
    p.add_argument("--train_split", default="train")
    p.add_argument("--val_split", default="val")
    p.add_argument("--max_val_samples", type=int, default=2000)
    p.add_argument("--obj_ids", type=int, nargs="+", default=None)
    p.add_argument("--overfit", type=int, default=0, help="sanity check: N fixed training instances, no augmentation")
    p.add_argument("--seed", type=int, default=0)
    return p


def make_datasets(args, assets):
    name = args.dataset.lower()
    cfg = SPLITS[name]
    obj_ids = args.obj_ids or list(cfg["obj_ids"])
    train_index = load_index(name, args.train_split)
    val_index = load_index(name, args.val_split)
    common = dict(
        assets=assets, size=args.size, margin=args.margin, dataset=name,
        real_background=False, noise_seed=args.seed,
        noise_levels=NOISE_LEVELS if args.noise == "mix" else (dict(NOISE_LEVELS[["small", "medium", "large"].index(args.noise)], p=1.0),),
    )
    train_rows = rows_for_objects(train_index, obj_ids)
    if args.overfit:
        rows = np.sort(np.random.default_rng(0).choice(train_rows, args.overfit, replace=False))
        index = select_rows(train_index, rows)
        return (RefineDataset(index=index, augment=False, **common),
                RefineDataset(index=index, augment=False, **common))
    val_rows = rows_for_objects(val_index, obj_ids)
    if args.max_val_samples and len(val_rows) > args.max_val_samples:
        val_rows = np.sort(np.random.default_rng(0).choice(val_rows, args.max_val_samples, replace=False))
    train = RefineDataset(
        index=select_rows(train_index, train_rows), augment=True, pixel_aug=args.pixel_aug,
        occlusion_aug=args.occlusion_aug, split="train", **common,
    )
    val = RefineDataset(index=select_rows(val_index, val_rows), augment=False, split="val", **common)
    return train, val


def step(model, renderer, loss_fn, batch, device, size):
    """One forward pass: (loss dict, loss dict of the starting poses)."""
    g = lambda k: batch[k].to(device, non_blocking=True)  # noqa: E731
    obj = g("obj_idx").long()
    R0, t0, K = g("R_init"), g("t_init"), g("K_crop")
    render, _ = renderer.render(obj, R0, t0, K, size)
    with torch.autocast("cuda", enabled=device.type == "cuda"):
        out = model(make_input(g("image"), render), obj)
    R1, t1 = apply_update(R0, t0, K, out.float(), size)
    after = loss_fn(R1, t1, g("R_gt"), g("t_gt"), obj)
    with torch.no_grad():
        before = loss_fn(R0, t0, g("R_gt"), g("t_gt"), obj)
    return after, before


@torch.no_grad()
def validate(model, renderer, loss_fn, loader, device, size) -> dict:
    model.eval()
    sums = {"after": 0.0, "before": 0.0, "after_rot": 0.0, "before_rot": 0.0, "after_trans": 0.0, "before_trans": 0.0}
    n = 0
    for batch in loader:
        after, before = step(model, renderer, loss_fn, batch, device, size)
        b = len(batch["row"])
        for key, val in (("after", after["full"]), ("before", before["full"]), ("after_rot", after["rot"]),
                         ("before_rot", before["rot"]), ("after_trans", after["trans"]), ("before_trans", before["trans"])):
            sums[key] += val.item() * b
        n += b
    out = {k: v / n for k, v in sums.items()}
    out["gain"] = 1.0 - out["after"] / max(out["before"], 1e-12)  # fraction of the error removed
    model.train()
    return out


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    name = args.dataset.lower()
    cfg = SPLITS[name]
    bop_root = default_bop_root()
    all_obj_ids = list(cfg["obj_ids"])
    obj_ids = args.obj_ids or all_obj_ids
    for k, v in sorted(vars(args).items()):
        logger.info(f"{k}: {v}")

    assets = load_render_assets(bop_root, name, all_obj_ids, args.n_points, default_cache_dir())
    centre, half = object_extents(bop_root, cfg["models_dataset"], all_obj_ids)
    renderer = PointRenderer(assets, centre, half, device)
    table, offsets = symmetry_table(bop_root, cfg["models_dataset"], all_obj_ids)
    diameters = load_models_info(bop_root, cfg["models_dataset"], all_obj_ids)["diameter"]
    loss_fn = RefineLoss(assets["points"][:, : args.loss_points], diameters, table, offsets).to(device)

    train_ds, val_ds = make_datasets(args, assets)
    sampler = None if args.overfit else ObjectBalancedSampler(
        train_ds.index["obj_idx"], args.epoch_samples, seed=args.seed
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=sampler is None, sampler=sampler,
        num_workers=args.num_workers, pin_memory=device.type == "cuda", drop_last=True,
        persistent_workers=args.num_workers > 1, prefetch_factor=4 if args.num_workers > 0 else None,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    logger.info(f"train {len(train_ds)} instances, {len(train_loader)} batches per epoch; val {len(val_ds)}")

    model = RefinerNet(args.vit_model, num_objects=len(all_obj_ids), render_init=args.render_init).to(device).train()
    new = [p for n, p in model.named_parameters() if n.startswith(("net.output", "net.object_token", "net.backbone.conv_proj", "net.backbone.features.0.0"))]
    new_ids = {id(p) for p in new}
    old = [p for p in model.parameters() if id(p) not in new_ids]
    optimizer = torch.optim.AdamW(
        [{"params": old, "lr": args.lr}, {"params": new, "lr": args.lr * args.head_lr_mult}],
        weight_decay=args.weight_decay,
    )
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    total_iters = args.epochs * len(train_loader)

    def set_lr(it: int) -> None:  # linear warmup, then cosine to 1 % of the lr
        scale = (it + 1) / args.warmup_iters if it < args.warmup_iters else (
            0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * (it - args.warmup_iters) / max(total_iters - args.warmup_iters, 1)))
        )
        for g, base in zip(optimizer.param_groups, base_lrs):
            g["lr"] = base * min(scale, 1.0)

    start_epoch, best = 0, float("inf")
    last = os.path.join(args.out_dir, "last.pt")
    if args.resume == "auto" and os.path.exists(last):
        ckpt = load_safe(last)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch, best = int(ckpt["epoch"]), float(ckpt["best"])
        logger.info(f"Resumed from {last} at epoch {start_epoch}")

    metrics_path = os.path.join(args.out_dir, "metrics.jsonl")
    for epoch in range(start_epoch, args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        start = time.time()
        run = {"total": 0.0, "full": 0.0, "before": 0.0}
        for i, batch in enumerate(train_loader):
            set_lr(epoch * len(train_loader) + i)
            after, before = step(model, renderer, loss_fn, batch, device, args.size)
            optimizer.zero_grad(set_to_none=True)
            after["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            run["total"] += after["total"].item()
            run["full"] += after["full"].item()
            run["before"] += before["full"].item()
            if i % 100 == 0:
                logger.info(f"epoch {epoch + 1} it {i}/{len(train_loader)} loss {after['total'].item():.4f} "
                            f"full {after['full'].item():.4f} (start {before['full'].item():.4f})")
        n = len(train_loader)
        val = validate(model, renderer, loss_fn, val_loader, device, args.size)
        record = {"epoch": epoch + 1, "seconds": time.time() - start, "train_total": run["total"] / n,
                  "train_full": run["full"] / n, "train_before": run["before"] / n, **{f"val_{k}": v for k, v in val.items()}}
        logger.info("epoch %d: train full %.4f (start %.4f) | val full %.4f (start %.4f) gain %.1f %% | %.0f s",
                    epoch + 1, record["train_full"], record["train_before"], val["after"], val["before"],
                    100 * val["gain"], record["seconds"])
        with open(metrics_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        if val["after"] < best:
            best = val["after"]
            torch.save(model.state_dict(), os.path.join(args.out_dir, "model.pth"))
        save_checkpoint(last, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                               "epoch": epoch + 1, "best": best, "args": vars(args)})
    logger.info("Training completed; best validation loss %.4f", best)


if __name__ == "__main__":
    main()
