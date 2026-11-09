# M9 stretch: multi-class LW-DETR detector for BOP (all objects of a dataset), so the on-device
# pipeline needs no external detections. The SPEED one-class detector (object_detector/train.py)
# is unchanged.
#
# Data: the cached BOP index of src/bop_dataset.py (PBR train scenes, instances with
# visib_fract >= 0.1 of the dataset's objects, bbox_visib), grouped by image. Validation: the
# held-out PBR scenes (val_pbr), COCO mAP with pycocotools. Images are resized (stretched) to
# --img_size squared; boxes are normalised cx, cy, w, h, so the stretch does not change them.
# Augmentation: the pose model's `domain_gap` pixel pipeline (src/bop_dataset.pixel_transform)
# and a horizontal flip.
#
# Model: LW-DETR (object_detector/LWDETR) with the original LW-DETR training recipe instead of
# the SPEED wrapper's single-object settings: group DETR with 13 query groups in training (one
# at eval), IoU-aware BCE classification loss, loss weights 1 / 5 / 2 (cls / L1 / GIoU), AdamW
# lr 1e-4 (encoder 1.5e-4), weight decay 1e-4, EMA, COCO-pretrained weights (all layers except
# the class heads, whose size depends on the number of classes).
#
# Commands (from the repo root, PYTHONPATH=.):
#   train:   .venv/bin/python object_detector/bop_detector.py train --dataset lmo \
#                --variant tiny --pretrain $NAS_HOME/weights/lwdetr/.../LWDETR_tiny_60e_coco.pth \
#                --out_dir $RUN_DIR --resume auto
#   predict: .venv/bin/python object_detector/bop_detector.py predict --dataset lmo \
#                --variant tiny --weights RUN/best.pth --out_dir OUT
#            -> OUT/lwdetr_<dataset>-test.json, BOP default-detections format (scene_id,
#               image_id, category_id = BOP obj_id, score, bbox [x, y, w, h], time), for
#               src/bop_evaluate.py --detections.

import argparse
import contextlib
import copy
import io
import json
import logging
import math
import os
import random
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from object_detector.LWDETR.lwdetr_models import LWDETRWrapper
from src.bop_dataset import (
    SPLITS,
    BOPDataset,
    bop_bbox_to_xyxy,
    load_index,
    pixel_transform,
)
from src.bop_evaluate import normalize_crops


class BOPLWDETR(LWDETRWrapper):
    """LW-DETR with the original LW-DETR training recipe (see the file header)."""

    def _create_single_object_config(self, variant, override_num_queries=None):
        args = super()._create_single_object_config(variant, override_num_queries)
        args.group_detr = 13
        args.ia_bce_loss = True
        args.cls_loss_coef, args.bbox_loss_coef, args.giou_loss_coef = 1.0, 5.0, 2.0
        args.set_cost_class, args.set_cost_bbox, args.set_cost_giou = 2.0, 5.0, 2.0
        return args

    def load_coco(self, path: str) -> dict:
        """COCO checkpoint of the LW-DETR release: every tensor whose shape matches."""
        state = torch.load(path, map_location="cpu", weights_only=False)
        state = state.get("model", state)
        own = self.lwdetr_model.state_dict()
        kept = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
        self.lwdetr_model.load_state_dict(kept, strict=False)
        skipped = sorted(k for k in state if k not in kept)
        missing = sorted(k for k in own if k not in kept)
        return {"loaded": len(kept), "skipped": skipped, "missing": missing}


def image_groups(index) -> list[list[int]]:
    """Rows of the index grouped by image, in first-seen order."""
    groups: dict[tuple, list[int]] = {}
    for row, key in enumerate(zip(index["scene_id"].tolist(), index["im_id"].tolist())):
        groups.setdefault(key, []).append(row)
    return list(groups.values())


class BOPDetDataset(Dataset):
    """One item per image: normalised image (3, S, S) and its targets (labels, boxes cxcywh)."""

    def __init__(self, index, img_size: int, augment: bool, max_images: int | None = None):
        self.index = index
        # Only for load_image (paths of the rows)
        self.loader = BOPDataset(index=index, augment=False, bbox_source="visib", img_size=384)
        self.images = image_groups(index)
        if max_images:
            self.images = self.images[:max_images]
        self.img_size = img_size
        self.augment = augment
        self.transform = pixel_transform("domain_gap" if augment else "none")

    def __len__(self):
        return len(self.images)

    def targets(self, i: int, w: int, h: int) -> dict:
        rows = self.images[i]
        boxes = np.array([bop_bbox_to_xyxy(self.index["bbox_visib"][r]) for r in rows])
        boxes = np.clip(boxes / [w, h, w, h], 0.0, 1.0)
        cxcywh = np.stack(
            [
                (boxes[:, 0] + boxes[:, 2]) / 2,
                (boxes[:, 1] + boxes[:, 3]) / 2,
                boxes[:, 2] - boxes[:, 0],
                boxes[:, 3] - boxes[:, 1],
            ],
            axis=1,
        )
        return {
            "labels": torch.as_tensor(self.index["obj_idx"][rows], dtype=torch.int64),
            "boxes": torch.as_tensor(cxcywh, dtype=torch.float32),
        }

    def __getitem__(self, i: int):
        row = self.images[i][0]
        image = self.loader.load_image(row)
        h, w = image.shape[:2]
        image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        x = self.transform(image=image.astype(np.float32) / 255.0)["image"]
        tgt = self.targets(i, w, h)
        if self.augment and random.random() < 0.5:
            x = x.flip(-1)
            tgt["boxes"][:, 0] = 1.0 - tgt["boxes"][:, 0]
        meta = {
            "scene_id": int(self.index["scene_id"][row]),
            "im_id": int(self.index["im_id"][row]),
            "wh": (w, h),
        }
        return x, tgt, meta


def collate(batch):
    images, targets, metas = zip(*batch)
    return torch.stack(images), list(targets), list(metas)


def build(args, device) -> BOPLWDETR:
    num_classes = len(SPLITS[args.dataset]["obj_ids"])
    model = BOPLWDETR(variant=args.variant, num_classes=num_classes)
    return model.to(device)


def raw_outputs(model, images):
    """All queries' outputs, also in eval mode."""
    model.return_all_queries = True
    return model(images)


def coco_map(model, loader, args, device) -> dict:
    """COCO bbox mAP / AP50 of `model` on `loader` (pycocotools, image size = original)."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    obj_ids = SPLITS[args.dataset]["obj_ids"]
    gt = {"images": [], "annotations": [], "categories": [{"id": o} for o in obj_ids]}
    dets = []
    model.eval()
    img_id = 0
    with torch.no_grad():
        for images, targets, metas in loader:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = raw_outputs(model, images.to(device, non_blocking=True))
            sizes = torch.tensor([[m["wh"][1], m["wh"][0]] for m in metas], device=device)
            out = {k: out[k].float() for k in ("pred_logits", "pred_boxes")}
            results = model._lwdetr_post["bbox"](out, sizes)
            for tgt, meta, res in zip(targets, metas, results):
                img_id += 1
                w, h = meta["wh"]
                gt["images"].append({"id": img_id, "width": w, "height": h})
                for lbl, (cx, cy, bw, bh) in zip(tgt["labels"].tolist(), tgt["boxes"].tolist()):
                    gt["annotations"].append(
                        {
                            "id": len(gt["annotations"]) + 1,
                            "image_id": img_id,
                            "category_id": obj_ids[lbl],
                            "bbox": [(cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h],
                            "area": bw * w * bh * h,
                            "iscrowd": 0,
                        }
                    )
                for s, lbl, (x1, y1, x2, y2) in zip(
                    res["scores"].tolist(), res["labels"].tolist(), res["boxes"].tolist()
                ):
                    dets.append(
                        {
                            "image_id": img_id,
                            "category_id": obj_ids[lbl],
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "score": s,
                        }
                    )
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = gt
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(dets) if dets else COCO()
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return {"mAP": float(ev.stats[0]), "AP50": float(ev.stats[1]), "AR100": float(ev.stats[8])}


class EMA:
    """Exponential moving average of the weights (with the usual warm-up of the decay)."""

    def __init__(self, model, decay: float, tau: int = 2000):
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.decay, self.tau, self.updates = decay, tau, 0

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = self.decay * (1 - math.exp(-self.updates / self.tau))
        msd = model.state_dict()
        for k, v in self.model.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)


def to_float(x):
    """bf16 tensors of a (nested) output structure -> fp32."""
    if torch.is_tensor(x):
        return x.float() if x.is_floating_point() else x
    if isinstance(x, dict):
        return {k: to_float(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(to_float(v) for v in x)
    return x


def lr_factor(it: int, total: int, warmup: int) -> float:
    if it < warmup:
        return (it + 1) / warmup
    progress = (it - warmup) / max(1, total - warmup)
    return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress))


def train(args) -> None:
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    train_ds = BOPDetDataset(
        load_index(args.dataset, "train"), args.img_size, augment=not args.no_augment,
        max_images=args.max_train_images,
    )
    val_ds = BOPDetDataset(
        load_index(args.dataset, "train" if args.val_on_train else "val_pbr"), args.img_size,
        augment=False, max_images=args.max_train_images if args.val_on_train else args.max_val_images,
    )  # --val_on_train: overfit check (same images, without augmentation)
    loader_kw = dict(
        num_workers=args.num_workers, collate_fn=collate, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, **loader_kw
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kw)
    logging.info(f"train {len(train_ds)} images, val {len(val_ds)} images")

    model = build(args, device)
    if args.pretrain:
        info = model.load_coco(args.pretrain)
        logging.info(
            f"COCO weights: {info['loaded']} tensors loaded; skipped {info['skipped']}; "
            f"missing {info['missing']}"
        )
    criterion = model._lwdetr_criterion
    weights = criterion.weight_dict
    enc = [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad]
    rest = [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{"params": enc, "lr": args.lr_encoder}, {"params": rest, "lr": args.lr}],
        weight_decay=args.weight_decay,
    )
    base_lrs = [args.lr_encoder, args.lr]
    ema = EMA(model, args.ema_decay) if args.ema_decay > 0 else None
    total_iters = args.epochs * len(train_loader)

    start_epoch, best, history = 0, -1.0, []
    last_path = os.path.join(args.out_dir, "last.pt")
    if args.resume == "auto" and os.path.exists(last_path):
        ckpt = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if ema is not None:
            ema.model.load_state_dict(ckpt["ema"])
            ema.updates = ckpt["ema_updates"]
        start_epoch, best, history = ckpt["epoch"] + 1, ckpt["best"], ckpt["history"]
        logging.info(f"Resumed from {last_path} at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0, losses = time.time(), []
        for b, (images, targets, _) in enumerate(train_loader):
            it = epoch * len(train_loader) + b
            for g, base in zip(optimizer.param_groups, base_lrs):
                g["lr"] = base * lr_factor(it, total_iters, args.warmup_iters)
            images = images.to(device, non_blocking=True)
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(images)
            out = to_float(out)  # loss in fp32 (also the aux / encoder outputs)
            loss_dict = criterion(out, targets)
            loss = sum(loss_dict[k] * weights[k] for k in loss_dict if k in weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            if ema is not None:
                ema.update(model)
            losses.append(loss.item())
            if b % args.log_every == 0:
                logging.info(
                    f"epoch {epoch} iter {b}/{len(train_loader)} loss "
                    f"{np.mean(losses[-args.log_every:]):.3f} "
                    f"({(time.time() - t0) / (b + 1):.3f} s/iter)"
                )
        eval_model = ema.model if ema is not None else model
        metrics = coco_map(eval_model, val_loader, args, device)
        metrics.update(epoch=epoch, train_loss=float(np.mean(losses)), minutes=(time.time() - t0) / 60)
        history.append(metrics)
        logging.info(f"epoch {epoch}: {metrics}")
        if metrics["mAP"] > best:
            best = metrics["mAP"]
            torch.save(eval_model.state_dict(), os.path.join(args.out_dir, "best.pth"))
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "ema": ema.model.state_dict() if ema is not None else None,
                "ema_updates": ema.updates if ema is not None else 0,
                "epoch": epoch,
                "best": best,
                "history": history,
                "args": vars(args),
            },
            last_path + ".tmp",
        )
        os.replace(last_path + ".tmp", last_path)
        with open(os.path.join(args.out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)
    logging.info(f"done, best val_pbr mAP {best:.4f}")


class Detector:
    """Image (H, W, 3) uint8 RGB -> BOP detections of all objects (on the GPU, then CPU)."""

    def __init__(self, model, img_size: int, obj_ids, top_k: int):
        self.model, self.img_size, self.obj_ids, self.top_k = model.eval(), img_size, obj_ids, top_k
        model._lwdetr_post["bbox"].num_select = top_k

    @torch.no_grad()
    def __call__(self, image: np.ndarray) -> list[dict]:
        h, w = image.shape[:2]
        x = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        x = normalize_crops(x[None], torch.device("cuda"))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = raw_outputs(self.model, x)
        out = {k: out[k].float() for k in ("pred_logits", "pred_boxes")}
        res = self.model._lwdetr_post["bbox"](out, torch.tensor([[h, w]], device="cuda"))[0]
        scores, labels, boxes = (res[k].cpu().numpy() for k in ("scores", "labels", "boxes"))
        return [
            {
                "category_id": int(self.obj_ids[lbl]),
                "score": float(s),
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
            }
            for s, lbl, (x1, y1, x2, y2) in zip(scores, labels, boxes)
        ]


def detect_split(detector, ds) -> tuple[list[dict], list[float]]:
    """Run `detector` on every image of `ds`: BOP detections (with the per-image time) and
    the per-image times in seconds (3 warm-up calls on the first images, GPU synced)."""
    out, times = [], []
    for i in range(len(ds)):
        row = ds.images[i][0]
        image = ds.loader.load_image(row)
        if i < 3:  # warm-up
            detector(image)
        torch.cuda.synchronize()
        start = time.perf_counter()
        dets = detector(image)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        for d in dets:
            d.update(
                scene_id=int(ds.index["scene_id"][row]),
                image_id=int(ds.index["im_id"][row]),
                time=elapsed,
            )
        out += dets
    return out, times


def predict(args) -> None:
    device = torch.device("cuda")
    model = build(args, device)
    model.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True))
    obj_ids = SPLITS[args.dataset]["obj_ids"]
    detector = Detector(model, args.img_size, obj_ids, args.top_k)
    ds = BOPDetDataset(load_index(args.dataset, args.split), args.img_size, augment=False)
    os.makedirs(args.out_dir, exist_ok=True)
    out, times = detect_split(detector, ds)
    path = os.path.join(args.out_dir, f"lwdetr_{args.dataset}-{args.split}.json")
    with open(path, "w") as f:
        json.dump(out, f)
    loader = DataLoader(ds, batch_size=16, num_workers=args.num_workers, collate_fn=collate)
    metrics = coco_map(model, loader, args, device)  # GT = the split's targets
    metrics.update(
        images=len(ds), detections=len(out), time_ms_median=1000 * float(np.median(times)),
        time_ms_mean=1000 * float(np.mean(times)), weights=os.path.abspath(args.weights),
    )
    with open(path.replace(".json", "_summary.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    logging.info(f"{path}: {metrics}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-class LW-DETR for BOP (M9 stretch)")
    parser.add_argument("mode", choices=["train", "predict"])
    parser.add_argument("--dataset", default="lmo", choices=sorted(SPLITS))
    parser.add_argument("--variant", default="tiny", choices=["tiny", "small", "medium"])
    parser.add_argument("--img_size", type=int, default=640)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--num_workers", type=int, default=8)
    # train
    parser.add_argument("--pretrain", default=None, help="LW-DETR COCO checkpoint")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_encoder", type=float, default=1.5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--clip_grad", type=float, default=0.1)
    parser.add_argument("--warmup_iters", type=int, default=500)
    parser.add_argument("--ema_decay", type=float, default=0.9997)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", default="auto", choices=["auto", "none"])
    parser.add_argument("--max_train_images", type=int, default=None)
    parser.add_argument("--max_val_images", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=200)
    parser.add_argument("--val_on_train", action="store_true", help="overfit check")
    parser.add_argument("--no_augment", action="store_true")
    # predict
    parser.add_argument("--weights", default=None, help="best.pth of a training run")
    parser.add_argument("--split", default="test")
    parser.add_argument("--top_k", type=int, default=100)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True
    )
    if args.mode == "train":
        train(args)
    else:
        if not args.weights:
            parser.error("predict needs --weights")
        predict(args)


if __name__ == "__main__":
    main()
