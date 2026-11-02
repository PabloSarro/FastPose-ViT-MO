# This file compares 2D detections (BOP default detections, COCO-style JSON) with the ground-truth
# boxes of a BOP split (M7): does the detector's box match bbox_visib or bbox_obj better?
#
# For each ground-truth instance it takes the detections of that object in that image, keeps the
# top inst_count by score (what the evaluation will use) and matches them to the instances by IoU
# with bbox_visib. Reports the IoU of the matched detection with both GT boxes, how often the
# detection is missing or wrong (IoU < 0.5), and the edge offsets (detection - GT) relative to the
# GT box size, to compare with the training jitter (each edge +-10 % of the box size).
#
#   PYTHONPATH=. .venv/bin/python scripts/detection_iou.py --detections FILE.json --out_dir DIR

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bop_dataset import bop_bbox_to_xyxy, load_index  # noqa: E402


def iou(a, b) -> float:
    """IoU of two [x1, y1, x2, y2] boxes."""
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def edge_offsets(det, gt) -> np.ndarray:
    """(left, top, right, bottom) of det - gt, as fractions of the gt box's width / height."""
    w, h = gt[2] - gt[0], gt[3] - gt[1]
    return np.array([(det[0] - gt[0]) / w, (det[1] - gt[1]) / h, (det[2] - gt[2]) / w, (det[3] - gt[3]) / h])


def main() -> None:
    parser = argparse.ArgumentParser(description="IoU of detections with bbox_visib / bbox_obj")
    parser.add_argument("--detections", required=True)
    parser.add_argument("--dataset", default="lmo")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    index = load_index(args.dataset, args.split)
    with open(args.detections) as f:
        dets = json.load(f)
    by_key: dict[tuple, list] = {}
    for d in dets:
        x, y, w, h = d["bbox"]
        by_key.setdefault((d["scene_id"], d["image_id"], d["category_id"]), []).append(
            (d["score"], [x, y, x + w, y + h])
        )

    # GT instances per (scene, image, object): inst_count = how many of them are targets
    gt_groups: dict[tuple, list[int]] = {}
    for row in range(len(index["obj_id"])):
        key = (int(index["scene_id"][row]), int(index["im_id"][row]), int(index["obj_id"][row]))
        gt_groups.setdefault(key, []).append(row)

    rows = []
    for key, gt_rows in gt_groups.items():
        cands = sorted(by_key.get(key, []), key=lambda c: -c[0])[: len(gt_rows)]
        best_any = {
            r: max((iou(c[1], bop_bbox_to_xyxy(index["bbox_visib"][r])) for c in by_key.get(key, [])), default=0.0)
            for r in gt_rows
        }
        # Greedy matching of the kept detections to the GT instances by IoU with bbox_visib
        free = list(range(len(cands)))
        for r in sorted(gt_rows, key=lambda r: -float(index["visib_fract"][r])):
            visib = bop_bbox_to_xyxy(index["bbox_visib"][r])
            full = bop_bbox_to_xyxy(index["bbox_obj"][r])
            if free:
                j = max(free, key=lambda j: iou(cands[j][1], visib))
                free.remove(j)
                score, box = cands[j]
                rec = dict(iou_visib=iou(box, visib), iou_obj=iou(box, full), score=score,
                           off_visib=edge_offsets(box, visib), off_obj=edge_offsets(box, full))
            else:
                rec = dict(iou_visib=0.0, iou_obj=0.0, score=np.nan, off_visib=None, off_obj=None)
            rec.update(obj_id=key[2], visib_fract=float(index["visib_fract"][r]), best_any_visib=best_any[r])
            rows.append(rec)

    os.makedirs(args.out_dir, exist_ok=True)
    max_inst = max(len(v) for v in gt_groups.values())
    lines = [
        f"Detections: {args.detections}",
        f"{args.dataset} {args.split}: {len(rows)} GT instances, {len(gt_groups)} (image, object) "
        f"pairs, max inst_count {max_inst}; top inst_count detections by score per pair",
        "",
        "| object | n | IoU with bbox_visib (mean / median) | IoU with bbox_obj (mean / median) | "
        "visib better | IoU_visib < 0.5 | best of all dets < 0.5 |",
        "|---|---|---|---|---|---|---|",
    ]

    def stats(sel, name):
        v = np.array([r["iou_visib"] for r in sel])
        o = np.array([r["iou_obj"] for r in sel])
        b = np.array([r["best_any_visib"] for r in sel])
        return (
            f"| {name} | {len(sel)} | {v.mean():.3f} / {np.median(v):.3f} | {o.mean():.3f} / "
            f"{np.median(o):.3f} | {np.mean(v > o):.0%} | {np.mean(v < 0.5):.1%} | {np.mean(b < 0.5):.1%} |"
        )

    for obj_id in sorted({r["obj_id"] for r in rows}):
        lines.append(stats([r for r in rows if r["obj_id"] == obj_id], str(obj_id)))
    lines.append(stats(rows, "**all**"))

    good = [r for r in rows if r["iou_visib"] >= 0.5]
    for name in ("visib", "obj"):
        off = np.array([r[f"off_{name}"] for r in good])
        lines += [
            "",
            f"Edge offsets detection - bbox_{name} (fraction of the GT box size; left, top, right, "
            f"bottom), {len(good)} detections with IoU_visib >= 0.5:",
            "  mean   " + " ".join(f"{v:+.3f}" for v in off.mean(0)),
            "  median " + " ".join(f"{v:+.3f}" for v in np.median(off, 0)),
            "  |x| p50 " + " ".join(f"{v:.3f}" for v in np.median(np.abs(off), 0))
            + " | p90 " + " ".join(f"{v:.3f}" for v in np.percentile(np.abs(off), 90, axis=0)),
            f"  share of edges within +-10 %: {np.mean(np.abs(off) <= 0.10):.1%}",
        ]
    vf = np.array([r["visib_fract"] for r in rows])
    v = np.array([r["iou_visib"] for r in rows])
    o = np.array([r["iou_obj"] for r in rows])
    lines += ["", "By visibility (visib_fract): IoU visib / obj"]
    for lo, hi in [(0.1, 0.5), (0.5, 0.9), (0.9, 1.01)]:
        m = (vf >= lo) & (vf < hi)
        lines.append(f"  [{lo:.1f}, {min(hi, 1):.1f}): n={m.sum():4d}  {v[m].mean():.3f} / {o[m].mean():.3f}")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(args.out_dir, "detection_iou.md"), "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
