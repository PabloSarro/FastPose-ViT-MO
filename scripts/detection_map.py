"""COCO bbox mAP of a BOP detections file against the split's targets (bbox_visib).

Same ground truth and computation as `object_detector/bop_detector.py predict` (instances with
visib_fract >= 0.1 of the dataset's objects, boxes from bop_bbox_to_xyxy), so the BOP default
detections and our own LW-DETR can be compared directly (docs/PAPER_TODO.md §2).

Usage: PYTHONPATH=. .venv/bin/python scripts/detection_map.py --dataset lmo --detections FILE.json
"""

import argparse
import contextlib
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bop_dataset import bop_bbox_to_xyxy, load_index  # noqa: E402


def main() -> None:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="lmo")
    parser.add_argument("--split", default="test")
    parser.add_argument("--detections", required=True)
    parser.add_argument("--out", default=None, help="write the metrics as JSON")
    args = parser.parse_args()
    index = load_index(args.dataset, args.split)
    obj_ids = [int(o) for o in index["obj_ids"]]
    images: dict[tuple, int] = {}
    gt = {"images": [], "annotations": [], "categories": [{"id": o} for o in obj_ids]}
    for row in range(len(index["obj_id"])):
        key = (int(index["scene_id"][row]), int(index["im_id"][row]))
        if key not in images:
            images[key] = len(images) + 1
            w, h = (int(v) for v in index["img_wh"][row])
            gt["images"].append({"id": images[key], "width": w, "height": h})
        x1, y1, x2, y2 = bop_bbox_to_xyxy(index["bbox_visib"][row])
        gt["annotations"].append(
            {
                "id": len(gt["annotations"]) + 1,
                "image_id": images[key],
                "category_id": int(index["obj_id"][row]),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": (x2 - x1) * (y2 - y1),
                "iscrowd": 0,
            }
        )
    with open(args.detections) as f:
        raw = json.load(f)
    dets = [
        {
            "image_id": images[(d["scene_id"], d["image_id"])],
            "category_id": int(d["category_id"]),
            "bbox": list(d["bbox"]),
            "score": float(d["score"]),
        }
        for d in raw
        if (d["scene_id"], d["image_id"]) in images and int(d["category_id"]) in obj_ids
    ]
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = gt
        coco_gt.createIndex()
        ev = COCOeval(coco_gt, coco_gt.loadRes(dets), "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    times = {(d["scene_id"], d["image_id"]): d.get("time", 0.0) for d in raw}
    out = {
        "detections": args.detections,
        "images": len(images),
        "targets": len(gt["annotations"]),
        "mAP": float(ev.stats[0]),
        "AP50": float(ev.stats[1]),
        "AP75": float(ev.stats[2]),
        "AR100": float(ev.stats[8]),
        "time_ms_mean": 1000 * sum(times.values()) / max(1, len(times)),
    }
    print(json.dumps(out, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
