"""M11 C1: RGB-D variant of FastPose-ViT-MO, the measured depth corrects the regressed depth.

The RGB model's pose (R, t) is kept except for its distance: for each estimate, the object model is
rendered at the current pose (depth only, bop_toolkit vispy renderer, headless EGL), compared with
the test depth image on the pixels where both are valid and agree to within a band (which drops
occluders in front and background behind), and the object is moved along its viewing ray by the
median depth difference: t <- t * (t_z + delta) / t_z. This keeps the 2D projection of the object
centre (and the rotation) and only changes the distance. Three iterations with a narrowing band
(20 / 10 / 5 % of t_z). Estimates with fewer than --min_pixels agreeing pixels are kept as they are.
The time column adds the measured correction time per image to the RGB time.

Usage (repo root, inside `dev` or a job; the renderer needs EGL):
  PYOPENGL_PLATFORM=egl PYTHONPATH=. .venv/bin/python scripts/depth_correction.py --dataset lmo \
      --csv RUN/eval_det/fastposevitmo_lmo-test.csv --out OUT/fastposevitmodepth_lmo-test.csv
  bash scripts/bop_eval.sh OUT fastposevitmodepth_lmo-test.csv
"""

import argparse
import csv
import json
import os
import time
from collections import defaultdict

import numpy as np
from bop_toolkit_lib import inout
from bop_toolkit_lib.rendering import renderer

from src.bop_dataset import SPLITS, default_bop_root

TEST_DIR = {"lmo": "test", "ycbv": "test", "tless": "test_primesense"}


def read(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(dict(scene_id=int(r["scene_id"]), im_id=int(r["im_id"]),
                             obj_id=int(r["obj_id"]), score=float(r["score"]),
                             R=np.array(r["R"].split(), float).reshape(3, 3),
                             t=np.array(r["t"].split(), float).reshape(3, 1),
                             time=float(r["time"])))
    return rows


def correct(ren, obj_id, R, t, depth, K, bands, min_pixels):
    """Move t along its viewing ray so that the rendered depth matches the measured one.
    Returns (t, number of agreeing pixels in the last iteration, total shift in mm)."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    t = t.copy()
    shift, n = 0.0, 0
    for band in bands:
        rendered = ren.render_object(obj_id, R, t, fx, fy, cx, cy)["depth"]
        tz = float(t[2, 0])
        valid = (rendered > 0) & (depth > 0) & (np.abs(depth - rendered) < band * tz)
        n = int(valid.sum())
        if n < min_pixels:
            break
        delta = float(np.median(depth[valid] - rendered[valid]))
        t *= (tz + delta) / tz
        shift += delta
    return t, n, shift


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=sorted(TEST_DIR))
    p.add_argument("--csv", required=True, help="BOP CSV of the RGB model")
    p.add_argument("--out", required=True, help="output CSV (METHOD_DATASET-test.csv)")
    p.add_argument("--bands", type=float, nargs="+", default=[0.2, 0.1, 0.05])
    p.add_argument("--min_pixels", type=int, default=50)
    args = p.parse_args()

    root = default_bop_root()
    models_ds = SPLITS[args.dataset].get("models_dataset", args.dataset)
    rows = read(args.csv)
    by_image = defaultdict(list)
    for r in rows:
        by_image[(r["scene_id"], r["im_id"])].append(r)
    cams = {}
    ren = None
    stats = []
    for (sc, im), ests in sorted(by_image.items()):
        sdir = os.path.join(root, args.dataset, TEST_DIR[args.dataset], f"{sc:06d}")
        if sc not in cams:
            cams[sc] = inout.load_json(os.path.join(sdir, "scene_camera.json"), keys_to_int=True)
        cam = cams[sc][im]
        K = np.array(cam["cam_K"]).reshape(3, 3)
        path = os.path.join(sdir, "depth", f"{im:06d}.png")
        depth = inout.load_depth(path) * float(cam["depth_scale"])  # mm
        if ren is None:
            h, w = depth.shape
            ren = renderer.create_renderer(w, h, "vispy", mode="depth")
            for o in SPLITS[args.dataset]["obj_ids"]:
                ren.add_object(o, os.path.join(root, models_ds, "models", f"obj_{o:06d}.ply"))
        start = time.perf_counter()
        for e in ests:
            if e["t"][2, 0] > 1e6:  # no detection: kept as a failure
                continue
            e["t"], n, shift = correct(ren, e["obj_id"], e["R"], e["t"], depth, K, args.bands,
                                       args.min_pixels)
            stats.append((n, shift, n >= args.min_pixels))
        elapsed = time.perf_counter() - start
        for e in ests:
            e["time"] += elapsed
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("scene_id,im_id,obj_id,score,R,t,time\n")
        for r in rows:
            f.write(f"{r['scene_id']},{r['im_id']},{r['obj_id']},{r['score']},"
                    f"{' '.join(map(str, r['R'].flatten()))},"
                    f"{' '.join(map(str, r['t'].flatten()))},{r['time']}\n")
    n = np.array([s[0] for s in stats])
    shift = np.array([s[1] for s in stats])
    ok = np.array([s[2] for s in stats])
    summary = dict(estimates=len(stats), corrected=int(ok.sum()),
                   median_shift_mm=float(np.median(shift[ok])) if ok.any() else None,
                   median_abs_shift_mm=float(np.median(np.abs(shift[ok]))) if ok.any() else None,
                   median_pixels=float(np.median(n)),
                   time_ms_per_image=1000 * float(np.mean([r["time"] for r in rows])),
                   bands=args.bands, min_pixels=args.min_pixels, input=os.path.abspath(args.csv))
    with open(args.out.replace(".csv", "_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(summary)


if __name__ == "__main__":
    main()
