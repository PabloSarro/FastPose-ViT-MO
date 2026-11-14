"""M11 C1: RGB-D variant of FastPose-ViT-MO, the measured depth corrects the regressed depth.

Two implementations (--method):
  splat (default, fast): all estimates of an image at once on the GPU in PyTorch. The object's
    model points (models_eval) are projected with the current pose; a coarse z-buffer (cells of
    --cell x --cell pixels, minimum depth per cell) keeps the points of the visible surface (within
    --vis_eps x diameter of the cell's front); each visible point is compared with the measured
    depth at its pixel. Iterations re-project the points (no rendering).
  render (reference): the bop_toolkit renderer, one depth render per estimate and iteration.

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
import torch
from bop_toolkit_lib import inout

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


class SplatCorrector:
    """GPU depth correction of all estimates of an image (see the module docstring)."""

    def __init__(self, root, models_ds, obj_ids, device, cell=4, vis_eps=0.05, max_points=None):
        self.device, self.cell, self.vis_eps = device, cell, vis_eps
        info = inout.load_json(os.path.join(root, models_ds, "models_eval", "models_info.json"),
                               keys_to_int=True)
        self.points, self.diameter = {}, {}
        for o in obj_ids:
            pts = inout.load_ply(os.path.join(root, models_ds, "models_eval",
                                              f"obj_{o:06d}.ply"))["pts"]
            if max_points and len(pts) > max_points:
                pts = pts[np.random.default_rng(0).choice(len(pts), max_points, replace=False)]
            self.points[o] = torch.as_tensor(pts, dtype=torch.float32, device=device)
            self.diameter[o] = float(info[o]["diameter"])

    @torch.no_grad()
    def __call__(self, obj_ids, Rs, ts, depth, K, bands, min_points):
        """obj_ids (E,), Rs (E, 3, 3), ts (E, 3) mm, depth (H, W) mm tensor, K (3, 3).
        Returns corrected ts (E, 3), visible-and-agreeing point counts (E,), shifts (E,)."""
        dev = self.device
        E = len(obj_ids)
        n_max = max(len(self.points[o]) for o in obj_ids)
        P = torch.zeros(E, n_max, 3, device=dev)
        valid_pt = torch.zeros(E, n_max, dtype=torch.bool, device=dev)
        for i, o in enumerate(obj_ids):
            n = len(self.points[o])
            P[i, :n], valid_pt[i, :n] = self.points[o], True
        diam = torch.tensor([self.diameter[o] for o in obj_ids], device=dev)
        R = torch.as_tensor(np.stack(Rs), dtype=torch.float32, device=dev)
        t = torch.as_tensor(np.stack(ts).reshape(E, 3), dtype=torch.float32, device=dev)
        H, W = depth.shape
        Hc, Wc = (H + self.cell - 1) // self.cell, (W + self.cell - 1) // self.cell
        fx, fy, cx, cy = (float(v) for v in (K[0, 0], K[1, 1], K[0, 2], K[1, 2]))
        Pc = P @ R.transpose(1, 2)  # (E, N, 3), rotated model points
        shift = torch.zeros(E, device=dev)
        count = torch.zeros(E, dtype=torch.long, device=dev)
        for band in bands:
            X = Pc + t[:, None, :]
            z = X[..., 2]
            u = torch.floor(fx * X[..., 0] / z + cx).long()  # pixel i spans [i, i + 1) in K coords
            v = torch.floor(fy * X[..., 1] / z + cy).long()
            ok = valid_pt & (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            uc, vc = u.clamp(0, W - 1), v.clamp(0, H - 1)
            # coarse z-buffer per estimate: the front surface of the object in each cell
            cell_idx = (torch.arange(E, device=dev)[:, None] * (Hc * Wc)
                        + (vc // self.cell) * Wc + uc // self.cell)
            zbuf = torch.full((E * Hc * Wc,), float("inf"), device=dev)
            zbuf.scatter_reduce_(0, cell_idx[ok], z[ok], reduce="amin")
            front = z <= zbuf[cell_idx] + self.vis_eps * diam[:, None]
            measured = depth[vc, uc]
            tz = t[:, 2]
            agree = ok & front & (measured > 0) & ((measured - z).abs() < band * tz[:, None])
            diff = torch.where(agree, measured - z, torch.full_like(z, float("nan")))
            delta = torch.nanmedian(diff, dim=1).values
            count = agree.sum(1)
            use = (count >= min_points) & torch.isfinite(delta)
            delta = torch.where(use, delta, torch.zeros_like(delta))
            t = t * ((tz + delta) / tz)[:, None]
            shift += delta
        return t.cpu().numpy(), count.cpu().numpy(), shift.cpu().numpy()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=sorted(TEST_DIR))
    p.add_argument("--csv", required=True, help="BOP CSV of the RGB model")
    p.add_argument("--out", required=True, help="output CSV (METHOD_DATASET-test.csv)")
    p.add_argument("--bands", type=float, nargs="+", default=[0.2, 0.1, 0.05])
    p.add_argument("--min_pixels", type=int, default=50,
                   help="render: agreeing pixels needed; splat: agreeing model points needed")
    p.add_argument("--method", default="splat", choices=["splat", "render"])
    p.add_argument("--cell", type=int, default=4, help="splat: z-buffer cell size [px]")
    p.add_argument("--vis_eps", type=float, default=0.05,
                   help="splat: a point is visible within this fraction of the diameter of its "
                   "cell's front")
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
    splat = None
    if args.method == "splat":
        device = torch.device("cuda")
        splat = SplatCorrector(root, models_ds, SPLITS[args.dataset]["obj_ids"], device,
                               cell=args.cell, vis_eps=args.vis_eps)
    times = []
    for (sc, im), ests in sorted(by_image.items()):
        sdir = os.path.join(root, args.dataset, TEST_DIR[args.dataset], f"{sc:06d}")
        if sc not in cams:
            cams[sc] = inout.load_json(os.path.join(sdir, "scene_camera.json"), keys_to_int=True)
        cam = cams[sc][im]
        K = np.array(cam["cam_K"]).reshape(3, 3)
        path = os.path.join(sdir, "depth", f"{im:06d}.png")
        depth = inout.load_depth(path) * float(cam["depth_scale"])  # mm
        if splat is not None:
            todo = [e for e in ests if e["t"][2, 0] <= 1e6]  # no detection: kept as a failure
            torch.cuda.synchronize()
            start = time.perf_counter()  # from the decoded depth image (as the RGB pipeline)
            if todo:
                d = torch.as_tensor(depth, dtype=torch.float32, device=splat.device)
                new_t, counts, shifts = splat([e["obj_id"] for e in todo], [e["R"] for e in todo],
                                              [e["t"] for e in todo], d, K, args.bands,
                                              args.min_pixels)
                for e, tt, n, sh in zip(todo, new_t, counts, shifts):
                    e["t"] = tt.reshape(3, 1).astype(float)
                    stats.append((int(n), float(sh), int(n) >= args.min_pixels))
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            times.append((len(todo), elapsed))
            for e in ests:
                e["time"] += elapsed
            continue
        if ren is None:
            from bop_toolkit_lib.rendering import renderer

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
        times.append((len(ests), elapsed))
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
                   correction_ms_per_image_mean=1000 * float(np.mean([x[1] for x in times])),
                   correction_ms_per_image_median=1000 * float(np.median([x[1] for x in times])),
                   method=args.method, cell=args.cell, vis_eps=args.vis_eps,
                   bands=args.bands, min_pixels=args.min_pixels, input=os.path.abspath(args.csv))
    with open(args.out.replace(".csv", "_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(summary)


if __name__ == "__main__":
    main()
