"""Paper figures: qualitative results and failure cases (as Figures 3 and 4 of the WACV paper).

Every test target of a BOP CSV (predictions with the default detections) is scored with the
official MSSD error (bop_toolkit, symmetry-aware, mm) divided by the object diameter. Crops around
the ground-truth box show the ground-truth axes (solid; for symmetric objects under the symmetry
closest to the prediction) and the predicted axes (dashed), with the translation error [mm] and
the symmetry-aware rotation error [deg] below.

  qualitative: per dataset, the best, median and worst instance (rows = datasets).
  failures:    hand-picked failure modes (occlusion on LM-O; the depth bias on YCB-V).

Usage (repo root):
  .venv/bin/python vis/paper_qualitative.py qualitative --csv lmo=RUN/eval_det/x.csv ycbv=...
  .venv/bin/python vis/paper_qualitative.py failures --csv lmo=... ycbv=...
Writes paper/figures/{qualitative,failures}.pdf (+ .png) and a CSV of the per-instance errors next
to them (paper/figures/errors_<dataset>.csv).
"""

import argparse
import csv
import json
import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from bop_toolkit_lib import inout, misc, pose_error  # noqa: E402

from src.bop_dataset import default_bop_root  # noqa: E402

TEST_DIR = {"lmo": "test", "ycbv": "test", "tless": "test_primesense"}
NAMES = {
    "lmo": {1: "ape", 5: "can", 6: "cat", 8: "driller", 9: "duck", 10: "eggbox", 11: "glue",
            12: "holepuncher"},
    "ycbv": {1: "master chef can", 2: "cracker box", 3: "sugar box", 4: "tomato soup can",
             5: "mustard bottle", 6: "tuna fish can", 7: "pudding box", 8: "gelatin box",
             9: "potted meat can", 10: "banana", 11: "pitcher base", 12: "bleach cleanser",
             13: "bowl", 14: "mug", 15: "power drill", 16: "wood block", 17: "scissors",
             18: "large marker", 19: "large clamp", 20: "extra large clamp", 21: "foam brick"},
}
AXIS_COLORS = ("#e34948", "#008300", "#2a78d6")  # x, y, z


def load_csv(path):
    rows = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            key = (int(r["scene_id"]), int(r["im_id"]), int(r["obj_id"]))
            rows.setdefault(key, []).append(
                (float(r["score"]), np.array(r["R"].split(), float).reshape(3, 3),
                 np.array(r["t"].split(), float).reshape(3, 1))
            )
    return rows


def rot_error_sym(R_e, R_g, t_g, syms):
    """Smallest rotation angle [deg] between R_e and R_g·S over the object's symmetries, and the
    ground-truth pose under that symmetry (R_g·S, t_g + R_g·s), drawn instead of the original so
    that the axes of a symmetric object are comparable with the prediction."""
    best, pose = 180.0, (R_g, t_g)
    for s in syms:
        R = R_g @ s["R"]
        cos = np.clip((np.trace(R_e.T @ R) - 1) / 2, -1, 1)
        angle = float(np.degrees(np.arccos(cos)))
        if angle < best:
            best, pose = angle, (R, t_g + R_g @ s["t"].reshape(3, 1))
    return best, pose


def score_instances(dataset, csv_path):
    """One record per BOP19 test target (instance) with its errors; highest-score estimate."""
    root = default_bop_root()
    ds_dir = os.path.join(root, dataset)
    targets = json.load(open(os.path.join(ds_dir, "test_targets_bop19.json")))
    info = inout.load_json(os.path.join(ds_dir, "models_eval", "models_info.json"), keys_to_int=True)
    preds = load_csv(csv_path)
    pts, syms = {}, {}
    cache = {}
    out = []
    for tg in targets:
        sc, im, obj = tg["scene_id"], tg["im_id"], tg["obj_id"]
        sdir = os.path.join(ds_dir, TEST_DIR[dataset], f"{sc:06d}")
        if sc not in cache:
            cache[sc] = (
                inout.load_json(os.path.join(sdir, "scene_gt.json"), keys_to_int=True),
                inout.load_json(os.path.join(sdir, "scene_gt_info.json"), keys_to_int=True),
                inout.load_json(os.path.join(sdir, "scene_camera.json"), keys_to_int=True),
            )
        gt, gt_info, cam = cache[sc]
        if obj not in pts:
            pts[obj] = inout.load_ply(
                os.path.join(ds_dir, "models_eval", f"obj_{obj:06d}.ply"))["pts"]
            syms[obj] = misc.get_symmetry_transformations(info[obj], max_sym_disc_step=0.01)
        insts = [(i, g) for i, g in enumerate(gt[im]) if g["obj_id"] == obj]
        ests = sorted(preds.get((sc, im, obj), []), key=lambda e: -e[0])
        for i, g in insts[: tg["inst_count"]]:
            rec = dict(dataset=dataset, scene_id=sc, im_id=im, obj_id=obj, gt_id=i,
                       visib=gt_info[im][i]["visib_fract"], bbox=gt_info[im][i]["bbox_visib"],
                       R_g=np.array(g["cam_R_m2c"]).reshape(3, 3),
                       t_g=np.array(g["cam_t_m2c"]).reshape(3, 1),
                       K=np.array(cam[im]["cam_K"]).reshape(3, 3),
                       diameter=info[obj]["diameter"])
            if not ests:
                rec.update(found=False, mssd_rel=np.inf)
            else:
                _, R_e, t_e = ests[0]
                mssd = pose_error.mssd(R_e, t_e, rec["R_g"], rec["t_g"], pts[obj], syms[obj])
                r_err, (R_s, t_s) = rot_error_sym(R_e, rec["R_g"], rec["t_g"], syms[obj])
                rec.update(found=True, R_e=R_e, t_e=t_e, mssd_rel=mssd / rec["diameter"],
                           t_err=float(np.linalg.norm(t_e - t_s)), r_err=r_err,
                           R_draw=R_s, t_draw=t_s, dz=float((t_e[2] - t_s[2])[0]))
            out.append(rec)
    return out


def write_errors(records, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene_id", "im_id", "obj_id", "visib", "mssd_rel", "t_err_mm", "r_err_deg",
                    "dz_mm"])
        for r in records:
            w.writerow([r["scene_id"], r["im_id"], r["obj_id"], f"{r['visib']:.3f}",
                        f"{r['mssd_rel']:.4f}", f"{r.get('t_err', np.nan):.1f}",
                        f"{r.get('r_err', np.nan):.1f}", f"{r.get('dz', np.nan):.1f}"])


def project(R, t, K, pts):
    x = K @ (R @ pts.T + t)
    return (x[:2] / x[2]).T - 0.5  # K coordinates -> pixel-index coordinates


def draw(ax, rec, title=None):
    root = default_bop_root()
    sdir = os.path.join(root, rec["dataset"], TEST_DIR[rec["dataset"]], f"{rec['scene_id']:06d}")
    path = os.path.join(sdir, "rgb", f"{rec['im_id']:06d}.png")
    if not os.path.exists(path):
        path = path.replace(".png", ".jpg")
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    x, y, w, h = rec["bbox"]
    side = 1.8 * max(w, h, 40)
    cx, cy = x + w / 2, y + h / 2
    x0, y0 = cx - side / 2, cy - side / 2
    ax.imshow(img)
    ax.set_xlim(x0, x0 + side)
    ax.set_ylim(y0 + side, y0)
    length = 0.5 * rec["diameter"]
    axes = np.vstack([np.zeros((1, 3)), np.eye(3) * length])
    uv_g = project(rec.get("R_draw", rec["R_g"]), rec.get("t_draw", rec["t_g"]), rec["K"], axes)
    for i, c in enumerate(AXIS_COLORS):
        ax.plot(uv_g[[0, i + 1], 0], uv_g[[0, i + 1], 1], color=c, lw=2.2, solid_capstyle="round")
    if rec["found"]:
        uv_e = project(rec["R_e"], rec["t_e"], rec["K"], axes)
        for i, c in enumerate(AXIS_COLORS):
            ax.plot(uv_e[[0, i + 1], 0], uv_e[[0, i + 1], 1], color="white", lw=2.6,
                    ls=(0, (2.5, 1.5)))
            ax.plot(uv_e[[0, i + 1], 0], uv_e[[0, i + 1], 1], color=c, lw=1.6, ls=(0, (2.5, 1.5)))
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    name = NAMES.get(rec["dataset"], {}).get(rec["obj_id"], f"obj {rec['obj_id']}")
    err = (f"{rec['t_err']:.0f} mm | {rec['r_err']:.1f}°" if rec["found"] else "not detected")
    if title is None or title:
        ax.set_title(title or name, fontsize=7.5, pad=2)
    ax.set_xlabel(f"{name}, visib. {100 * rec['visib']:.0f} %\n{err}", fontsize=6.5, labelpad=2)


def pick(records, q):
    found = sorted([r for r in records if r["found"]], key=lambda r: r["mssd_rel"])
    return found[min(len(found) - 1, int(round(q * (len(found) - 1))))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("figure", choices=["qualitative", "failures"])
    parser.add_argument("--csv", nargs="+", required=True, help="dataset=path pairs")
    parser.add_argument("--out_dir", default="paper/figures")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    recs = {}
    for item in args.csv:
        ds, path = item.split("=", 1)
        recs[ds] = score_instances(ds, path)
        write_errors(recs[ds], os.path.join(args.out_dir, f"errors_{ds}.csv"))
    plt.rcParams.update({"font.family": "serif"})

    if args.figure == "qualitative":
        cols = [("Best", 0.0), ("Median", 0.5), ("90th percentile", 0.9), ("Worst", 1.0)]
        fig, axs = plt.subplots(len(recs), len(cols), figsize=(6.9, 2.15 * len(recs)),
                                squeeze=False)
        row_names = {"lmo": "LM-O", "ycbv": "YCB-V", "tless": "T-LESS"}
        for r, (ds, records) in enumerate(recs.items()):
            for c, (label, q) in enumerate(cols):
                draw(axs[r, c], pick(records, q), title=label if r == 0 else "")
            axs[r, 0].set_ylabel(row_names.get(ds, ds), fontsize=8)
    else:
        picks = []
        if "lmo" in recs:
            for obj in (10, 9):  # eggbox, duck: the occluded LM-O failures
                cand = [r for r in recs["lmo"] if r["obj_id"] == obj and r["found"]
                        and r["visib"] < 0.5]
                picks.append((pick(cand, 0.75), "LM-O, occlusion"))
        if "ycbv" in recs:
            for obj in (16, 17):  # wood block, scissors: the YCB-V depth bias
                cand = [r for r in recs["ycbv"] if r["obj_id"] == obj and r["found"]
                        and r["visib"] > 0.8]
                picks.append((pick(cand, 0.75), "YCB-V, depth"))
        fig, axs = plt.subplots(1, len(picks), figsize=(1.75 * len(picks), 2.35), squeeze=False)
        for ax, (rec, label) in zip(axs[0], picks):
            draw(ax, rec, title=label)
            if rec["found"]:
                ax.set_xlabel(ax.get_xlabel() + f"\n$\\Delta t_z$ = {rec['dz']:+.0f} mm",
                              fontsize=6.5, labelpad=2)
    fig.tight_layout(pad=0.4, w_pad=0.3, h_pad=0.6)
    out = os.path.join(args.out_dir, args.figure)
    fig.savefig(out + ".pdf", dpi=300)
    fig.savefig(out + ".png", dpi=200)
    print(f"wrote {out}.pdf / .png")


if __name__ == "__main__":
    main()
