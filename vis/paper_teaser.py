"""Paper figure 1 (teaser, first version): image -> detections -> batch of crops -> one shared ViT
with per-object class tokens -> all poses, on a real LM-O test image with real predictions.

Usage (repo root):
  .venv/bin/python vis/paper_teaser.py --csv RUN/tok4x/eval_det/fastposevitmo_lmo-test.csv \
      --detections DET.json [--scene 2 --image N]
Writes paper/figures/teaser.pdf (+ .png).
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
from bop_toolkit_lib import inout  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from src.bop_dataset import default_bop_root  # noqa: E402

AXIS_COLORS = ("#e34948", "#008300", "#2a78d6")
BOX = "#eda100"
INK, MUTED = "#0b0b0b", "#52514e"
NAMES = {1: "ape", 5: "can", 6: "cat", 8: "driller", 9: "duck", 10: "eggbox", 11: "glue",
         12: "holepuncher"}
SHORT = {12: "puncher"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--detections", required=True)
    p.add_argument("--scene", type=int, default=2)
    p.add_argument("--image", type=int, default=None, help="default: most objects detected")
    p.add_argument("--out", default="paper/figures/teaser")
    args = p.parse_args()
    root = os.path.join(default_bop_root(), "lmo")
    preds = {}
    for r in csv.DictReader(open(args.csv)):
        if int(r["scene_id"]) == args.scene:
            preds.setdefault(int(r["im_id"]), {})[int(r["obj_id"])] = (
                np.array(r["R"].split(), float).reshape(3, 3),
                np.array(r["t"].split(), float).reshape(3, 1))
    im = args.image or max(preds, key=lambda k: (len(preds[k]), -k))
    dets = {}
    for d in json.load(open(args.detections)):
        if d["scene_id"] == args.scene and d["image_id"] == im:
            o = d["category_id"]
            if o not in dets or d["score"] > dets[o]["score"]:
                dets[o] = d
    objs = sorted(o for o in preds[im] if o in dets)
    sdir = os.path.join(root, "test", f"{args.scene:06d}")
    K = np.array(inout.load_json(os.path.join(sdir, "scene_camera.json"),
                                 keys_to_int=True)[im]["cam_K"]).reshape(3, 3)
    info = inout.load_json(os.path.join(root, "models_eval", "models_info.json"), keys_to_int=True)
    img = cv2.cvtColor(cv2.imread(os.path.join(sdir, "rgb", f"{im:06d}.png")), cv2.COLOR_BGR2RGB)

    plt.rcParams.update({"font.family": "serif", "font.size": 7})
    fig = plt.figure(figsize=(6.9, 1.85))
    # [image + boxes] [crops 2 x 4] [shared ViT] [image + poses]
    ax_in = fig.add_axes([0.0, 0.08, 0.245, 0.84])
    ax_vit = fig.add_axes([0.54, 0.1, 0.16, 0.8])
    ax_out = fig.add_axes([0.755, 0.08, 0.245, 0.84])
    for ax in (ax_in, ax_out):
        ax.imshow(img)
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    ax_in.set_title("1. Detect all objects", fontsize=7.5, pad=3)
    ax_out.set_title("4. All 6D poses", fontsize=7.5, pad=3)
    for o in objs:
        x, y, w, h = dets[o]["bbox"]
        ax_in.add_patch(plt.Rectangle((x - 0.5, y - 0.5), w + 1, h + 1, fill=False, ec=BOX, lw=1.1))
        R, t = preds[im][o]
        axes = np.vstack([np.zeros((1, 3)), np.eye(3) * 0.5 * info[o]["diameter"]])
        uv = K @ (R @ axes.T + t)
        uv = (uv[:2] / uv[2]).T - 0.5
        for i, c in enumerate(AXIS_COLORS):
            ax_out.plot(uv[[0, i + 1], 0], uv[[0, i + 1], 1], color=c, lw=1.6,
                        solid_capstyle="round")

    # crops: square around each box, as the network sees them
    n = len(objs)
    cols = 4
    rows = int(np.ceil(n / cols))
    fw, fh = fig.get_size_inches()
    cw = 0.055
    ch = cw * fw / fh  # square crops
    gap_y = 0.11  # room for the label under each crop
    top = 0.5 + (rows * ch + (rows - 1) * gap_y) / 2
    crop_axes = []
    for k, o in enumerate(objs):
        r, c = divmod(k, cols)
        ax = fig.add_axes([0.272 + c * (cw + 0.01), top - (r + 1) * ch - r * gap_y, cw, ch])
        x, y, w, h = dets[o]["bbox"]
        side = max(w, h)
        cx, cy = x + w / 2, y + h / 2
        ax.imshow(img)
        ax.set_xlim(cx - side / 2, cx + side / 2)
        ax.set_ylim(cy + side / 2, cy - side / 2)
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor(BOX)
        ax.set_xlabel(SHORT.get(o, NAMES[o]), fontsize=5.5, labelpad=1)
        crop_axes.append(ax)
    fig.text(0.272 + 2 * (cw + 0.01) - 0.005, 0.93, "2. Crops, one batch", ha="center",
             va="bottom", fontsize=7.5)

    ax_vit.set_xlim(0, 1), ax_vit.set_ylim(0, 1)
    ax_vit.axis("off")
    ax_vit.add_patch(FancyBboxPatch((0.03, 0.05), 0.94, 0.9, boxstyle="round,pad=0.02,rounding_size=0.06",
                                    fc="#cde2fb", ec="#2a78d6", lw=1))
    ax_vit.text(0.5, 0.85, "3. One shared\nViT-B/16", ha="center", va="center", fontsize=7.5,
                color=INK)
    ax_vit.text(0.5, 0.62, "class token\n$\\mathbf{c} + \\mathbf{e}_k$ per object", ha="center",
                va="center", fontsize=6.5, color=INK)
    ax_vit.text(0.5, 0.38, "linear head\n$\\rightarrow \\mathbf{U},\\ \\mathbf{R}'$", ha="center",
                va="center", fontsize=6.5, color=INK)
    ax_vit.text(0.5, 0.14, "closed form $\\rightarrow \\mathbf{R}, \\mathbf{T}$", ha="center",
                va="center", fontsize=6.5, color=MUTED)

    def arrow(x0, x1, y=0.5):
        fig.add_artist(FancyArrowPatch((x0, y), (x1, y), transform=fig.transFigure,
                                       arrowstyle="-|>", mutation_scale=9, color=MUTED, lw=1))

    arrow(0.248, 0.268)
    arrow(0.523, 0.539)
    arrow(0.708, 0.752)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out + ".pdf", dpi=300)
    fig.savefig(args.out + ".png", dpi=200)
    print(f"wrote {args.out}.pdf / .png (scene {args.scene}, image {im}, {n} objects)")


if __name__ == "__main__":
    main()
