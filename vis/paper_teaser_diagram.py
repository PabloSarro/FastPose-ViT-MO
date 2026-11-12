"""Paper figure 1 (teaser), clean diagram version: image -> multi-class detector -> N crops in one
batch -> one shared ViT-B/16 whose class token is shifted per object (c + e_k) -> linear head
(U, R') -> closed-form recovery -> N 6D poses. Real thumbnails (LM-O test image, default
detections, our predictions) inside a schematic drawing.

Usage (repo root):
  .venv/bin/python vis/paper_teaser_diagram.py --csv RUN/tok4x/eval_det/fastposevitmo_lmo-test.csv \
      --detections DET.json [--scene 2 --image 3]
Writes paper/figures/teaser_diagram.pdf (+ .png).
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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

from src.bop_dataset import default_bop_root  # noqa: E402

# Reference palette, fixed roles: blue = network, orange = per-object class token,
# yellow = detections; axes in the usual x / y / z = red / green / blue.
BLUE, BLUE_LIGHT, ORANGE, YELLOW = "#2a78d6", "#cde2fb", "#eb6834", "#eda100"
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#f0efec"
AXIS_COLORS = ("#e34948", "#008300", "#2a78d6")
NAMES = {1: "ape", 5: "can", 6: "cat", 8: "driller", 9: "duck", 10: "eggbox", 11: "glue",
         12: "holepuncher"}


def arrow(ax, x0, y0, x1, y1, text=None, dy=1.6):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=10,
                                 color=MUTED, lw=1.1, shrinkA=0, shrinkB=0))
    if text:
        ax.text((x0 + x1) / 2, y0 + dy, text, ha="center", va="bottom", fontsize=6.3, color=MUTED)


def box(ax, x, y, w, h, fc, ec, lw=1.0, r=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc,
                                ec=ec, lw=lw))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--detections", required=True)
    p.add_argument("--scene", type=int, default=2)
    p.add_argument("--image", type=int, default=3)
    p.add_argument("--out", default="paper/figures/teaser_diagram")
    args = p.parse_args()
    root = os.path.join(default_bop_root(), "lmo")
    sdir = os.path.join(root, "test", f"{args.scene:06d}")
    img = cv2.cvtColor(cv2.imread(os.path.join(sdir, "rgb", f"{args.image:06d}.png")),
                       cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    K = np.array(inout.load_json(os.path.join(sdir, "scene_camera.json"),
                                 keys_to_int=True)[args.image]["cam_K"]).reshape(3, 3)
    info = inout.load_json(os.path.join(root, "models_eval", "models_info.json"), keys_to_int=True)
    preds = {}
    for r in csv.DictReader(open(args.csv)):
        if int(r["scene_id"]) == args.scene and int(r["im_id"]) == args.image:
            preds[int(r["obj_id"])] = (np.array(r["R"].split(), float).reshape(3, 3),
                                       np.array(r["t"].split(), float).reshape(3, 1))
    dets = {}
    for d in json.load(open(args.detections)):
        if d["scene_id"] == args.scene and d["image_id"] == args.image:
            o = d["category_id"]
            if o in preds and (o not in dets or d["score"] > dets[o]["score"]):
                dets[o] = d
    objs = sorted(dets)

    plt.rcParams.update({"font.family": "serif", "font.size": 7})
    fig = plt.figure(figsize=(6.9, 1.9))
    ax = fig.add_axes([0, 0, 1, 1])
    X, Y = 160.0, 44.0  # drawing units; fig aspect 6.9 / 1.9 ~ 3.6 = X / Y
    ax.set_xlim(0, X)
    ax.set_ylim(0, Y)
    ax.set_aspect("equal")
    ax.axis("off")

    # 1. input image with detections (image coords -> drawing coords)
    iw, ih = 34.0, 34.0 * H / W
    ix, iy = 1.0, (Y - ih) / 2 - 1
    ax.imshow(img, extent=(ix, ix + iw, iy, iy + ih))
    sx, sy = iw / W, ih / H
    for o in objs:
        x, y, w, h = dets[o]["bbox"]
        ax.add_patch(Rectangle((ix + x * sx, iy + ih - (y + h) * sy), w * sx, h * sy, fill=False,
                               ec=YELLOW, lw=0.9))
    ax.text(ix + iw / 2, iy + ih + 1.5, "RGB image", ha="center", va="bottom", fontsize=7.5)

    # 2. detector arrow, then a fanned stack of crops
    arrow(ax, ix + iw + 1, Y / 2 - 1, 46.5, Y / 2 - 1, "detector\n(all objects)")
    cs = 11.0  # crop side
    show = objs[:4]
    cx0, cy0 = 48.0, Y / 2 - 1 - cs / 2 - 4.5
    for i, o in enumerate(reversed(show)):
        k = len(show) - 1 - i
        x0, y0 = cx0 + 2.4 * k, cy0 + 3.0 * k
        x, y, w, h = dets[o]["bbox"]
        side = max(w, h)
        cx, cy = x + w / 2, y + h / 2
        u0, u1 = int(max(cx - side / 2, 0)), int(min(cx + side / 2, W))
        v0, v1 = int(max(cy - side / 2, 0)), int(min(cy + side / 2, H))
        crop = cv2.resize(img[v0:v1, u0:u1], (96, 96))
        ax.imshow(crop, extent=(x0, x0 + cs, y0, y0 + cs), zorder=3 + i)
        ax.add_patch(Rectangle((x0, y0), cs, cs, fill=False, ec=YELLOW, lw=0.9, zorder=3 + i))
    ax.text(cx0 + 3.6 + cs / 2, cy0 - 1.3, f"$N$ = {len(objs)} crops,\none batch", ha="center",
            va="top", fontsize=6.5)
    ax.text(cx0 + 3.6 + cs / 2, cy0 + 9.0 + cs + 1.2, "object labels $k$", ha="center",
            va="bottom", fontsize=6.5, color=MUTED)

    # 3. shared ViT: token row + transformer block + linear head
    vx, vy, vw, vh = 69.0, 6.0, 37.0, 32.0
    arrow(ax, 63.5, Y / 2 - 1, vx - 0.5, Y / 2 - 1)
    box(ax, vx, vy, vw, vh, BLUE_LIGHT, BLUE)
    ax.text(vx + vw / 2, vy + vh - 2.6, "One ViT-B/16 for all objects", ha="center",
            va="top", fontsize=7.3)
    # tokens: class token (orange) + patch tokens (blue)
    ty, ts = vy + 4.0, 3.2
    ax.add_patch(Rectangle((vx + 3, ty), ts, ts, fc=ORANGE, ec="white", lw=0.6))
    for j in range(7):
        ax.add_patch(Rectangle((vx + 3 + (j + 1) * (ts + 0.6), ty), ts, ts, fc=BLUE, ec="white",
                               lw=0.6, alpha=0.85))
    ax.text(vx + 3 + 8 * (ts + 0.6) + 0.4, ty + ts / 2, "...", va="center", fontsize=7, color=INK)
    ax.text(vx + 3 + ts / 2, ty - 0.8, "$\\mathbf{c}+\\mathbf{e}_k$", ha="center", va="top",
            fontsize=6.5, color=ORANGE)
    ax.text(vx + 3 + 4.5 * (ts + 0.6), ty - 0.8, "patch tokens", ha="center", va="top",
            fontsize=6.5, color=MUTED)
    # transformer layers
    box(ax, vx + 3, ty + ts + 2.4, vw - 6, 9.0, "white", BLUE, lw=0.8, r=0.8)
    ax.text(vx + vw / 2, ty + ts + 2.4 + 4.5, "12 transformer layers", ha="center", va="center",
            fontsize=6.8)
    ax.text(vx + vw / 2, vy + vh - 6.8, "class token of object $k$: $\\mathbf{c}+\\mathbf{e}_k$",
            ha="center", va="top", fontsize=6.3, color=ORANGE)

    # linear head
    hx = vx + vw + 3.5
    arrow(ax, vx + vw + 0.3, Y / 2 - 1, hx - 0.3, Y / 2 - 1)
    box(ax, hx, Y / 2 - 6.5, 11.5, 11, "white", BLUE, lw=0.9, r=0.8)
    ax.text(hx + 5.75, Y / 2 - 1, "linear\nhead\n$\\mathbf{U},\\ \\mathbf{R}'$", ha="center",
            va="center", fontsize=6.5)

    # 4. closed-form recovery -> poses on the image
    ox = 129.0
    ow, oh = 160.0 - ox - 1, (160.0 - ox - 1) * H / W
    oy = (Y - oh) / 2 - 1
    arrow(ax, hx + 11.8, Y / 2 - 1, ox - 0.5, Y / 2 - 1)
    ax.text((hx + 11.8 + ox) / 2, Y / 2 - 2.6, "closed\nform", ha="center", va="top", fontsize=6.3,
            color=MUTED)
    ax.imshow(img, extent=(ox, ox + ow, oy, oy + oh))
    for o in objs:
        R, t = preds[o]
        pts = np.vstack([np.zeros((1, 3)), np.eye(3) * 0.5 * info[o]["diameter"]])
        uv = K @ (R @ pts.T + t)
        uv = (uv[:2] / uv[2]).T - 0.5
        px, py = ox + uv[:, 0] * ow / W, oy + oh - uv[:, 1] * oh / H
        for i, c in enumerate(AXIS_COLORS):
            ax.plot(px[[0, i + 1]], py[[0, i + 1]], color=c, lw=1.1, solid_capstyle="round")
    ax.text(ox + ow / 2, oy + oh + 1.5, "$N$ 6D poses $(\\mathbf{R}, \\mathbf{T})$", ha="center",
            va="bottom", fontsize=7.5)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out + ".pdf", dpi=300)
    fig.savefig(args.out + ".png", dpi=220)
    print(f"wrote {args.out}.pdf / .png ({len(objs)} objects)")


if __name__ == "__main__":
    main()
