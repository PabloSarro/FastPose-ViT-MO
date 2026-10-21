# This file illustrates the M2 reprojection test and BOP's pixel convention
#
# Top: model points and their box projected with R (correct) and with R^T (wrong), against
#      BOP's bbox_obj.
# Bottom: a pixel-level zoom on the bottom-right corner of an object mask: the projected model
#      reaches x + w + 1, not x + w, which is why bop_bbox_to_xyxy adds 1.
#
#   python vis/reprojection.py

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from common import (
    BOX_OBJ_COLOR,
    OBJ_COLORS,
    TEXT,
    TEXT_2,
    image_axes,
    model_points,
    obj_label,
    project,
    save,
    to_pixel_index,
)
from matplotlib.lines import Line2D

from src.bop_dataset import bop_bbox_to_xyxy, image_path, load_index

WRONG = "#52514e"


def box_of(uv):
    return np.array([uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()])


def iou(a, b):
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def rect(ax, xyxy, **kw):
    x1, y1, x2, y2 = np.asarray(xyxy) - 0.5  # K coordinates -> pixel-index coordinates
    ax.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], **kw)


def example(ax, index, row):
    obj_id = int(index["obj_id"][row])
    R, t, K = index["R"][row], index["t"][row], index["K"][row]
    pts = model_points(obj_id, 1500, "models_eval")
    image = plt.imread(image_path(index, row))
    ax.imshow(image)
    good = project(pts, R, t, K)
    bad = project(pts, R.T, t, K)
    gt_box = bop_bbox_to_xyxy(index["bbox_obj"][row])
    ax.scatter(*to_pixel_index(bad).T, s=1.5, color=WRONG, alpha=0.5, linewidths=0)
    ax.scatter(*to_pixel_index(good).T, s=1.5, color=OBJ_COLORS[obj_id], alpha=0.7, linewidths=0)
    rect(ax, box_of(bad), color=WRONG, lw=1.5, ls=":")
    rect(ax, box_of(good), color=OBJ_COLORS[obj_id], lw=1.8)
    rect(ax, gt_box, color=BOX_OBJ_COLOR, lw=1.4, ls="--")
    allb = np.vstack([box_of(good), box_of(bad), gt_box])
    x1, y1 = allb[:, :2].min(0) - 15
    x2, y2 = allb[:, 2:].max(0) + 15
    ax.set_xlim(max(x1, -0.5), min(x2, image.shape[1] - 0.5))
    ax.set_ylim(min(y2, image.shape[0] - 0.5), max(y1, -0.5))
    image_axes(ax)
    ax.set_title(
        f"{obj_label(obj_id)}: IoU with R {iou(box_of(good), gt_box):.2f}, with Rᵀ {iou(box_of(bad), gt_box):.2f}",
        fontsize=9,
        loc="left",
    )


def pixel_zoom(ax, index, row, half=6):
    """Pixel grid around the rightmost pixel of the full object mask (the right edge of bbox_obj)."""
    x, y, w, h = (int(v) for v in index["bbox_obj"][row])
    scene_dir = os.path.dirname(os.path.dirname(image_path(index, row)))
    mask_path = os.path.join(
        scene_dir, "mask", f"{int(index['im_id'][row]):06d}_{int(index['gt_id'][row]):06d}.png"
    )
    mask = plt.imread(mask_path) > 0
    obj_id = int(index["obj_id"][row])
    uv = project(
        model_points(obj_id, None, "models"), index["R"][row], index["t"][row], index["K"][row]
    )
    cx = x + w  # last mask column (BOP: w = last - first)
    cy = int(np.median(np.flatnonzero(mask[:, cx])))
    ax.imshow(mask, cmap="Greys", vmin=0, vmax=2.5, interpolation="nearest")
    for k in range(-half, half + 2):
        ax.axvline(cx + k - 0.5, color="#d0cfca", lw=0.5)
        ax.axhline(cy + k - 0.5, color="#d0cfca", lw=0.5)
    ax.scatter(
        *to_pixel_index(uv).T, s=7, color=OBJ_COLORS[obj_id], alpha=0.85, linewidths=0, zorder=3
    )
    # Right edge as x + w (a pixel centre, wrong) and as x + w + 1 (the pixel's far edge, used)
    ax.axvline(cx, color=WRONG, lw=1.8, ls=":")
    ax.axvline(cx + 0.5, color=BOX_OBJ_COLOR, lw=2, ls="--")
    ax.set_xlim(cx - 2 * half - 0.5, cx + 2.5)
    ax.set_ylim(cy + half + 0.5, cy - half - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"{obj_label(obj_id)}: rightmost mask pixels, 1 square = 1 pixel", fontsize=9, loc="left"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    train, test = load_index("lmo", "train"), load_index("lmo", "test")

    def inside(ix):
        x, y, w, h = ix["bbox_obj"].T
        return np.flatnonzero(
            (x > 20) & (y > 20) & (x + w < 620) & (y + h < 460) & (ix["visib_fract"] > 0.5)
        )

    picks = [(train, int(rng.choice(inside(train)))) for _ in range(2)] + [
        (test, int(rng.choice(inside(test))))
    ]
    fig = plt.figure(figsize=(16, 10))
    grid = fig.add_gridspec(2, 3, height_ratios=[1, 1.05], hspace=0.18, wspace=0.08)
    for i, (ix, row) in enumerate(picks):
        example(fig.add_subplot(grid[0, i]), ix, row)
    for i, (ix, row) in enumerate(picks):
        pixel_zoom(fig.add_subplot(grid[1, i]), ix, row)
    handles = [
        Line2D(
            [], [], color="#888", lw=1.8, label="model projected with R (correct): points and box"
        ),
        Line2D(
            [],
            [],
            color=WRONG,
            lw=1.5,
            ls=":",
            label="with Rᵀ (wrong) / bottom: edge at x + w (wrong)",
        ),
        Line2D(
            [],
            [],
            color=BOX_OBJ_COLOR,
            lw=1.6,
            ls="--",
            label="BOP bbox_obj / bottom: edge at x + w + 1 (used)",
        ),
        Line2D(
            [],
            [],
            color="#bbb",
            marker="s",
            ls="",
            markersize=8,
            label="full object mask pixels (bottom)",
        ),
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0), labelcolor=TEXT
    )
    fig.suptitle(
        "Reprojection test (top: PBR, PBR, LM-O test) and BOP's pixel convention (bottom)",
        x=0.01,
        ha="left",
        fontsize=12,
    )
    fig.text(
        0.01,
        0.945,
        "Bottom: the projected model reaches the far edge of the last mask pixel, so a BOP box [x, y, w, h] ends at x + w + 1, not x + w.",
        color=TEXT_2,
        fontsize=9,
    )
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.08)
    save(fig, "05_reprojection.png", args.out_dir)
