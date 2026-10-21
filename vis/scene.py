# This file draws one full BOP image with every indexed instance: bbox_obj, bbox_visib,
# the model points projected with the GT pose and the object axes
#
#   python vis/scene.py                       # a PBR training image and an LM-O test image
#   python vis/scene.py --split test --scene 2 --im 3

import argparse

import matplotlib.pyplot as plt
import numpy as np
from common import (
    BOX_OBJ_COLOR,
    BOX_VISIB_COLOR,
    OBJ_COLORS,
    TEXT,
    draw_axes,
    draw_box,
    image_axes,
    model_points,
    obj_label,
    project,
    save,
    to_pixel_index,
)
from matplotlib.lines import Line2D

from src.bop_dataset import bop_bbox_to_xyxy, image_path, load_index


def pick_image(index, rng) -> tuple[int, int, int]:
    """An image with many instances, some of them occluded."""
    keys = np.stack([index["source"], index["scene_id"], index["im_id"]], 1)
    uniq, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    min_visib = np.ones(len(uniq))
    np.minimum.at(min_visib, inverse, index["visib_fract"])
    candidates = np.flatnonzero((counts >= np.percentile(counts, 90)) & (min_visib < 0.6))
    if len(candidates) == 0:
        candidates = np.arange(len(uniq))
    return tuple(int(v) for v in uniq[rng.choice(candidates)])


def draw_scene(index, source, scene_id, im_id, title, ax, zoom=False, labels=True):
    rows = np.flatnonzero(
        (index["source"] == source) & (index["scene_id"] == scene_id) & (index["im_id"] == im_id)
    )
    image = plt.imread(image_path(index, rows[0]))
    ax.imshow(image)
    for row in rows:
        obj_id = int(index["obj_id"][row])
        R, t, K = index["R"][row], index["t"][row], index["K"][row]
        uv = to_pixel_index(project(model_points(obj_id, 800), R, t, K))
        ax.scatter(uv[:, 0], uv[:, 1], s=0.6, color=OBJ_COLORS[obj_id], alpha=0.35, linewidths=0)
        draw_box(ax, bop_bbox_to_xyxy(index["bbox_obj"][row]), BOX_OBJ_COLOR, lw=1.2, ls="--")
        draw_box(ax, bop_bbox_to_xyxy(index["bbox_visib"][row]), BOX_VISIB_COLOR, lw=1.6)
        draw_axes(ax, R, t, K, length=0.4 * index["diameters"][index["obj_idx"][row]], lw=1.6)
        x, y = index["bbox_visib"][row][:2]
        if not labels:
            continue
        ax.text(
            x,
            y - 4,
            f"{obj_label(obj_id)}  {100 * index['visib_fract'][row]:.0f}% visible",
            color=TEXT,
            fontsize=7.5,
            bbox=dict(
                boxstyle="round,pad=0.2", fc="white", ec=OBJ_COLORS[obj_id], lw=1.2, alpha=0.9
            ),
        )
    if zoom:
        boxes = np.array([bop_bbox_to_xyxy(index["bbox_obj"][r]) for r in rows])
        x1, y1 = boxes[:, :2].min(0) - 25
        x2, y2 = boxes[:, 2:].max(0) + 25
        ax.set_xlim(max(x1, -0.5), min(x2, image.shape[1] - 0.5))
        ax.set_ylim(min(y2, image.shape[0] - 0.5), max(y1, -0.5))
        ax.set_title("Zoom on the objects", loc="left")
    else:
        ax.set_xlim(-0.5, image.shape[1] - 0.5)
        ax.set_ylim(image.shape[0] - 0.5, -0.5)
        ax.set_title(
            f"{title}: scene {scene_id}, image {im_id}, {len(rows)} instances", loc="left"
        )
    image_axes(ax)


def legend(fig):
    handles = [
        Line2D(
            [],
            [],
            color=BOX_OBJ_COLOR,
            ls="--",
            lw=1.4,
            label="bbox_obj: whole object, incl. hidden parts",
        ),
        Line2D(
            [],
            [],
            color=BOX_VISIB_COLOR,
            lw=1.6,
            label="bbox_visib: visible pixels only (what we crop)",
        ),
        Line2D(
            [],
            [],
            color="#888",
            marker=".",
            ls="",
            label="3D model points projected with the GT pose",
        ),
        Line2D([], [], color="#e34948", lw=2, label="object x axis"),
        Line2D([], [], color="#008300", lw=2, label="y axis"),
        Line2D([], [], color="#2a78d6", lw=2, label="z axis"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "val_real", "test"], default=None)
    parser.add_argument("--scene", type=int, default=None)
    parser.add_argument("--im", type=int, default=None)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    splits = [args.split] if args.split else ["train", "test"]
    names = {
        "train": "LM PBR (synthetic, training)",
        "val_real": "LM real (validation)",
        "test": "LM-O test (real)",
    }
    for split in splits:
        index = load_index("lmo", split)
        if args.scene is not None and args.im is not None:
            source = int(index["source"][np.flatnonzero(index["scene_id"] == args.scene)[0]])
            key = (source, args.scene, args.im)
        else:
            key = pick_image(index, rng)
        fig, axes = plt.subplots(1, 2, figsize=(16, 6.2), gridspec_kw={"width_ratios": [1, 1.25]})
        draw_scene(index, *key, names[split], axes[0], labels=False)
        draw_scene(index, *key, names[split], axes[1], zoom=True)
        fig.subplots_adjust(wspace=0.04, bottom=0.12)
        legend(fig)
        suffix = "" if args.scene is None else f"_{key[1]:06d}_{key[2]:06d}"
        save(fig, f"02_scene_{split}{suffix}.png", args.out_dir)
        if args.out_dir is None and args.scene is None:
            print(
                f"  ({split}: scene {key[1]}, image {key[2]}; rerun with --scene/--im for another image)"
            )
