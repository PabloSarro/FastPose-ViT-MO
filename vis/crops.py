# This file draws a grid of model-input crops exactly as BOPDataset returns them, with the GT
# model points warped into the crop and the pose decoded back from the labels
#
# Each tile shows:
# - the crop (denormalized; training crops include bbox jitter and colour augmentation),
# - small dots: 3D model points projected with the GT pose (R_m2c, t_m2c) and warped into the crop,
# - axes and a cross: the pose recovered from the labels (U, allocentric R) with the inverse
#   functions the model's predictions go through. Axes on the object and the cross on its centre
#   mean crop, labels and inverse transforms agree.
#
#   python vis/crops.py                     # train (augmented) and test, 4 crops per object
#   python vis/crops.py --split val_real --per_object 8

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from common import (
    OBJ_COLORS,
    TEXT_2,
    denormalize,
    image_axes,
    model_points,
    obj_label,
    project,
    save,
)

from src.bop_dataset import LMO_OBJ_IDS, BOPDataset, crop_warp_matrix
from src.camera import Camera
from src.utils import (
    bbox_relative_translation_to_translation,
    get_absolute_orientation,
    rotation_matrix_to_quaternion,
)
from src.utils import quaternion_to_rotation_matrix


def decode_pose(sample, depth_mode="site"):
    """Absolute pose (R, t) from the sample's labels, as the evaluation does for predictions."""
    U = sample["translations"].double().unsqueeze(0)
    bbox = sample["bboxes"].double().unsqueeze(0)
    t = bbox_relative_translation_to_translation(
        U,
        bbox,
        Camera,
        K=sample["K"].double(),
        depth_mode=depth_mode,
        diameter=sample["diameter"].double(),
    )
    q = get_absolute_orientation(
        t, rotation_matrix_to_quaternion(sample["rotations"].double().unsqueeze(0))
    )
    return quaternion_to_rotation_matrix(q)[0].numpy(), t[0].numpy()


def to_crop(uv, sample, size):
    """K coordinates -> crop pixel-index coordinates (the warp the dataset applied)."""
    M = crop_warp_matrix(sample["bboxes"].tolist(), size)
    return (uv - 0.5) @ M[:, :2].T + M[:, 2]


def draw_tile(ax, sample, size):
    obj_id = sample["obj_id"]
    ax.imshow(denormalize(sample["images"]))
    K = sample["K"].double().numpy()
    uv = to_crop(
        project(
            model_points(obj_id, 600),
            sample["R_m2c"].double().numpy(),
            sample["t_m2c"].double().numpy(),
            K,
        ),
        sample,
        size,
    )
    ax.scatter(uv[:, 0], uv[:, 1], s=2.5, color=OBJ_COLORS[obj_id], alpha=0.7, linewidths=0)
    R, t = decode_pose(sample)
    length = 0.35 * sample["diameter"].item()
    pts = to_crop(project(np.vstack([np.zeros(3), np.eye(3) * length]), R, t, K), sample, size)
    for i, color in enumerate(("#e34948", "#008300", "#2a78d6")):
        ax.plot(
            pts[[0, i + 1], 0], pts[[0, i + 1], 1], color=color, lw=1.8, solid_capstyle="round"
        )
    ax.plot(*pts[0], marker="+", color="white", markersize=9, mew=2.2)
    ax.plot(*pts[0], marker="+", color="black", markersize=7, mew=1.0)
    ax.set_xlim(-0.5, size - 0.5)
    ax.set_ylim(size - 0.5, -0.5)
    image_axes(ax)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(OBJ_COLORS[obj_id])
        spine.set_linewidth(2)
    ax.set_title(f"{obj_label(obj_id)} · {100 * sample['visib_fract']:.0f}%", fontsize=7.5, pad=2)


def grid(split, per_object, size, seed, augment):
    torch.manual_seed(seed)
    np.random.seed(seed)
    ds = BOPDataset("lmo", split, img_size=size, augment=augment)
    rng = np.random.default_rng(seed)
    rows = [
        rng.choice(np.flatnonzero(ds.index["obj_id"] == o), per_object, replace=False)
        for o in LMO_OBJ_IDS
    ]
    fig, axes = plt.subplots(per_object, 8, figsize=(16, 2.15 * per_object + 0.9))
    axes = np.atleast_2d(axes)
    for col, obj_rows in enumerate(rows):
        for r, row in enumerate(obj_rows):
            sample = ds[int(row)]
            sample["visib_fract"] = float(ds.index["visib_fract"][row])
            draw_tile(axes[r, col], sample, size)
    what = "augmented (jitter + colour)" if ds.augment else "no augmentation"
    fig.suptitle(
        f"Model-input crops, {split} split, {what}. Dots: GT model points. Axes/cross: pose decoded from the labels.",
        x=0.01,
        ha="left",
        fontsize=11,
        color=TEXT_2,
    )
    fig.subplots_adjust(
        left=0.01, right=0.99, wspace=0.06, hspace=0.18, top=1 - 0.75 / (2.15 * per_object + 0.9)
    )
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "val_pbr", "val_real", "test"], default=None)
    parser.add_argument("--per_object", type=int, default=4)
    parser.add_argument("--size", type=int, default=192)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no_augment", action="store_true", help="Training crops without augmentation"
    )
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()
    for split in [args.split] if args.split else ["train", "test"]:
        augment = None if not args.no_augment else False
        fig = grid(split, args.per_object, args.size, args.seed, augment)
        save(fig, f"03_crops_{split}.png", args.out_dir)
