# This file shows what the training augmentations do to the same crops: bbox jitter alone, the
# "standard" colour pipeline (default) and the strong "domain_gap" one (open decision D5)
#
#   python vis/augmentations.py

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from common import TEXT_2, denormalize, image_axes, obj_label, save

from src.bop_dataset import BOPDataset, load_index

COLUMNS = [
    ("no augmentation", dict(augment=False)),
    ("bbox jitter", dict(augment=True, pixel_aug="none", bbox_jitter_prob=1.0)),
    ("bbox jitter", dict(augment=True, pixel_aug="none", bbox_jitter_prob=1.0)),
    ("standard colour", dict(augment=True, pixel_aug="standard", bbox_jitter_percent=0.0)),
    ("standard colour", dict(augment=True, pixel_aug="standard", bbox_jitter_percent=0.0)),
    ("standard colour", dict(augment=True, pixel_aug="standard", bbox_jitter_percent=0.0)),
    ("domain_gap colour", dict(augment=True, pixel_aug="domain_gap", bbox_jitter_percent=0.0)),
    ("domain_gap colour", dict(augment=True, pixel_aug="domain_gap", bbox_jitter_percent=0.0)),
    ("domain_gap colour", dict(augment=True, pixel_aug="domain_gap", bbox_jitter_percent=0.0)),
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects", type=int, nargs="+", default=[1, 8, 10, 11])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    index = load_index("lmo", "train")
    datasets = [BOPDataset(index=index, img_size=160, **kwargs) for _, kwargs in COLUMNS]
    rng = np.random.default_rng(args.seed)
    fig, axes = plt.subplots(
        len(args.objects), len(COLUMNS), figsize=(16, 1.9 * len(args.objects) + 0.8)
    )
    for r, obj_id in enumerate(args.objects):
        rows = np.flatnonzero(
            (index["obj_id"] == obj_id)
            & (index["visib_fract"] > 0.8)
            & (index["bbox_visib"][:, 2] > 70)
        )
        row = int(rng.choice(rows))
        for c, ((name, _), ds) in enumerate(zip(COLUMNS, datasets)):
            ax = axes[r, c]
            ax.imshow(denormalize(ds[row]["images"]))
            image_axes(ax)
            if r == 0:
                ax.set_title(name, fontsize=9)
        axes[r, 0].set_ylabel(obj_label(obj_id), color=TEXT_2)
    fig.suptitle(
        "Training augmentations on the same crops (each column is an independent random draw)",
        x=0.01,
        ha="left",
        fontsize=11,
    )
    fig.subplots_adjust(left=0.03, right=0.99, wspace=0.04, hspace=0.04, top=0.9)
    save(fig, "04_augmentations.png", args.out_dir)
