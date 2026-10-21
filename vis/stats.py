# This file plots statistics of the LM-O index: instances per object and split, visibility,
# box sizes, and the translation labels U (why the "site" depth mode replaced "legacy" for BOP)
#
#   python vis/stats.py

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from common import OBJ_NAMES, PALETTE, TEXT_2, save

from src.bop_dataset import LMO_OBJ_IDS, bop_bbox_to_xyxy, load_index
from src.camera import Camera
from src.utils import translation_to_bbox_relative_translation

SPLITS = [
    ("train", "train: LM PBR scenes 0–47"),
    ("val_pbr", "val_pbr: PBR scenes 48–49"),
    ("val_real", "val_real: LM real"),
    ("test", "test: LM-O (BOP19 targets)"),
]
TRAIN_COLOR, TEST_COLOR = PALETTE[0], PALETTE[1]


def counts_figure(indices):
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.4))
    names = [f"{OBJ_NAMES[o]}\n{o}" for o in LMO_OBJ_IDS]
    for ax, (split, title) in zip(axes, SPLITS):
        counts = np.bincount(indices[split]["obj_idx"], minlength=8)
        bars = ax.bar(names, counts, color=PALETTE[0], width=0.7)
        ax.bar_label(
            bars,
            labels=[f"{c / 1000:.1f}k" if c >= 10000 else str(c) for c in counts],
            fontsize=7,
            color=TEXT_2,
        )
        ax.set_title(f"{title}\n{counts.sum():,} instances", loc="left", fontsize=10)
        ax.tick_params(axis="x", labelsize=7)
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("instances")
    fig.suptitle(
        "Instances per object and split (each panel has its own scale)",
        x=0.01,
        ha="left",
        fontsize=11,
    )
    fig.subplots_adjust(left=0.05, right=0.99, top=0.78, bottom=0.17, wspace=0.25)
    return fig


def labels_U(index, rows, depth_mode):
    bbox = torch.tensor(
        np.array([bop_bbox_to_xyxy(b) for b in index["bbox_visib"][rows]]), dtype=torch.float64
    )
    return translation_to_bbox_relative_translation(
        torch.from_numpy(index["t"][rows]),
        bbox,
        Camera,
        K=torch.from_numpy(index["K"][rows]),
        img_wh=torch.from_numpy(index["img_wh"][rows]).double(),
        depth_mode=depth_mode,
        diameter=torch.from_numpy(index["diameters"][index["obj_idx"][rows]]),
    ).numpy()


def step_hist(ax, values, bins, color, label):
    hist, edges = np.histogram(values, bins=bins)
    ax.stairs(hist / hist.sum(), edges, color=color, lw=2, label=label)


def distributions_figure(indices, n=40000, seed=0):
    rng = np.random.default_rng(seed)
    train, test = indices["train"], indices["test"]
    tr = rng.choice(len(train["source"]), min(n, len(train["source"])), replace=False)
    te = np.arange(len(test["source"]))
    fig, axes = plt.subplots(2, 3, figsize=(16, 7.4))

    ax = axes[0, 0]
    bins = np.linspace(0, 1, 41)
    step_hist(ax, train["visib_fract"][tr], bins, TRAIN_COLOR, "train (PBR)")
    step_hist(ax, test["visib_fract"], bins, TEST_COLOR, "test (LM-O)")
    ax.axvline(0.1, color=TEXT_2, lw=1, ls=":")
    ax.text(0.11, ax.get_ylim()[1] * 0.92, "train threshold 0.1", fontsize=8, color=TEXT_2)
    ax.set_title("Visible fraction of each instance", loc="left")
    ax.set_xlabel("visib_fract")
    ax.set_ylabel("share of instances")
    ax.legend()

    ax = axes[0, 1]
    size = lambda ix, r: np.sqrt((ix["bbox_visib"][r, 2] + 1.0) * (ix["bbox_visib"][r, 3] + 1.0))  # noqa: E731
    bins = np.linspace(0, 300, 41)
    step_hist(ax, size(train, tr), bins, TRAIN_COLOR, "train (PBR)")
    step_hist(ax, size(test, te), bins, TEST_COLOR, "test (LM-O)")
    ax.set_title("Size of the crop box (√(w·h) of bbox_visib)", loc="left")
    ax.set_xlabel("pixels (image is 640 × 480)")
    ax.legend()

    ax = axes[0, 2]
    bins = np.linspace(250, 2000, 41)
    step_hist(ax, train["t"][tr, 2], bins, TRAIN_COLOR, "train (PBR)")
    step_hist(ax, test["t"][:, 2], bins, TEST_COLOR, "test (LM-O)")
    ax.set_title("Object distance t_z", loc="left")
    ax.set_xlabel("mm")
    ax.legend()

    U_site_tr, U_site_te = labels_U(train, tr, "site"), labels_U(test, te, "site")
    U_leg_tr = labels_U(train, tr, "legacy")

    ax = axes[1, 0]
    bins = np.linspace(-0.6, 0.6, 49)
    step_hist(ax, U_site_tr[:, 0], bins, PALETTE[0], "U_x")
    step_hist(ax, U_site_tr[:, 1], bins, PALETTE[2], "U_y")
    ax.set_title("U_x, U_y: object centre relative to the box (train)", loc="left")
    ax.set_xlabel("offset from the box centre, in box widths/heights")
    ax.set_ylabel("share of instances")
    ax.legend()

    ax = axes[1, 1]
    bins = np.linspace(0, np.percentile(U_leg_tr[:, 2], 99.5), 41)
    step_hist(ax, U_leg_tr[:, 2], bins, PALETTE[3], "legacy (train)")
    ax.set_title("U_z, legacy (SPEED) depth: t_z·(w+h)/(2H)", loc="left")
    ax.set_xlabel(f"U_z: median {np.median(U_leg_tr[:, 2]):.0f}, about 100× larger than U_x, U_y")
    ax.legend()

    ax = axes[1, 2]
    bins = np.linspace(0, 1.6, 41)
    step_hist(ax, U_site_tr[:, 2], bins, TRAIN_COLOR, "site (train)")
    step_hist(ax, U_site_te[:, 2], bins, TEST_COLOR, "site (test)")
    ax.set_title("U_z, site depth: t_z·s/(f·D), dimensionless", loc="left")
    ax.set_xlabel("U_z: around 1, same scale for every object and camera")
    ax.legend()

    for ax in axes.ravel():
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
    fig.suptitle(
        f"LM-O index distributions ({len(tr):,} random training instances, all {len(te):,} test targets)",
        x=0.01,
        ha="left",
        fontsize=11,
    )
    fig.subplots_adjust(left=0.05, right=0.99, top=0.9, hspace=0.42, wspace=0.2)
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()
    indices = {split: load_index("lmo", split) for split, _ in SPLITS}
    save(counts_figure(indices), "06_index_counts.png", args.out_dir)
    save(distributions_figure(indices), "07_index_distributions.png", args.out_dir)
