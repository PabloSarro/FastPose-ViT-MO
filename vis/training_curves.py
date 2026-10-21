# This file plots the per-epoch curves of one or more training runs from their TensorBoard logs
# (train loss, validation translation and rotation error), on a log scale
#
#   python vis/training_curves.py RUN_DIR [RUN_DIR ...] --name 10_lmo_m3_overfit.png

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from common import PALETTE, TEXT_2, save


def scalars(run_dir, tag):
    """(epoch, value) pairs of a TensorBoard scalar, over all event files of a run."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    points = {}
    for path in sorted(
        glob.glob(os.path.join(run_dir, "**", "events.out.tfevents.*"), recursive=True)
    ):
        ea = EventAccumulator(path, size_guidance={"scalars": 0})
        ea.Reload()
        if tag in ea.Tags()["scalars"]:
            for e in ea.Scalars(tag):
                points[e.step] = e.value  # a resumed run rewrites later epochs
    return np.array(sorted(points.items()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--name", default="10_training_curves.png")
    parser.add_argument("--title", default=None)
    parser.add_argument(
        "--restarts",
        type=int,
        nargs="*",
        default=[10, 30, 70, 150],
        help="warm-restart epochs to mark",
    )
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()
    labels = args.labels or [os.path.basename(os.path.normpath(r)) for r in args.runs]
    panels = [
        ("Loss/Train/Total", "Training loss", 1.0),
        ("Loss/Validation/Translation", "Validation translation error ‖Δt‖/‖t‖", 1.0),
        ("Loss/Validation/Rotation", "Validation rotation error (degrees)", 180 / np.pi),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, (tag, title, scale) in zip(axes, panels):
        for i, (run, label) in enumerate(zip(args.runs, labels)):
            pts = scalars(run, tag)
            if len(pts):
                ax.plot(pts[:, 0] + 1, pts[:, 1] * scale, color=PALETTE[i], lw=2, label=label)
        for r in args.restarts:
            ax.axvline(r, color=TEXT_2, lw=0.8, ls=":")
        ax.set_yscale("log")
        ax.set_title(title, loc="left")
        ax.set_xlabel("epoch (dotted: learning-rate warm restarts)")
        ax.yaxis.grid(True, which="major")
        ax.set_axisbelow(True)
    if len(args.runs) > 1:
        axes[0].legend()
    fig.suptitle(args.title or ", ".join(labels), x=0.01, ha="left", fontsize=11)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.82, wspace=0.22)
    save(fig, args.name, args.out_dir)
