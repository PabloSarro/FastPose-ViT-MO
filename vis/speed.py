# This file plots the SPEED reference results of M0 and M1:
# - the published model on SPEED (the golden eval every code change is checked against),
#   as per-image error distributions;
# - the short SPEED training runs before and after the M1 refactor, and the kill/resume run.
#
#   python vis/speed.py

import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from common import NAS_HOME, PALETTE, TEXT_2, save

GOLDEN = os.path.join(NAS_HOME, "results", "eval-rcp-check", "results.json")
RUNS = [
    (
        "smoke-02 (before M1)",
        os.path.join(NAS_HOME, "outputs", "FastPose-ViT", "runs", "smoke-02"),
    ),
    ("smoke-m1-refactor (after refactor)", os.path.join(NAS_HOME, "results", "smoke-m1-refactor")),
    (
        "speed-short-001 (3 epochs)",
        os.path.join(NAS_HOME, "outputs", "FastPose-ViT", "runs", "speed-short-001"),
    ),
    (
        "smoke-m1-resume (killed after epoch 1)",
        os.path.join(NAS_HOME, "results", "smoke-m1-resume"),
    ),
]


def rotation_error_deg(q1, q2):
    dot = np.abs(np.sum(q1 * q2, axis=1)) / (
        np.linalg.norm(q1, axis=1) * np.linalg.norm(q2, axis=1)
    )
    return np.degrees(2 * np.arccos(np.clip(dot, -1, 1)))


def golden_figure():
    with open(GOLDEN) as f:
        data = {k: v for k, v in json.load(f).items() if k != "metrics"}
    pt = np.array([v["prediction"]["translation"] for v in data.values()])
    gt = np.array([v["ground_truth"]["translation"] for v in data.values()])
    pq = np.array([v["prediction"]["rotation"] for v in data.values()])
    gq = np.array([v["ground_truth"]["rotation"] for v in data.values()])
    e_t = np.linalg.norm(pt - gt, axis=1) * 100  # cm
    e_r = rotation_error_deg(pq, gq)
    fig, axes = plt.subplots(1, 3, figsize=(16, 3.8))
    for ax, values, bins, xlabel, title in [
        (
            axes[0],
            e_t,
            np.linspace(0, np.percentile(e_t, 99), 41),
            "cm",
            "Translation error ‖t̂ − t‖",
        ),
        (axes[1], e_r, np.linspace(0, np.percentile(e_r, 99), 41), "degrees", "Rotation error"),
    ]:
        ax.hist(values, bins=bins, color=PALETTE[0], rwidth=0.85)
        for stat, ls in ((np.median(values), "-"), (np.mean(values), ":")):
            ax.axvline(stat, color=TEXT_2, lw=1.2, ls=ls)
        ax.set_title(
            f"{title}: median {np.median(values):.2f}, mean {np.mean(values):.2f} {xlabel}",
            loc="left",
        )
        ax.set_xlabel(f"{xlabel} (top 1 % not shown; solid line median, dotted mean)")
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("test images")
    ax = axes[2]
    ax.scatter(gt[:, 2], e_t, s=6, color=PALETTE[0], alpha=0.5, linewidths=0)
    ax.set_title("Translation error grows with distance", loc="left")
    ax.set_xlabel("distance to the spacecraft t_z (m)")
    ax.set_ylabel("translation error (cm)")
    ax.set_ylim(0, np.percentile(e_t, 99.5))
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    fig.suptitle(
        f"Golden SPEED eval: published ViT-B/16-384 on {len(e_t):,} SPEED test images (SPEED score 0.0093)",
        x=0.01,
        ha="left",
        fontsize=11,
    )
    fig.subplots_adjust(left=0.05, right=0.99, top=0.8, wspace=0.22)
    return fig


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
    return sorted(points.items())


def training_figure():
    fig, axes = plt.subplots(1, 3, figsize=(16, 3.9))
    tags = [
        ("Loss/Validation/Total", "Validation loss (SPEED score on val)"),
        ("Loss/Train/Total", "Training loss"),
        ("LearningRate", "Learning rate at the start of each epoch"),
    ]
    for ax, (tag, title) in zip(axes, tags):
        for i, (name, run_dir) in enumerate(RUNS):
            pts = scalars(run_dir, tag)
            if not pts:
                continue
            x = [p[0] + 1 for p in pts]
            y = [p[1] for p in pts]
            ax.plot(
                x,
                y,
                color=PALETTE[i],
                marker="o",
                markersize=6,
                lw=2,
                label=name,
                ls="--" if i == 3 else "-",
            )
        ax.set_title(title, loc="left")
        ax.set_xlabel("epoch")
        ax.set_xticks([1, 2, 3])
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
    for ax in axes:
        ax.axvline(1.5, color=TEXT_2, lw=1, ls=":")
    axes[1].text(
        1.53,
        axes[1].get_ylim()[1] * 0.97,
        "smoke-m1-resume killed here,\nthen resumed from last.pt",
        fontsize=8,
        color=TEXT_2,
        va="top",
    )
    axes[2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[0].legend(loc="lower left", fontsize=7.5)
    fig.suptitle(
        "Short SPEED training runs: the M1 refactor and the kill/resume run behave like the runs before them",
        x=0.01,
        ha="left",
        fontsize=11,
    )
    fig.subplots_adjust(left=0.05, right=0.99, top=0.8, wspace=0.22)
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()
    save(golden_figure(), "08_speed_golden.png", args.out_dir)
    save(training_figure(), "09_m1_training_runs.png", args.out_dir)
