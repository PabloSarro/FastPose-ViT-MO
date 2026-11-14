"""Supplementary figure: relative depth error (t_z,est - t_z,gt) / t_z,gt over all test instances,
LM-O vs YCB-V, from the per-instance errors written by vis/paper_qualitative.py
(paper/figures/errors_<dataset>.csv) and the ground-truth depth of each instance.

Usage (repo root): .venv/bin/python vis/paper_depth_bias.py
Writes paper/figures/depth_bias.pdf (+ .png) and prints the summary numbers.
"""

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from bop_toolkit_lib import inout  # noqa: E402

from src.bop_dataset import default_bop_root  # noqa: E402

# Categorical slots 1-2 of the reference palette; the two series are also told apart by line style
# and direct labels.
COLORS = {"lmo": "#2a78d6", "ycbv": "#eb6834"}
NAMES = {"lmo": "LM-O", "ycbv": "YCB-V"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def relative_depth_errors(dataset):
    root = os.path.join(default_bop_root(), dataset, "test")
    gts, rel = {}, []
    for r in csv.DictReader(open(f"paper/figures/errors_{dataset}.csv")):
        if r["dz_mm"] == "nan":
            continue
        sc, im, obj = int(r["scene_id"]), int(r["im_id"]), int(r["obj_id"])
        if sc not in gts:
            gts[sc] = inout.load_json(os.path.join(root, f"{sc:06d}", "scene_gt.json"),
                                      keys_to_int=True)
        tz = [g["cam_t_m2c"][2] for g in gts[sc][im] if g["obj_id"] == obj]
        if len(tz) == 1:  # one instance per object and image on LM-O / YCB-V
            rel.append(100.0 * float(r["dz_mm"]) / tz[0])
    return np.array(rel)


def main() -> None:
    plt.rcParams.update({"font.family": "serif", "font.size": 8})
    fig, ax = plt.subplots(figsize=(3.3, 2.1))
    bins = np.linspace(-15, 15, 61)
    for ds, ls in (("lmo", "-"), ("ycbv", (0, (4, 1.5)))):
        rel = relative_depth_errors(ds)
        med = float(np.median(rel))
        print(f"{ds}: n={len(rel)} median {med:+.2f} % | too far {100 * (rel > 0).mean():.1f} % "
              f"| |err| median {np.median(np.abs(rel)):.2f} %")
        h, _ = np.histogram(np.clip(rel, bins[0], bins[-1]), bins=bins)
        ax.stairs(h / len(rel) * 100, bins, color=COLORS[ds], lw=1.8, ls=ls,
                  label=f"{NAMES[ds]} (median {med:+.1f} %)")
        ax.axvline(med, color=COLORS[ds], lw=1, ls=ls)
    ax.axvline(0, color=MUTED, lw=0.6)
    ax.set_xlabel("Relative depth error $(\\hat t_z - t_z)/t_z$ [%]", color=INK)
    ax.set_ylabel("Instances [%]", color=INK)
    ax.set_xlim(bins[0], bins[-1])
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout(pad=0.3)
    fig.savefig("paper/figures/depth_bias.pdf")
    fig.savefig("paper/figures/depth_bias.png", dpi=200)
    print("wrote paper/figures/depth_bias.pdf / .png")


if __name__ == "__main__":
    main()
