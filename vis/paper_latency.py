"""Paper figure: latency per image vs number of objects (LM-O model, A100).

Reads the latency.json files written by optimization/bop_latency.py and writes
paper/figures/latency.pdf (+ .png). Series: PyTorch FP16 pipeline (384 px), TensorRT FP16
pipeline (384 / 224 px) and the TensorRT pipeline plus our LW-DETR detector (384 px).

Usage (repo root): .venv/bin/python vis/paper_latency.py [--root $NAS_HOME/results/m9/det-small-trt]
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Categorical slots 1-4 of the reference palette (fixed order); each series also has its own
# marker and a legend entry, so identity never relies on color alone.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default=os.path.join(os.environ.get("NAS_HOME", ""), "results/m9/det-small-trt")
    )
    parser.add_argument("--out", default="paper/figures/latency")
    args = parser.parse_args()
    r384 = json.load(open(os.path.join(args.root, "latency_384/latency.json")))
    r224 = json.load(open(os.path.join(args.root, "latency_224/latency.json")))
    ns = r384["n_objects"]
    det = r384["detector_ms"]

    def pipe(report, backend):
        return [report["pipeline"][backend][str(n)]["median_ms"] for n in ns]

    series = [
        ("PyTorch FP16, 384 px", pipe(r384, "torch_amp"), BLUE, "o"),
        ("TensorRT FP16, 384 px", pipe(r384, "trt_fp16"), ORANGE, "s"),
        ("TensorRT FP16, 224 px", pipe(r224, "trt_fp16"), AQUA, "^"),
        ("TensorRT 384 px + our detector", [v + det for v in pipe(r384, "trt_fp16")], YELLOW, "D"),
    ]

    plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.edgecolor": MUTED})
    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    for label, ys, color, marker in series:
        ax.plot(ns, ys, color=color, lw=2, marker=marker, ms=4.5, label=label, zorder=3,
                markeredgecolor="white", markeredgewidth=0.8)
    ax.axhline(1000 / 30, color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.text(16.6, 1000 / 30 - 0.6, "30 FPS", fontsize=7, color=MUTED, va="top", ha="right")
    ax.set_xticks(ns)
    ax.set_xlim(0, 17)
    ax.set_ylim(0, 36)
    ax.set_xlabel("Objects per image $N$", color=INK)
    ax.set_ylabel("Latency per image [ms]", color=INK)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.45, -0.25),
              ncol=2, handlelength=2.0, columnspacing=1.0)
    fig.tight_layout(pad=0.3)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out + ".pdf")
    fig.savefig(args.out + ".png", dpi=200)
    print(f"wrote {args.out}.pdf / .png (detector {det:.1f} ms)")


if __name__ == "__main__":
    main()
