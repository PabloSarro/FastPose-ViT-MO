# This file contains shared helpers for the visualisation scripts in vis/

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from src.bop_dataset import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    LMO_OBJ_IDS,
    default_bop_root,
)

FIG_DIR = os.path.join(REPO, "vis", "figures")
NAS_HOME = os.environ.get("NAS_HOME", os.path.dirname(REPO))

OBJ_NAMES = {
    1: "ape",
    5: "can",
    6: "cat",
    8: "driller",
    9: "duck",
    10: "eggbox",
    11: "glue",
    12: "holepuncher",
}

# Reference categorical palette (dataviz skill), fixed order: one colour per LM-O object, and
# the same colour for that object in every figure. Identity is always also written as text.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
OBJ_COLORS = dict(zip(LMO_OBJ_IDS, PALETTE))
BOX_OBJ_COLOR = "#2a78d6"  # bbox_obj (whole object)
BOX_VISIB_COLOR = "#eb6834"  # bbox_visib (visible part)
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_2,
        "axes.titlecolor": TEXT,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "xtick.color": TEXT_2,
        "ytick.color": TEXT_2,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.grid": False,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 2,
    }
)


def bop_root() -> str:
    return default_bop_root()


def save(fig, name: str, out_dir: str | None = None) -> str:
    """Save a figure as PNG in vis/figures (or out_dir) and print its path."""
    out_dir = out_dir or FIG_DIR
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(path)
    return path


def obj_label(obj_id: int) -> str:
    return f"{OBJ_NAMES[int(obj_id)]} ({int(obj_id)})"


def denormalize(image_tensor) -> np.ndarray:
    """(3, S, S) normalized tensor -> (S, S, 3) float RGB in [0, 1]."""
    image = image_tensor.permute(1, 2, 0).numpy() * np.array(IMAGENET_STD) + np.array(
        IMAGENET_MEAN
    )
    return image.clip(0, 1)


_POINTS = {}


def model_points(obj_id: int, n: int | None = 1500, kind: str = "models_eval") -> np.ndarray:
    """3D model points (mm) of an LM-O object, subsampled to n for drawing."""
    from bop_toolkit_lib import inout

    key = (int(obj_id), kind)
    if key not in _POINTS:
        path = os.path.join(bop_root(), "lmo", kind, f"obj_{int(obj_id):06d}.ply")
        _POINTS[key] = inout.load_ply(path)["pts"]
    pts = _POINTS[key]
    if n is not None and len(pts) > n:
        pts = pts[np.random.default_rng(0).choice(len(pts), n, replace=False)]
    return pts


def project(points: np.ndarray, R: np.ndarray, t: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Project (N, 3) model points with pose (R, t) and K: (N, 2) in K coordinates."""
    x = K @ (R @ points.T + np.reshape(t, (3, 1)))
    return (x[:2] / x[2]).T


def to_pixel_index(uv: np.ndarray) -> np.ndarray:
    """K coordinates -> image pixel-index coordinates for drawing (BOP pixel i is at i + 0.5)."""
    return uv - 0.5


def draw_axes(ax, R, t, K, length: float, transform=None, lw: float = 2.0):
    """Draw the object's x/y/z axes (red/green/blue) from its origin."""
    origin = np.zeros((1, 3))
    tips = np.eye(3) * length
    uv = project(np.vstack([origin, tips]), R, t, K)
    uv = to_pixel_index(uv) if transform is None else transform(uv)
    for i, color in enumerate(("#e34948", "#008300", "#2a78d6")):
        ax.plot(
            [uv[0, 0], uv[i + 1, 0]],
            [uv[0, 1], uv[i + 1, 1]],
            color=color,
            lw=lw,
            solid_capstyle="round",
        )


def draw_box(ax, xyxy, color, lw=1.5, ls="-", transform=None, label=None):
    """Draw an [x1, y1, x2, y2] box given in K coordinates (edges, see bop_bbox_to_xyxy)."""
    x1, y1, x2, y2 = xyxy
    corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]], dtype=float)
    # Edges in K coordinates are pixel edges: in pixel-index coordinates they sit at -0.5.
    corners = corners - 0.5 if transform is None else transform(corners)
    ax.plot(corners[:, 0], corners[:, 1], color=color, lw=lw, ls=ls, label=label)


def image_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
