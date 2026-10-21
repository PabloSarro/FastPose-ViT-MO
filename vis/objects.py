# This file draws the 8 LM-O objects: one real crop (LM, well visible) and one synthetic PBR crop
# each, with BOP id, diameter and symmetry
#
#   python vis/objects.py

import argparse

import matplotlib.pyplot as plt
import numpy as np
from common import OBJ_COLORS, TEXT_2, denormalize, image_axes, obj_label, save

from src.bop_dataset import LMO_OBJ_IDS, BOPDataset, load_index

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()
    real = BOPDataset("lmo", "val_real", img_size=224, crop_context=True)
    pbr_index = load_index("lmo", "train")
    pbr = BOPDataset(index=pbr_index, augment=False, img_size=224, crop_context=True)
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(2, 8, figsize=(16, 4.9))
    for col, obj_id in enumerate(LMO_OBJ_IDS):
        rows = np.flatnonzero(real.index["obj_id"] == obj_id)
        real_row = rows[np.argmax(real.index["px_count_visib"][rows])]
        rows = np.flatnonzero(
            (pbr_index["obj_id"] == obj_id)
            & (pbr_index["visib_fract"] > 0.95)
            & (pbr_index["bbox_visib"][:, 2] > 90)
        )
        pbr_row = rng.choice(rows)
        for r, (ds, row) in enumerate(((real, real_row), (pbr, pbr_row))):
            ax = axes[r, col]
            ax.imshow(denormalize(ds[int(row)]["images"]))
            image_axes(ax)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(OBJ_COLORS[obj_id])
                spine.set_linewidth(2.5)
        info = real.index
        diameter = info["diameters"][LMO_OBJ_IDS.index(obj_id)]
        sym = "  ·  symmetric" if info["symmetric"][LMO_OBJ_IDS.index(obj_id)] else ""
        axes[0, col].set_title(f"{obj_label(obj_id)}\n{diameter:.0f} mm{sym}", fontsize=9)
    axes[0, 0].set_ylabel("real (LM)", color=TEXT_2, fontsize=10)
    axes[1, 0].set_ylabel("synthetic (PBR)", color=TEXT_2, fontsize=10)
    fig.suptitle(
        "The 8 LM-O objects: name (BOP id), diameter; real LM photo vs synthetic training render",
        x=0.01,
        ha="left",
        fontsize=11,
    )
    fig.subplots_adjust(left=0.03, right=0.99, wspace=0.05, hspace=0.05, top=0.83)
    save(fig, "01_objects.png", args.out_dir)
