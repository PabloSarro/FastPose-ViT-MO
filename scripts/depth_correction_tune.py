"""Audit W7: sensitivity of the RGB-D depth correction to its settings, on VALIDATION data.

The settings of scripts/depth_correction.py (agreement bands, number of iterations, minimum
agreeing points, z-buffer cell, visibility threshold) were set a priori. Here the RGB predictions
of the final LM-O model on the LM real validation images (200 per object, GT boxes; depth images
available) are corrected with each setting, one change at a time around the defaults, and scored
with the official BOP19 AR (VSD, MSSD, MSPD) on `test_targets_bop19_lmo8.json`. The choice is
made here, on validation; the test set is then read once for the chosen setting.

Usage (repo root, GPU, EGL for VSD):
  PYTHONPATH=. .venv/bin/python scripts/depth_correction_tune.py RGB_VAL_CSV OUT_DIR
  (RGB_VAL_CSV: .../eval_bop_val/fastposevitmo_lmo-val_real.csv of the final model)
"""
import json
import os
import subprocess
import sys

DEFAULT = dict(bands=[0.2, 0.1, 0.05], min_pixels=50, cell=4, vis_eps=0.05)
CONFIGS = {
    "default": {},
    "bands_0.1_0.05": dict(bands=[0.1, 0.05]),
    "bands_0.3_0.15_0.05": dict(bands=[0.3, 0.15, 0.05]),
    "bands_0.2_one_iteration": dict(bands=[0.2]),
    "bands_0.05_x3": dict(bands=[0.05, 0.05, 0.05]),
    "bands_five_iterations": dict(bands=[0.2, 0.1, 0.05, 0.05, 0.05]),
    "min_points_20": dict(min_pixels=20),
    "min_points_100": dict(min_pixels=100),
    "min_points_300": dict(min_pixels=300),
    "cell_2": dict(cell=2),
    "cell_8": dict(cell=8),
    "vis_eps_0.02": dict(vis_eps=0.02),
    "vis_eps_0.10": dict(vis_eps=0.10),
}


def main() -> None:
    csv_in, out_dir = sys.argv[1], sys.argv[2]
    results = {}
    for name, change in CONFIGS.items():
        cfg = {**DEFAULT, **change}
        out = os.path.join(out_dir, name)
        os.makedirs(out, exist_ok=True)
        method_csv = os.path.join(out, "fastposevitmodepth_lm-test.csv")
        if not os.path.exists(os.path.join(out, "DONE")):
            cmd = [sys.executable, "scripts/depth_correction.py", "--dataset", "lmo",
                   "--data_name", "lm", "--csv", csv_in, "--out", method_csv,
                   "--bands", *map(str, cfg["bands"]), "--min_pixels", str(cfg["min_pixels"]),
                   "--cell", str(cfg["cell"]), "--vis_eps", str(cfg["vis_eps"])]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
            with open(os.path.join(out, "eval.log"), "w") as log:
                subprocess.run(["bash", "scripts/bop_eval.sh", out, "fastposevitmodepth_lm-test.csv",
                                "--targets_filename=test_targets_bop19_lmo8.json"],
                               check=True, stdout=log, stderr=subprocess.STDOUT)
            open(os.path.join(out, "DONE"), "w").close()
        score = None
        for root, _, files in os.walk(out):
            if "scores_bop19.json" in files:
                with open(os.path.join(root, "scores_bop19.json")) as f:
                    d = json.load(f)
                score = {k: d[k] for k in ("bop19_average_recall", "bop19_average_recall_vsd",
                                           "bop19_average_recall_mssd", "bop19_average_recall_mspd")}
        results[name] = {"config": cfg, **(score or {})}
        print(name, cfg, score and round(score["bop19_average_recall"], 4), flush=True)
    with open(os.path.join(out_dir, "tune.json"), "w") as f:
        json.dump(results, f, indent=1)
    print("rgb baseline = no correction: see the RGB validation eval of the same model")


if __name__ == "__main__":
    main()
