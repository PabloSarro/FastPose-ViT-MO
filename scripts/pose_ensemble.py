"""M11 B4: average the poses of several models (e.g. the two seeds of the final model) at test time.

For each (scene, image, object) estimate, the poses of the other models are first moved to the
symmetric pose closest to the first model's (R·S_j, t + R·s_j, BOP model symmetries, continuous
ones discretized at 1 % of the diameter), then t is averaged and the mean of the rotation matrices
is projected onto SO(3) (SVD). Inputs are BOP CSVs of the same detections (one estimate per object
and image, as src/bop_evaluate.py writes them); the time column is the sum of the input times, i.e.
running the models one after the other (it includes each run's detector time).

Usage (repo root):
  PYTHONPATH=. .venv/bin/python scripts/pose_ensemble.py --dataset lmo --out OUT/x_lmo-test.csv A.csv B.csv
  bash scripts/bop_eval.sh OUT x_lmo-test.csv
"""

import argparse
import csv
import os

import numpy as np
from bop_toolkit_lib import inout, misc

from src.bop_dataset import SPLITS, default_bop_root


def read(path):
    rows = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            key = (int(r["scene_id"]), int(r["im_id"]), int(r["obj_id"]))
            rows[key] = (float(r["score"]), np.array(r["R"].split(), float).reshape(3, 3),
                         np.array(r["t"].split(), float), float(r["time"]))
    return rows


def project_so3(M):
    U, _, Vt = np.linalg.svd(M)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def angle(Ra, Rb):
    return np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=sorted(SPLITS))
    p.add_argument("--out", required=True, help="output BOP CSV (name: METHOD_DATASET-test.csv)")
    p.add_argument("csvs", nargs="+")
    args = p.parse_args()
    models_ds = SPLITS[args.dataset].get("models_dataset", args.dataset)
    info = inout.load_json(os.path.join(default_bop_root(), models_ds, "models_eval",
                                        "models_info.json"), keys_to_int=True)
    syms = {o: misc.get_symmetry_transformations(info[o], max_sym_disc_step=0.01) for o in info}
    preds = [read(c) for c in args.csvs]
    keys = sorted(set().union(*preds))
    out = []
    for key in keys:
        avail = [pr[key] for pr in preds if key in pr]
        score0, R0, t0, _ = avail[0]
        Rs, ts = [R0], [t0]
        for _, R, t, _ in avail[1:]:
            best = min(syms[key[2]], key=lambda s: angle(R0, R @ s["R"]))
            Rs.append(R @ best["R"])
            ts.append(t + R @ best["t"].reshape(3))
        R = project_so3(np.mean(Rs, axis=0))
        t = np.mean(ts, axis=0)
        score = float(np.mean([a[0] for a in avail]))
        time = float(sum(a[3] for a in avail))
        out.append((key, score, R, t, time))
    # BOP needs the same time for all estimates of an image
    per_image = {}
    for (sc, im, _), _, _, _, time in out:
        per_image[(sc, im)] = max(per_image.get((sc, im), 0.0), time)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("scene_id,im_id,obj_id,score,R,t,time\n")
        for (sc, im, obj), score, R, t, _ in out:
            f.write(f"{sc},{im},{obj},{score},{' '.join(map(str, R.flatten()))},"
                    f"{' '.join(map(str, t))},{per_image[(sc, im)]}\n")
    n_both = sum(all(k in pr for pr in preds) for k in keys)
    print(f"{args.out}: {len(out)} estimates ({n_both} from all {len(preds)} models)")


if __name__ == "__main__":
    main()
