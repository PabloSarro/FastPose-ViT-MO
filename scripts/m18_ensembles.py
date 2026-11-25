"""M18.7: ensembles of trained pose models on LM-O, evaluation only (no training).

Two ways to combine several models:
  pool     dense models: the 2D-3D correspondences of all members go into one PnP-RANSAC + LM
           (the DR pose of the first member is the fallback);
  average  any models: their final poses are averaged per target (scripts/pose_ensemble.py),
           from the result CSVs the runs stored.
Every ensemble is scored like a single run: the noisy-box LM real validation (own score,
scripts/val_score_csv.py) and the LM-O test with the default detections (official BOP19 eval).
A pool of one model must reproduce that run's stored numbers.

Spec (default: the one below): {"models": {NAME: {"dir": VARIANT_DIR, "dense": true, "vit": "vit_b_16_384"}},
              "ensembles": {NAME: {"mode": "pool" | "average", "members": [NAME, ...]}}}
`dir` is relative to $NAS_HOME/outputs/FastPose-ViT/runs. Finished ensembles
(OUT/<name>/result.json) are skipped, so an interrupted call can be repeated.

Usage (repo root, in dev):
  PYTHONPATH=. .venv/bin/python scripts/m18_ensembles.py --out OUT [--spec SPEC.json]
"""
import argparse
import glob
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from bop_toolkit_lib import inout

from scripts.val_score_csv import score_csv
from src.bop_dataset import (
    SPLITS,
    BOPDataset,
    default_bop_root,
    load_index,
    object_extents,
    rows_for_objects,
    select_rows,
)
from src.bop_evaluate import assign_detections, build_model, noisy_boxes_for_rows, pose_transformer
from src.engine import forward_dense
from src.models import VIT_MODELS
from src.pnp import Correspondences, dense_correspondences, estimate_pose
from src.utils import process_rotation, quaternion_to_rotation_matrix

OBJ_IDS = SPLITS["lmo"]["obj_ids"]
BATCH = 32  # as bop_evaluate: the PnP seed is the position in the batch
VAL_CSV, TEST_CSV = "fastposevitmo_lmo-val_real.csv", "fastposevitmo_lmo-test.csv"

# The ensembles of 2026-11-25 (TASKS.md §9); --spec FILE.json replaces them (same two keys).
MODELS = {
    "dx5a2": {"dir": "lmo-m18-densex5-001/densex5", "dense": True, "vit": "vit_b_16_384"},
    "dx5b2": {"dir": "lmo-m18-densex52-001/densex52", "dense": True, "vit": "vit_b_16_384"},
    "dx5a4": {"dir": "lmo-m18x4-densex5-001/densex5", "dense": True, "vit": "vit_b_16_384"},
    "dx5b4": {"dir": "lmo-m18x4-densex52-001/densex52", "dense": True, "vit": "vit_b_16_384"},
    "syma4": {"dir": "lmo-m18x4-densesymx5-001/densesymx5", "dense": True, "vit": "vit_b_16_384"},
    "symb4": {"dir": "lmo-m18x4-densesymx52-001/densesymx52", "dense": True, "vit": "vit_b_16_384"},
    "La4": {"dir": "lmo-m15x4-vitl512-001/vitl512", "dense": False, "vit": "vit_l_16_512"},
    "Lb4": {"dir": "lmo-m15x4-vitl5122-001/vitl5122", "dense": False, "vit": "vit_l_16_512"},
    "Ha2": {"dir": "lmo-m15-vith518-001/vith518", "dense": False, "vit": "vit_h_14_518"},
    "Hb2": {"dir": "lmo-m15-vith5182-001/vith5182", "dense": False, "vit": "vit_h_14_518"},
}
ENSEMBLES = {
    "check_pool_of_one_x2": {"mode": "pool", "members": ["dx5a2"]},
    "pool_seeds_x2": {"mode": "pool", "members": ["dx5a2", "dx5b2"]},
    "pool_seeds_x4": {"mode": "pool", "members": ["dx5a4", "dx5b4"]},
    "avg_B_H_x2_a": {"mode": "average", "members": ["dx5a2", "Ha2"]},
    "avg_B_H_x2_b": {"mode": "average", "members": ["dx5b2", "Hb2"]},
    "avg_B_L_x4_a": {"mode": "average", "members": ["dx5a4", "La4"]},
    "avg_B_L_x4_b": {"mode": "average", "members": ["dx5b4", "Lb4"]},
    "avg_cfg_x4_a": {"mode": "average", "members": ["dx5a4", "syma4"]},
    "avg_cfg_x4_b": {"mode": "average", "members": ["dx5b4", "symb4"]},
    "pool_cfg_x4_a": {"mode": "pool", "members": ["dx5a4", "syma4"]},
    "pool_cfg_x4_b": {"mode": "pool", "members": ["dx5b4", "symb4"]},
    "avg_all4_x4": {"mode": "average", "members": ["dx5a4", "dx5b4", "syma4", "symb4"]},
    "pool_all4_x4": {"mode": "pool", "members": ["dx5a4", "dx5b4", "syma4", "symb4"]},
}


def runs_dir() -> str:
    return os.path.join(os.environ["NAS_HOME"], "outputs", "FastPose-ViT", "runs")


def stored_csv(model: dict, split: str) -> str:
    """The result CSV a run stored for the noisy-box validation ("valn") or the test ("det")."""
    folder = f"eval_{split}_pnp" if model["dense"] else f"eval_{split}"
    name = VAL_CSV if split == "valn" else TEST_CSV
    path = os.path.join(runs_dir(), model["dir"], folder, name)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return path


def split_data(split: str):
    """(index with the crop boxes, per-row score) of the noisy validation or the detected test."""
    if split == "valn":
        index = load_index("lmo", "val_real")
        rows = rows_for_objects(index, OBJ_IDS)
        pred_index = dict(select_rows(index, rows))
        pred_index["bbox_visib"] = noisy_boxes_for_rows(index, rows, 0.1, 0)
        return pred_index, np.ones(len(rows))
    index = load_index("lmo", "test")
    split_index = select_rows(index, rows_for_objects(index, OBJ_IDS))
    det_path = glob.glob(os.path.join(
        os.environ["NAS_HOME"], "datasets/bop_detections/default_task1",
        "bop23_default_detections_for_task1/gdrnppdet-pbr/gdrnppdet-pbr_lmo-test_*.json"))[0]
    det = assign_detections(split_index, det_path)
    detected = ~np.isnan(det["score"])
    pred_index = select_rows(split_index, np.flatnonzero(detected))
    pred_index["bbox_visib"] = det["bbox"][detected]
    return pred_index, det["score"][detected]


def forward(spec: dict, split: str, device) -> dict:
    """Dense outputs and DR poses of one dense model on every row of a split."""
    args = argparse.Namespace(
        dataset="lmo", vit_model=spec["vit"], object_conditioning="token",
        model_weights=os.path.join(runs_dir(), spec["dir"], "last.pt"), dense_heads=True,
        dense_size=64, depth_mode="site", no_rotation_compensation=False,
    )
    model = build_model(args, device)
    pred_index, score = split_data(split)
    dataset = BOPDataset(
        dataset="lmo", split="val_real" if split == "valn" else "test", index=pred_index,
        img_size=VIT_MODELS[spec["vit"]][3][0], augment=False, depth_mode="site",
    )
    transformer = pose_transformer(dataset, args)
    out = {k: [] for k in ("dense", "R", "t", "bboxes", "K", "obj_idx")}
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH, num_workers=8, shuffle=False)
    for batch in loader:
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            pred_t, pred_R, dense = forward_dense(
                model, batch["images"].to(device), batch["obj_idx"].to(device), merge_outputs=True
            )
            pred_R = process_rotation(pred_R, "matrix")
        abs_t, abs_q, _, _ = transformer.transform_to_absolute(
            pred_t.float(), pred_R.float(), batch["translations"].to(device),
            batch["rotations"].to(device), batch["bboxes"], K=batch["K"],
            img_wh=batch["img_wh"], diameter=batch["diameter"],
        )
        out["dense"].append(dense.float().cpu().numpy())
        out["R"].append(quaternion_to_rotation_matrix(abs_q).cpu().numpy())
        out["t"].append(abs_t.cpu().numpy())
        for k in ("bboxes", "K", "obj_idx"):
            out[k].append(batch[k].numpy())
    result = {k: np.concatenate(v) for k, v in out.items()}
    result.update(index=pred_index, score=score)
    return result


def pooled_csv(members: list[dict], extents, path: str) -> int:
    """PnP on the pooled correspondences of the members' outputs; returns the DR fallbacks."""
    centre, half = extents
    first = members[0]
    index, rows, fallbacks = first["index"], [], 0
    for i in range(len(first["R"])):
        o = int(first["obj_idx"][i])
        corrs = [dense_correspondences(m["dense"][i], m["bboxes"][i], centre[o], half[o])
                 for m in members]
        corr = Correspondences(np.concatenate([c.pts2d for c in corrs]),
                               np.concatenate([c.pts3d for c in corrs]))
        R, t, info = estimate_pose("pnp", corr, first["K"][i].astype(np.float64),
                                   first["R"][i].astype(np.float64),
                                   first["t"][i].astype(np.float64), seed=i % BATCH)
        fallbacks += int(info["fallback"])
        rows.append({"scene_id": int(index["scene_id"][i]), "im_id": int(index["im_id"][i]),
                     "obj_id": int(index["obj_id"][i]), "score": float(first["score"][i]),
                     "R": R, "t": t, "time": -1})
    inout.save_bop_results(path, rows)
    return fallbacks


def averaged_csv(paths: list[str], out: str) -> None:
    subprocess.run([sys.executable, "scripts/pose_ensemble.py", "--dataset", "lmo", "--out", out,
                    *paths], check=True, stdout=subprocess.DEVNULL)


def official_ar(folder: str) -> dict:
    env = {**os.environ, "BOP_NUM_WORKERS": "5"}
    with open(os.path.join(folder, "bop_eval_test.log"), "w") as log:
        subprocess.run(["bash", "scripts/bop_eval.sh", folder, TEST_CSV], check=True,
                       stdout=log, stderr=subprocess.STDOUT, env=env)
    path = os.path.join(folder, "eval", TEST_CSV[:-4], "scores_bop19.json")
    with open(path) as f:
        d = json.load(f)
    return {"AR": d["bop19_average_recall"], "AR_VSD": d["bop19_average_recall_vsd"],
            "AR_MSSD": d["bop19_average_recall_mssd"], "AR_MSPD": d["bop19_average_recall_mspd"]}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spec", help="JSON file with \"models\" and \"ensembles\"")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    spec = {"models": MODELS, "ensembles": ENSEMBLES}
    if a.spec:
        with open(a.spec) as f:
            spec = json.load(f)
    todo = {n: e for n, e in spec["ensembles"].items()
            if not os.path.exists(os.path.join(a.out, n, "result.json"))}
    device = torch.device("cuda")
    extents = object_extents(default_bop_root(), "lmo", OBJ_IDS)
    cache: dict = {}
    results = {}
    for name, ens in todo.items():  # first all CSVs and validation scores
        folders = {s: os.path.join(a.out, name, f"eval_{s}") for s in ("valn", "det")}
        fallbacks = {}
        for split, folder in folders.items():
            os.makedirs(folder, exist_ok=True)
            out = os.path.join(folder, VAL_CSV if split == "valn" else TEST_CSV)
            if ens["mode"] == "pool":
                for m in ens["members"]:
                    if (m, split) not in cache:
                        cache[m, split] = forward(spec["models"][m], split, device)
                fallbacks[split] = pooled_csv(
                    [cache[m, split] for m in ens["members"]], extents, out)
            elif ens["mode"] == "average":
                averaged_csv([stored_csv(spec["models"][m], split) for m in ens["members"]], out)
            else:
                raise ValueError(f"unknown mode {ens['mode']}")
        val = score_csv(os.path.join(folders["valn"], VAL_CSV), "lmo", "val_real")
        results[name] = {**ens, "val": val, "fallbacks": fallbacks, "det_folder": folders["det"]}
        print(f"{name}: validation {val['score']:.4f}", flush=True)

    def evaluate(name):
        r = results[name]
        r["test"] = official_ar(r.pop("det_folder"))
        with open(os.path.join(a.out, name, "result.json"), "w") as f:
            json.dump(r, f, indent=2)
        print(f"{name}: validation {r['val']['score']:.4f} test AR {r['test']['AR']:.4f}", flush=True)

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(evaluate, results))
    print("\n| ensemble | mode | members | val | test AR |\n|---|---|---|---|---|")
    for name in spec["ensembles"]:
        with open(os.path.join(a.out, name, "result.json")) as f:
            r = json.load(f)
        print(f"| {name} | {r['mode']} | {' + '.join(r['members'])} | {r['val']['score']:.4f} "
              f"| {r['test']['AR']:.4f} |")


if __name__ == "__main__":
    main()
