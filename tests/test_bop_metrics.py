# Tests of src/bop_metrics.py (M5): the GPU validation errors equal bop_toolkit's pose_error
# (ADD / ADD-S, MSSD, MSPD with symmetries) and the recalls equal src/bop_evaluate.py's.
# Need the BOP tree (skipped without it).

import os

import numpy as np
import pytest
import torch

from src.bop_dataset import LMO_OBJ_IDS, default_bop_root, load_index, rows_for_objects, select_rows

pytest.importorskip("bop_toolkit_lib")
from bop_toolkit_lib import inout, misc, pose_error  # noqa: E402

from src.bop_evaluate import instance_errors, summarize  # noqa: E402
from src.bop_metrics import BOPPoseMetrics  # noqa: E402

BOP_ROOT = default_bop_root()
pytestmark = pytest.mark.skipif(
    not os.path.isdir(os.path.join(BOP_ROOT, "lmo")), reason="BOP tree not available"
)


@pytest.fixture(scope="module")
def metrics():
    return BOPPoseMetrics(BOP_ROOT, "lmo", LMO_OBJ_IDS, torch.device("cpu"))


@pytest.fixture(scope="module")
def perturbed():
    """LM-O test instances of driller, eggbox and glue with noisy 'predicted' poses."""
    index = load_index("lmo", "test")
    sub = select_rows(index, np.concatenate([rows_for_objects(index, [o])[:6] for o in (8, 10, 11)]))
    rng = np.random.default_rng(0)
    R_gt, t_gt = sub["R"].astype(np.float64), sub["t"].astype(np.float64)
    # Snap GT to SO(3) as in the M2 round-trip test, then rotate by up to ~10 deg, shift ~15 mm
    u, _, vt = np.linalg.svd(R_gt)
    R_gt = u @ vt
    w = rng.normal(scale=0.1, size=(len(R_gt), 3))
    dR = np.stack([cv_rodrigues(v) for v in w])
    R_est = dR @ R_gt
    t_est = t_gt + rng.normal(scale=15.0, size=t_gt.shape)
    # Two eggbox/glue predictions near the flipped (symmetric) pose: ADD-S/MSSD must stay small
    info = inout.load_json(os.path.join(BOP_ROOT, "lmo", "models_eval", "models_info.json"), keys_to_int=True)
    for i in np.flatnonzero(np.isin(sub["obj_id"], [10, 11]))[:2]:
        S = misc.get_symmetry_transformations(info[int(sub["obj_id"][i])], 0.01)[1]
        R_est[i] = R_gt[i] @ S["R"]
        t_est[i] = (R_gt[i] @ S["t"])[:, 0] + t_gt[i]
    return sub, R_est, t_est, R_gt, t_gt


def cv_rodrigues(v):
    th = np.linalg.norm(v)
    k = v / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx


def test_errors_match_bop_toolkit(metrics, perturbed):
    sub, R_est, t_est, R_gt, t_gt = perturbed
    obj_idx = torch.as_tensor(sub["obj_idx"])
    T = lambda x: torch.as_tensor(x, dtype=torch.float32)  # noqa: E731
    got = metrics.errors(T(R_est), T(t_est), T(R_gt), T(t_gt), T(sub["K"]), obj_idx)
    info = inout.load_json(os.path.join(BOP_ROOT, "lmo", "models_eval", "models_info.json"), keys_to_int=True)
    for i in range(len(R_est)):
        o = int(sub["obj_id"][i])
        pts = inout.load_ply(os.path.join(BOP_ROOT, "lmo", "models_eval", f"obj_{o:06d}.ply"))["pts"]
        syms = misc.get_symmetry_transformations(info[o], 0.01)
        a = (R_est[i], t_est[i][:, None], R_gt[i], t_gt[i][:, None])
        add = (pose_error.adi if o in (10, 11) else pose_error.add)(*a, pts)
        mssd = pose_error.mssd(*a, pts, syms)
        mspd = pose_error.mspd(a[0], a[1], a[2], a[3], sub["K"][i].astype(np.float64), pts, syms)
        # fp32 on the GPU path vs fp64 in bop_toolkit: ~1e-3 mm / px at ~1 m
        assert got["add"][i].item() == pytest.approx(add, abs=0.02, rel=1e-4), (o, "add")
        assert got["mssd"][i].item() == pytest.approx(mssd, abs=0.02, rel=1e-4), (o, "mssd")
        assert got["mspd"][i].item() == pytest.approx(mspd, abs=0.02, rel=1e-4), (o, "mspd")
    # The flipped eggbox/glue predictions are (near) perfect under the symmetric metrics
    flipped = np.flatnonzero(np.isin(sub["obj_id"], [10, 11]))[:2]
    assert got["mssd"][flipped].max() < 0.05 and got["mspd"][flipped].max() < 0.05


def test_recalls_match_bop_evaluate(metrics, perturbed):
    sub, R_est, t_est, R_gt, t_gt = perturbed
    T = lambda x: torch.as_tensor(x, dtype=torch.float32)  # noqa: E731
    metrics.reset()
    for o in (8, 10, 11):  # two batches of mixed size, as in an epoch
        m = sub["obj_id"] == o
        metrics.update(T(R_est[m]), T(t_est[m]), T(R_gt[m]), T(t_gt[m]), T(sub["K"][m]),
                       torch.as_tensor(sub["obj_idx"][m]), torch.as_tensor(sub["img_wh"][m, 0]))
    got = metrics.compute()
    assert got["n"] == len(R_est) and set(got["per_object"]) == {8, 10, 11}
    for o in (8, 10, 11):
        rows = np.flatnonzero(sub["obj_id"] == o)
        o_sub = select_rows(sub, rows)
        o_sub["R"] = R_gt[rows]
        ref = summarize(instance_errors(o_sub, {"R": R_est[rows], "t": t_est[rows]}, o, BOP_ROOT))
        assert got["per_object"][o][0] == pytest.approx(ref["recall_MSSD"], abs=1e-6)
        assert got["per_object"][o][1] == pytest.approx(ref["recall_MSPD"], abs=1e-6)
    assert got["score"] == pytest.approx((got["AR_MSSD"] + got["AR_MSPD"]) / 2)
