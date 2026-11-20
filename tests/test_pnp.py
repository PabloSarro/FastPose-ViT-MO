"""PnP solvers (M18.4): exact, noisy and outlier-contaminated correspondences, the fallback, and
the whole chain on the dataset's own dense targets (GT in, GT pose out)."""

import os

import numpy as np
import pytest
import torch

from src.bop_dataset import (
    LMO_OBJ_IDS,
    SPLITS,
    BOPDataset,
    Source,
    build_index,
    default_bop_root,
    nearest_rotation,
)
from src.pnp import (
    POSE_MODES,
    Correspondences,
    dense_correspondences,
    estimate_pose,
    reprojection_errors,
)

K = np.array([[572.4, 0.0, 325.3], [0.0, 573.6, 242.0], [0.0, 0.0, 1.0]])


def random_rotation(rng, max_deg=180.0):
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = np.deg2rad(rng.uniform(0, max_deg))
    A = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * A + (1 - np.cos(angle)) * A @ A


def rot_err_deg(R1, R2):
    return np.degrees(np.arccos(np.clip((np.trace(R1.T @ R2) - 1) / 2, -1, 1)))


def scene(rng, n=600, noise=0.0, outliers=0.0):
    pts3d = rng.uniform(-40, 40, size=(n, 3))
    R = random_rotation(rng)
    t = np.array([20.0, -30.0, 700.0]) + rng.uniform(-50, 50, 3)
    cam = pts3d @ R.T + t
    pts2d = cam[:, :2] / cam[:, 2:3] * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
    pts2d = pts2d + rng.normal(0, noise, pts2d.shape)
    bad = rng.random(n) < outliers
    pts3d = pts3d.copy()
    pts3d[bad] = rng.uniform(-40, 40, size=(bad.sum(), 3))  # wrong 3D point for the pixel
    return Correspondences(pts2d, pts3d), R, t


@pytest.mark.parametrize("mode", ["pnp", "pnp_init", "dr_rot_pnp_t"])
def test_exact_correspondences_give_the_exact_pose(mode):
    rng = np.random.default_rng(0)
    for _ in range(5):
        corr, R, t = scene(rng)
        # a starting pose near the truth for the modes that use the DR pose
        R0 = random_rotation(rng, 3.0) @ R if mode != "dr_rot_pnp_t" else R
        t0 = t + rng.uniform(-15, 15, 3)
        R_est, t_est, info = estimate_pose(mode, corr, K, R0, t0)
        assert not info["fallback"]
        assert rot_err_deg(R_est, R) < 0.02 and np.linalg.norm(t_est - t) < 0.2


@pytest.mark.parametrize("mode", ["pnp", "pnp_init", "dr_rot_pnp_t"])
def test_noise_and_outliers(mode):
    rng = np.random.default_rng(1)
    rot, trans = [], []
    for _ in range(10):
        corr, R, t = scene(rng, noise=0.5, outliers=0.3)
        R0 = random_rotation(rng, 3.0) @ R if mode != "dr_rot_pnp_t" else R
        R_est, t_est, info = estimate_pose(mode, corr, K, R0, t + rng.uniform(-15, 15, 3))
        assert not info["fallback"]
        rot.append(rot_err_deg(R_est, R))
        trans.append(np.linalg.norm(t_est - t))
    assert np.max(rot) < 1.0 and np.max(trans) < 3.0, (max(rot), max(trans))


def test_dr_mode_and_failures_return_the_dr_pose():
    rng = np.random.default_rng(2)
    corr, R, t = scene(rng)
    R0, t0 = random_rotation(rng), t + 10
    R_est, t_est, info = estimate_pose("dr", corr, K, R0, t0)
    assert np.array_equal(R_est, R0) and np.array_equal(t_est, t0) and not info["fallback"]
    few = Correspondences(corr.pts2d[:4], corr.pts3d[:4])
    for mode in ("pnp", "pnp_init", "dr_rot_pnp_t"):
        R_est, t_est, info = estimate_pose(mode, few, K, R0, t0)
        assert info["fallback"] and np.array_equal(R_est, R0) and np.array_equal(t_est, t0)
    with pytest.raises(ValueError, match="pose mode"):
        estimate_pose("bad", corr, K, R0, t0)


def test_pnp_init_depends_on_the_dr_pose_but_pnp_does_not():
    """pnp ignores the DR pose; pnp_init converges from a moderately wrong one (a few degrees,
    centimetres) and, from a pose behind the camera, falls back instead of returning nonsense."""
    rng = np.random.default_rng(3)
    corr, R, t = scene(rng)
    R_est, t_est, info = estimate_pose("pnp", corr, K, np.eye(3), np.zeros(3))
    assert not info["fallback"] and rot_err_deg(R_est, R) < 0.02
    R_est, t_est, info = estimate_pose("pnp_init", corr, K, random_rotation(rng, 10.0) @ R, t + 40)
    assert not info["fallback"] and rot_err_deg(R_est, R) < 0.05
    _, _, info = estimate_pose("pnp_init", corr, K, R, np.array([0.0, 0.0, -500.0]))
    assert info["fallback"]


def test_modes_listed():
    assert POSE_MODES == ("dr", "pnp", "pnp_init", "dr_rot_pnp_t")


BOP_ROOT = default_bop_root()
HAS_BOP = os.path.exists(os.path.join(BOP_ROOT, "lmo", "test_targets_bop19.json"))


@pytest.mark.skipif(not HAS_BOP, reason="BOP tree not available")
@pytest.mark.parametrize("which", ["pbr", "real"])
def test_gt_dense_targets_give_the_gt_pose(which):
    """The chain dataset targets -> correspondences -> PnP returns the ground-truth pose: it checks
    the 2D coordinates of the pixels, the normalisation and the solvers against the dataset."""
    if which == "pbr":
        index = build_index(BOP_ROOT, (Source("lm", "train_pbr", (0, 17)),), LMO_OBJ_IDS, "lmo")
    else:
        index = build_index(BOP_ROOT, SPLITS["lmo"]["test"], LMO_OBJ_IDS, "lmo")
    ds = BOPDataset(dataset="lmo", split="val", index=index, img_size=64, dense_size=64, augment=False)
    errors = {mode: [] for mode in POSE_MODES[1:]}
    for row in range(0, len(ds), len(ds) // 30):
        s = ds[row]
        obj = s["obj_idx"]
        # a "prediction" made of the targets: confident mask logits and the exact XYZ
        dense = torch.cat([(s["mask"] * 2 - 1) * 10, s["xyz"]]).numpy()
        corr = dense_correspondences(
            dense, s["bboxes"].double().tolist(), ds.xyz_centre[obj], ds.xyz_half[obj]
        )
        R_gt = nearest_rotation(s["R_m2c"].double().numpy())
        t_gt = s["t_m2c"].double().numpy()
        # the texture-less valid pixels also carry targets: only valid ones, as in training
        keep = s["xyz_valid"][0].numpy().ravel()[np.flatnonzero(dense[0].ravel() > 0)] > 0
        corr = Correspondences(corr.pts2d[keep], corr.pts3d[keep])
        for mode in POSE_MODES[1:]:
            R0 = random_rotation(np.random.default_rng(row), 2.0) @ R_gt
            R_est, t_est, info = estimate_pose(mode, corr, K=s["K"].double().numpy(), R_dr=R0, t_dr=t_gt + 8)
            assert not info["fallback"], (mode, row)
            errors[mode].append((rot_err_deg(R_est, R_gt), np.linalg.norm(t_est - t_gt)))
    for mode, e in errors.items():
        e = np.array(e)
        if mode == "dr_rot_pnp_t":
            # keeps the DR rotation (2 deg off here): its translation compensates, median ~1 mm,
            # worst case ~15 mm for a bad viewing direction
            assert np.median(e[:, 0]) < 2.0 and np.median(e[:, 1]) < 5.0, (mode, e.mean(0))
        else:
            # the targets are the GT, so PnP must return it (measured 0.000 deg, 0.00 mm)
            assert e[:, 0].max() < 0.05 and e[:, 1].max() < 0.5, (which, mode, e.max(0))
