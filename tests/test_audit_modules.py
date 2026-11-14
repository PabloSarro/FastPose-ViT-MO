# Audit W6 (2026-11-14): tests of the scripts / helpers added after the first 150 tests: the RGB-D
# depth correction, the pose ensemble, and the multi-class detector's training helpers.
import math

import numpy as np
import pytest
import torch

from object_detector.bop_detector import EMA, image_groups, lr_factor, to_float
from scripts.depth_correction import SplatCorrector
from scripts.pose_ensemble import angle, combine, project_so3


def rot_z(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])


# ------------------------------------------------------------------ depth correction (RGB-D)
K = np.array([[600.0, 0, 320.0], [0, 600.0, 240.0], [0, 0, 1.0]])


def sphere_points(n=20000, radius=50.0, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    return (v / np.linalg.norm(v, axis=1, keepdims=True) * radius).astype(np.float32)


def depth_image(points, t, shape=(480, 640)):
    """Measured depth: z-buffer of the model points at pose (I, t), nearest point per pixel."""
    X = points + t
    u = np.floor(K[0, 0] * X[:, 0] / X[:, 2] + K[0, 2]).astype(int)
    v = np.floor(K[1, 1] * X[:, 1] / X[:, 2] + K[1, 2]).astype(int)
    depth = np.full(shape, np.inf, dtype=np.float32)
    for ui, vi, z in zip(u, v, X[:, 2]):
        if 0 <= ui < shape[1] and 0 <= vi < shape[0]:
            depth[vi, ui] = min(depth[vi, ui], z)
    depth[np.isinf(depth)] = 0.0
    return depth


@pytest.fixture(scope="module")
def corrector():
    c = SplatCorrector.__new__(SplatCorrector)  # without the BOP models: synthetic geometry
    c.device, c.cell, c.vis_eps = torch.device("cpu"), 4, 0.05
    c.points = {1: torch.as_tensor(sphere_points())}
    c.diameter = {1: 100.0}
    return c


def run(corrector, t_true, t_pred, depth, bands=(0.2, 0.1, 0.05), min_points=50):
    t, count, shift = corrector([1], [np.eye(3)], [np.asarray(t_pred, float).reshape(3, 1)],
                                torch.as_tensor(depth), K, list(bands), min_points)
    return t[0], int(count[0]), float(shift[0])


def test_depth_correction_recovers_the_distance(corrector):
    t_true = np.array([20.0, -10.0, 800.0])
    depth = depth_image(sphere_points(), t_true)
    t_pred = t_true * 1.06  # 6 % too far along the same ray
    t, count, shift = run(corrector, t_true, t_pred, depth)
    assert count > 50
    assert abs(t[2] - t_true[2]) < 3.0  # mm, at 800 mm
    # the correction moves the object along its viewing ray: x / z and y / z are unchanged
    assert np.allclose(t[:2] / t[2], t_pred[:2] / t_pred[2], atol=1e-6)
    assert shift < 0  # it moved the object closer


def test_depth_correction_ignores_occluders_and_missing_depth(corrector):
    t_true = np.array([0.0, 0.0, 800.0])
    depth = depth_image(sphere_points(), t_true)
    occluded = depth.copy()
    occluded[:, :330] = np.where(occluded[:, :330] > 0, 400.0, 0.0)  # a plane 400 mm in front
    occluded[:, :330][depth[:, :330] == 0] = 400.0
    t, count, _ = run(corrector, t_true, t_true * 1.05, occluded)
    assert abs(t[2] - 800.0) < 6.0  # the occluder (400 mm away, outside the band) is dropped
    # no valid depth: the RGB estimate is kept
    t0, n0, shift0 = run(corrector, t_true, t_true * 1.05, np.zeros_like(depth))
    assert n0 < 50 and shift0 == 0.0 and np.allclose(t0, t_true * 1.05)


def test_depth_correction_is_batched_over_estimates(corrector):
    t_a, t_b = np.array([0.0, 0.0, 700.0]), np.array([60.0, 30.0, 900.0])
    depth = np.maximum(depth_image(sphere_points(), t_a), 0)
    depth_b = depth_image(sphere_points(seed=1), t_b)
    depth = np.where(depth_b > 0, depth_b, depth)
    t, count, _ = corrector([1, 1], [np.eye(3), np.eye(3)],
                            [(t_a * 1.04).reshape(3, 1), (t_b * 0.96).reshape(3, 1)],
                            torch.as_tensor(depth), K, [0.2, 0.1, 0.05], 50)
    assert abs(t[0][2] - 700.0) < 4.0 and abs(t[1][2] - 900.0) < 5.0


# ------------------------------------------------------------------------------- pose ensemble
def test_project_so3_gives_a_rotation():
    rng = np.random.default_rng(0)
    M = rot_z(30) + 0.05 * rng.normal(size=(3, 3))
    R = project_so3(M)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12) and np.isclose(np.linalg.det(R), 1.0)
    assert np.allclose(project_so3(rot_z(30)), rot_z(30), atol=1e-12)


def test_combine_identical_estimates_and_means():
    R, t = rot_z(20), np.array([1.0, 2.0, 300.0])
    ident = [{"R": np.eye(3), "t": np.zeros((3, 1))}]
    R2, t2, score, time = combine([(0.8, R, t, 0.01), (0.6, R, t, 0.02)], ident)
    assert np.allclose(R2, R) and np.allclose(t2, t) and np.isclose(score, 0.7)
    assert np.isclose(time, 0.03)
    # two different estimates: the translation mean, the rotation halfway (here about z)
    R3, t3, _, _ = combine([(1, rot_z(10), np.zeros(3), 0), (1, rot_z(30), np.ones(3) * 2, 0)], ident)
    assert np.allclose(t3, 1.0) and angle(R3, rot_z(20)) < 1e-9


def test_combine_aligns_symmetric_poses_with_offsets():
    # An object symmetric by a 180 degree flip about z whose axis is shifted by (10, 0, 0) mm:
    # (R, t) and (R·S, t + R·s) are the same pose. The second model predicted the other one.
    S = rot_z(180)
    s = np.array([[10.0], [0.0], [0.0]])
    syms = [{"R": np.eye(3), "t": np.zeros((3, 1))}, {"R": S, "t": s}]
    R0, t0 = rot_z(25), np.array([5.0, 0.0, 500.0])
    R1, t1 = R0 @ S, t0 + R0 @ s.reshape(3)  # the symmetric twin of the reference
    # the twin of the twin is the reference again: build the estimate that needs aligning
    R_b = R0 @ np.linalg.inv(S)
    t_b = t0 - R_b @ s.reshape(3)
    R, t, _, _ = combine([(1, R0, t0, 0), (1, R_b, t_b, 0)], syms)
    assert angle(R, R0) < 1e-9 and np.allclose(t, t0, atol=1e-9)
    # without symmetry handling, averaging the twins would collapse the rotation
    assert angle(R0, R_b) > 3.0 and R1.shape == (3, 3) and t1.shape == (3,)


# ------------------------------------------------------------------------- detector helpers
def test_lr_factor_warmup_and_cosine():
    total, warm = 1000, 100
    assert lr_factor(0, total, warm) == pytest.approx(1 / warm)
    assert lr_factor(warm - 1, total, warm) == pytest.approx(1.0)
    assert lr_factor(warm, total, warm) == pytest.approx(1.0)  # cosine starts at 1
    assert lr_factor(total, total, warm) == pytest.approx(0.01)  # ends at 1 % of the peak
    mid = lr_factor(warm + (total - warm) // 2, total, warm)
    assert 0.4 < mid < 0.6


def test_ema_follows_the_model_with_warmed_up_decay():
    model = torch.nn.Linear(4, 2)
    ema = EMA(model, decay=0.9, tau=1)
    w0 = ema.model.weight.clone()
    with torch.no_grad():
        model.weight.add_(1.0)
    ema.update(model)
    d = 0.9 * (1 - math.exp(-1 / 1))
    assert torch.allclose(ema.model.weight, d * w0 + (1 - d) * model.weight, atol=1e-6)
    for _ in range(300):
        ema.update(model)
    assert torch.allclose(ema.model.weight, model.weight, atol=1e-3)
    assert not any(p.requires_grad for p in ema.model.parameters())


def test_to_float_converts_nested_bf16_only():
    x = {"a": torch.ones(2, dtype=torch.bfloat16), "b": [torch.arange(3), (torch.ones(1).half(),)],
         "c": 5}
    y = to_float(x)
    assert y["a"].dtype == torch.float32 and y["b"][0].dtype == torch.int64
    assert y["b"][1][0].dtype == torch.float32 and y["c"] == 5


def test_image_groups_in_first_seen_order():
    index = {"scene_id": np.array([2, 2, 5, 2, 5]), "im_id": np.array([1, 1, 7, 1, 8])}
    assert image_groups(index) == [[0, 1, 3], [2], [4]]


# ------------------------------------------------------------------------------ safe loading
def test_load_safe_reads_checkpoints_and_refuses_code(tmp_path):
    import argparse
    import os
    import pickle

    from src.safe_load import load_safe

    ckpt = {"model": {"w": torch.ones(2, 2)}, "epoch": 3, "args": argparse.Namespace(lr=1e-4),
            "rng": np.random.get_state()}
    path = str(tmp_path / "ok.pt")
    torch.save(ckpt, path)
    out = load_safe(path)
    assert out["epoch"] == 3 and out["args"].lr == 1e-4 and torch.equal(out["model"]["w"], ckpt["model"]["w"])

    marker = tmp_path / "pwned"

    class Evil:
        def __reduce__(self):
            return (os.system, (f"touch {marker}",))

    bad = str(tmp_path / "bad.pt")
    with open(bad, "wb") as f:
        pickle.dump({"model": Evil()}, f)
    torch.save({"model": Evil()}, bad)
    with pytest.raises(Exception):
        load_safe(bad)
    assert not marker.exists()  # the pickled code did not run
