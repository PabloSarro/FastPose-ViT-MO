"""SymmetricRotation6DLoss (M5): min of the relative 6D Frobenius loss over each object's symmetries.

- Loss = 0 for every R_gt·S_k (any symmetric pose is a perfect prediction).
- Equal to RelativeRotation6DFrobeniusLoss for objects with the identity only.
- Finite gradients, also at a tie between two symmetric targets.
- pose_loss (M8, symmetries with a translation offset, e.g. the YCB-V clamps): its labels of every
  symmetric pose (R·S_k, t + R·s_k) equal what BOPDataset computes for that pose, a prediction of
  any of them gives zero loss and that pose's translation label, and without offsets it equals
  the rotation-only loss.
"""

import os

import numpy as np
import pytest
import torch

from src.bop_dataset import (
    LMO_OBJ_IDS,
    YCBV_OBJ_IDS,
    BOPDataset,
    default_bop_root,
    load_index,
    select_rows,
    symmetry_table,
)
from src.losses import RelativeRotation6DFrobeniusLoss, SymmetricRotation6DLoss
from src.utils import rotation_matrix_to_6d


def random_rotations(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(n, 3, 3, generator=g, dtype=torch.float64))
    q = q * torch.sign(torch.diagonal(r, dim1=-2, dim2=-1)).unsqueeze(-2)
    q[torch.det(q) < 0, :, 0] *= -1  # proper rotations
    return q.float()


def rz(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


# Object 0: asymmetric; 1: 180 deg about z (padded with the identity); 2: 4-fold about z
TABLE = torch.tensor(
    np.stack(
        [
            np.stack([np.eye(3)] * 4),
            np.stack([np.eye(3), rz(180), np.eye(3), np.eye(3)]),
            np.stack([rz(0), rz(90), rz(180), rz(270)]),
        ]
    ),
    dtype=torch.float32,
)


@pytest.mark.parametrize("obj", [1, 2])
def test_zero_for_every_symmetric_pose(obj):
    loss = SymmetricRotation6DLoss(TABLE)
    R = random_rotations(16)
    for k in range(TABLE.shape[1]):
        pred = rotation_matrix_to_6d(R @ TABLE[obj, k])
        assert loss(pred, R, torch.full((16,), obj)).item() < 1e-6


def test_equals_plain_loss_for_asymmetric_objects():
    loss, plain = SymmetricRotation6DLoss(TABLE), RelativeRotation6DFrobeniusLoss()
    R, pred = random_rotations(32, seed=1), torch.randn(32, 6)
    torch.testing.assert_close(
        loss(pred, R, torch.zeros(32, dtype=torch.long)), plain(pred, rotation_matrix_to_6d(R))
    )


def test_symmetric_loss_never_exceeds_plain_loss():
    loss, plain = SymmetricRotation6DLoss(TABLE), RelativeRotation6DFrobeniusLoss()
    R, pred = random_rotations(1), torch.randn(1, 6)
    for obj in (1, 2):
        assert loss(pred, R, torch.tensor([obj])) <= plain(pred, rotation_matrix_to_6d(R)) + 1e-6


def test_mixed_batch_picks_each_samples_own_object():
    """A sample flipped by 180 deg about z is perfect for object 1 but not for object 0."""
    loss = SymmetricRotation6DLoss(TABLE)
    R = random_rotations(2, seed=2)
    pred = rotation_matrix_to_6d(R @ TABLE[1, 1])
    per_obj0 = loss(pred[:1], R[:1], torch.tensor([0]))
    assert per_obj0 > 0.5
    torch.testing.assert_close(loss(pred, R, torch.tensor([0, 1])), (per_obj0 + 0) / 2, atol=1e-6, rtol=0)


def test_finite_gradients_including_ties():
    loss = SymmetricRotation6DLoss(TABLE)
    R = random_rotations(8, seed=3)
    pred = torch.randn(8, 6, requires_grad=True)
    loss(pred, R, torch.tensor([0, 1, 2, 0, 1, 2, 1, 2])).backward()
    assert torch.isfinite(pred.grad).all() and pred.grad.abs().sum() > 0
    # Exactly halfway between R and R·Rz(180): both symmetric targets at the same distance
    tie = rotation_matrix_to_6d(R[:1] @ torch.tensor(rz(90), dtype=torch.float32)).requires_grad_()
    loss(tie, R[:1], torch.tensor([1])).backward()
    assert torch.isfinite(tie.grad).all()


def test_allocentric_targets():
    """Symmetry on the right commutes with the allocentric change on the left: R_rel^T (R S) = (R_rel^T R) S."""
    loss = SymmetricRotation6DLoss(TABLE)
    R, R_rel = random_rotations(4, seed=4), random_rotations(4, seed=5)
    allo = R_rel.transpose(1, 2) @ R
    pred = rotation_matrix_to_6d(R_rel.transpose(1, 2) @ (R @ TABLE[2, 3]))
    assert loss(pred, allo, torch.full((4,), 2)).item() < 1e-6


def test_fp16_inputs_under_autocast_are_computed_in_fp32():
    loss = SymmetricRotation6DLoss(TABLE)
    R = random_rotations(4)
    pred = rotation_matrix_to_6d(R @ TABLE[2, 1]).half()
    assert loss(pred, R.half(), torch.full((4,), 2)).dtype == torch.float32


@pytest.mark.skipif(
    not os.path.isdir(os.path.join(default_bop_root(), "lmo")), reason="BOP tree not available"
)
def test_lmo_symmetry_table():
    table, offsets = symmetry_table(default_bop_root(), "lmo", LMO_OBJ_IDS)
    assert table.shape == (8, 2, 3, 3) and offsets.shape == (8, 2, 3)
    assert np.linalg.norm(offsets, axis=-1).max() < 1  # mm
    np.testing.assert_allclose(table[:, 0], np.tile(np.eye(3), (8, 1, 1)), atol=1e-6)
    flipped = [o for i, o in enumerate(LMO_OBJ_IDS) if not np.allclose(table[i, 1], np.eye(3))]
    assert flipped == [10, 11]  # eggbox and glue
    # A ~180 deg flip, as measured on the scanned models: eggbox 179.8 deg, glue 178.5 deg
    for i in (LMO_OBJ_IDS.index(10), LMO_OBJ_IDS.index(11)):
        S = table[i, 1]
        assert np.degrees(np.arccos(np.clip((np.trace(S) - 1) / 2, -1, 1))) > 178


def test_pose_loss_without_offsets_equals_rotation_loss():
    loss = SymmetricRotation6DLoss(TABLE)
    assert not loss.has_offsets
    n = 6
    obj = torch.tensor([0, 1, 2, 0, 1, 2])
    pose = (
        random_rotations(n, seed=6),  # R_m2c
        torch.tensor([[30.0, -20.0, 800.0]]).repeat(n, 1),  # t_m2c, mm
        torch.tensor([[1000.0, 0, 320], [0, 1000.0, 240], [0, 0, 1]]).repeat(n, 1, 1),  # K
        torch.tensor([[300.0, 200.0, 400.0, 300.0]]).repeat(n, 1),  # crop bbox
        torch.tensor([[640.0, 480.0]]).repeat(n, 1),  # img_wh
        torch.full((n,), 100.0),  # diameter, mm
    )
    U, rotations = loss.symmetric_labels(obj, *pose)
    pred = torch.randn(n, 6)
    value, U_best = loss.pose_loss(pred, obj, *pose)
    torch.testing.assert_close(value, loss(pred, rotations[:, 0].float(), obj))
    for k in range(TABLE.shape[1]):  # no offset: every symmetric pose has the same U
        torch.testing.assert_close(U[:, k], U[:, 0])
    torch.testing.assert_close(U_best, U[:, 0].float())


YCBV_ROOT = os.path.join(default_bop_root(), "ycbv", "models_eval")


@pytest.fixture(scope="module")
def clamp_samples():
    """YCB-V test crops of the clamps (19, 20) and the bowl (13, continuous), with bbox jitter."""
    index = load_index("ycbv", "test")
    rows = np.flatnonzero(np.isin(index["obj_id"], [13, 19, 20]))
    rows = np.random.default_rng(0).choice(rows, 24, replace=False)
    index = select_rows(index, rows)
    np.random.seed(0)
    ds = BOPDataset(dataset="ycbv", index=index, img_size=32, augment=True, pixel_aug="none",
                    bbox_jitter_prob=1.0)
    batch = [ds[i] for i in range(len(ds))]
    batch = {k: torch.stack([torch.as_tensor(b[k]) for b in batch]) for k in batch[0]}
    table, offsets = symmetry_table(default_bop_root(), "ycbv", YCBV_OBJ_IDS)
    loss = SymmetricRotation6DLoss(torch.from_numpy(table), torch.from_numpy(offsets))
    return ds, batch, loss


def pose_args(batch):
    return [batch[k] for k in ("obj_idx", "R_m2c", "t_m2c", "K", "bboxes", "img_wh", "diameter")]


@pytest.mark.skipif(not os.path.isdir(YCBV_ROOT), reason="YCB-V tree not available")
def test_ycbv_clamp_offsets(clamp_samples):
    _, _, loss = clamp_samples
    assert loss.has_offsets
    norms = loss.sym_offsets.norm(dim=-1).amax(dim=1)
    assert norms[YCBV_OBJ_IDS.index(19)] > 19 and norms[YCBV_OBJ_IDS.index(20)] > 19
    assert norms[YCBV_OBJ_IDS.index(13)] == 0


@pytest.mark.skipif(not os.path.isdir(YCBV_ROOT), reason="YCB-V tree not available")
def test_identity_symmetry_labels_equal_the_dataset_labels(clamp_samples):
    _, batch, loss = clamp_samples
    U, rotations = loss.symmetric_labels(*pose_args(batch))
    torch.testing.assert_close(U[:, 0].float(), batch["translations"], atol=1e-5, rtol=0)
    torch.testing.assert_close(rotations[:, 0].float(), batch["rotations"], atol=1e-5, rtol=0)


@pytest.mark.skipif(not os.path.isdir(YCBV_ROOT), reason="YCB-V tree not available")
def test_symmetric_pose_labels_equal_the_dataset_labels_of_that_pose(clamp_samples):
    """For k = 1 (the clamps' 180 deg flip + ~19 mm shift), the labels BOPDataset gives to the
    pose (R·S_1, t + R·s_1), with the same crop bbox."""
    ds, batch, loss = clamp_samples
    U, rotations = loss.symmetric_labels(*pose_args(batch))
    k = 1
    for i in range(len(ds)):
        o = int(batch["obj_idx"][i])
        R = ds.index["R"][i].astype(np.float64)
        S, s = loss.sym_table[o, k].double().numpy(), loss.sym_offsets[o, k].double().numpy()
        ds.index["R"][i], ds.index["t"][i] = R @ S, ds.index["t"][i] + R @ s
        U_ds, rot_ds = ds.labels(i, batch["bboxes"][i].double().tolist())
        ds.index["R"][i], ds.index["t"][i] = R, ds.index["t"][i] - R @ s
        torch.testing.assert_close(U[i, k], U_ds, atol=1e-6, rtol=0)
        torch.testing.assert_close(rotations[i, k], rot_ds, atol=1e-6, rtol=0)
    clamps = torch.isin(batch["obj_id"], torch.tensor([19, 20]))
    assert clamps.any() and (U[clamps, 1] - U[clamps, 0]).abs().amax() > 1e-3  # the shift matters


@pytest.mark.skipif(not os.path.isdir(YCBV_ROOT), reason="YCB-V tree not available")
@pytest.mark.parametrize("k", [0, 1])
def test_predicting_a_symmetric_pose_gives_zero_loss_and_its_translation(clamp_samples, k):
    _, batch, loss = clamp_samples
    U, rotations = loss.symmetric_labels(*pose_args(batch))
    pred = rotation_matrix_to_6d(rotations[:, k].float())
    value, U_best = loss.pose_loss(pred, *pose_args(batch))
    assert value.item() < 1e-5
    torch.testing.assert_close(U_best, U[:, k].float())
