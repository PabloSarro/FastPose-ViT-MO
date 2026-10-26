"""SymmetricRotation6DLoss (M5): min of the relative 6D Frobenius loss over each object's symmetries.

- Loss = 0 for every R_gt·S_k (any symmetric pose is a perfect prediction).
- Equal to RelativeRotation6DFrobeniusLoss for objects with the identity only.
- Finite gradients, also at a tie between two symmetric targets.
"""

import os

import numpy as np
import pytest
import torch

from src.bop_dataset import LMO_OBJ_IDS, default_bop_root, symmetry_table
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
    table = symmetry_table(default_bop_root(), "lmo", LMO_OBJ_IDS)
    assert table.shape == (8, 2, 3, 3)
    np.testing.assert_allclose(table[:, 0], np.tile(np.eye(3), (8, 1, 1)), atol=1e-6)
    flipped = [o for i, o in enumerate(LMO_OBJ_IDS) if not np.allclose(table[i, 1], np.eye(3))]
    assert flipped == [10, 11]  # eggbox and glue
    # A ~180 deg flip, as measured on the scanned models: eggbox 179.8 deg, glue 178.5 deg
    for i in (LMO_OBJ_IDS.index(10), LMO_OBJ_IDS.index(11)):
        S = table[i, 1]
        assert np.degrees(np.arccos(np.clip((np.trace(S) - 1) / 2, -1, 1))) > 178
