"""Rotation conventions used by the SPEED pipeline, and why they carry over to BOP.

- quaternion_to_rotation_matrix(q) is the transpose of ESA's quat2dcm(q), so the
  network regresses R model->camera (x_cam = R x_model + t), like BOP's cam_R_m2c.
- get_apparent_orientation (ego -> allo, R_rel(t)^T R) and get_absolute_orientation
  (allo -> ego, R_rel(t) R) are inverses, where R_rel(t) is the smallest rotation
  taking the optical axis z to the viewing ray t/|t|.
- A camera roll about z rotates the allocentric orientation by the same roll.
"""

import math

import numpy as np
import pytest
import torch

from other_utils.visualization import Camera as ESACamera
from other_utils.visualization import project_axes, quat2dcm
from src.utils import (
    _compute_orientation_transform,
    get_absolute_orientation,
    get_apparent_orientation,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
)

N = 256
ATOL = 1e-9  # float64 throughout


def random_quaternions(n, seed=0):
    q = torch.randn(n, 4, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    return q / q.norm(dim=-1, keepdim=True)


def random_translations(n, seed=1):
    """Translations in front of the camera, from on-axis to ~60 deg off-axis."""
    g = torch.Generator().manual_seed(seed)
    t = torch.empty(n, 3, dtype=torch.float64)
    t[:, 2] = torch.rand(n, generator=g, dtype=torch.float64) * 1900 + 100
    t[:, :2] = (torch.rand(n, 2, generator=g, dtype=torch.float64) * 2 - 1) * 1.7 * t[:, 2:]
    t[0, :2] = 0.0  # exactly on the optical axis (parallel case)
    return t


def rot_z(theta):
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)


def to_matrix(q):
    return quaternion_to_rotation_matrix(q)


def test_quat2dcm_is_transpose_of_quaternion_to_rotation_matrix():
    q = random_quaternions(N)
    R = to_matrix(q).numpy()
    for i in range(N):
        np.testing.assert_allclose(quat2dcm(q[i].numpy()).T, R[i], atol=ATOL)


def test_esa_projection_uses_model_to_camera_rotation():
    """project_axes (ESA) projects K (R p + t) with R = quaternion_to_rotation_matrix(q)."""
    q = random_quaternions(16, seed=2).numpy()
    t = np.array([0.3, -0.2, 8.0])
    K = ESACamera.K
    axes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64).T
    for qi in q:
        x, y = project_axes(qi, t)
        R = to_matrix(torch.from_numpy(qi)).numpy()
        p = K @ (R @ axes + t[:, None])
        np.testing.assert_allclose(x, p[0] / p[2], atol=1e-6)
        np.testing.assert_allclose(y, p[1] / p[2], atol=1e-6)


def test_r_rel_maps_optical_axis_to_viewing_ray():
    t = random_translations(N)
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64).expand(N, -1)
    R_rel = to_matrix(_compute_orientation_transform(t, identity, inverse=False))
    z = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    torch.testing.assert_close(R_rel @ z, t / t.norm(dim=-1, keepdim=True), atol=ATOL, rtol=0)
    # Smallest rotation: its axis is perpendicular to z, so it has no roll about the ray.
    assert torch.allclose(R_rel[0], torch.eye(3, dtype=torch.float64), atol=ATOL)
    axis = torch.stack(
        [
            R_rel[:, 2, 1] - R_rel[:, 1, 2],
            R_rel[:, 0, 2] - R_rel[:, 2, 0],
            R_rel[:, 1, 0] - R_rel[:, 0, 1],
        ],
        dim=-1,
    )
    torch.testing.assert_close(axis[:, 2], torch.zeros(N, dtype=torch.float64), atol=ATOL, rtol=0)


def test_ego_allo_ego_round_trip():
    q, t = random_quaternions(N), random_translations(N)
    allo = get_apparent_orientation(t, q)
    ego = get_absolute_orientation(t, allo)
    torch.testing.assert_close(to_matrix(ego), to_matrix(q), atol=ATOL, rtol=0)
    # And the other way round.
    back = get_apparent_orientation(t, get_absolute_orientation(t, q))
    torch.testing.assert_close(to_matrix(back), to_matrix(q), atol=ATOL, rtol=0)


def test_allocentric_is_ego_rotation_seen_along_the_ray():
    """allo = R_rel(t)^T R, i.e. the object's rotation as if it sat on the optical axis."""
    q, t = random_quaternions(N), random_translations(N)
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64).expand(N, -1)
    R_rel = to_matrix(_compute_orientation_transform(t, identity, inverse=False))
    allo = to_matrix(get_apparent_orientation(t, q))
    torch.testing.assert_close(allo, R_rel.transpose(-2, -1) @ to_matrix(q), atol=ATOL, rtol=0)


@pytest.mark.parametrize("theta", [0.3, -1.2, math.pi / 2, 2.9])
def test_roll_equivariance(theta):
    """Rolling the camera by Rz (t' = Rz t, R' = Rz R) rolls the allocentric rotation: allo' = Rz allo."""
    q, t = random_quaternions(N), random_translations(N)
    Rz = rot_z(theta)
    R = to_matrix(q)
    t_rolled = t @ Rz.T
    q_rolled = rotation_matrix_to_quaternion(Rz @ R)
    allo = to_matrix(get_apparent_orientation(t, q))
    allo_rolled = to_matrix(get_apparent_orientation(t_rolled, q_rolled))
    torch.testing.assert_close(allo_rolled, Rz @ allo, atol=1e-8, rtol=0)
