"""Bbox-relative translation parameterisation U = (U_x, U_y, U_z).

- With K = Camera.K and a 1920x1200 image, the new per-sample-K code path matches the
  original SPEED path bit for bit (float32, as in training).
- The "site" depth mode (U_z = t_z s / (f D)) inverts exactly with the LM and YCB-V cameras,
  per-sample K and per-sample diameters, and gives U_z close to 1 for realistic boxes.
"""

import pytest
import torch

from src.camera import Camera
from src.utils import (
    bbox_relative_translation_to_translation,
    translation_to_bbox_relative_translation,
)

N = 512

# BOP intrinsics (camera.json / scene_camera.json); translations in mm.
LM_K = [[572.4114, 0.0, 325.2611], [0.0, 573.57043, 242.04899], [0.0, 0.0, 1.0]]
YCBV_UW_K = [[1066.778, 0.0, 312.9869], [0.0, 1067.487, 241.3109], [0.0, 0.0, 1.0]]
YCBV_CMU_K = [[1077.836, 0.0, 323.7872], [0.0, 1078.189, 279.6921], [0.0, 0.0, 1.0]]
CAMERAS = {"lm": LM_K, "ycbv_uw": YCBV_UW_K, "ycbv_cmu": YCBV_CMU_K}


def speed_batch(n=N, seed=0):
    """SPEED-like poses (metres) and boxes around the projected centre, float32."""
    g = torch.Generator().manual_seed(seed)
    t = torch.empty(n, 3)
    t[:, 2] = torch.rand(n, generator=g) * 40 + 3
    t[:, :2] = (torch.rand(n, 2, generator=g) * 2 - 1) * 0.25 * t[:, 2:]
    K = Camera.K
    u = K[0, 0] * t[:, 0] / t[:, 2] + K[0, 2]
    v = K[1, 1] * t[:, 1] / t[:, 2] + K[1, 2]
    size = (3000 / t[:, 2]).unsqueeze(1) * (0.5 + torch.rand(n, 2, generator=g))
    jitter = (torch.rand(n, 2, generator=g) - 0.5) * 0.3 * size
    centre = torch.stack([u, v], dim=1) + jitter
    bbox = torch.cat([centre - size / 2, centre + size / 2], dim=1)
    return t, bbox


def bop_batch(K, n=N, seed=1, dtype=torch.float64):
    """BOP-like poses (mm), diameters 100-270 mm and boxes from the projected sphere."""
    g = torch.Generator().manual_seed(seed)
    K = torch.as_tensor(K, dtype=dtype)
    t = torch.empty(n, 3, dtype=dtype)
    t[:, 2] = torch.rand(n, generator=g, dtype=dtype) * 1500 + 400
    t[:, :2] = (torch.rand(n, 2, generator=g, dtype=dtype) * 2 - 1) * 0.35 * t[:, 2:]
    diameter = torch.rand(n, generator=g, dtype=dtype) * 170 + 100
    u = K[0, 0] * t[:, 0] / t[:, 2] + K[0, 2]
    v = K[1, 1] * t[:, 1] / t[:, 2] + K[1, 2]
    # Visible box: projected diameter, shrunk/stretched per axis, with a centre offset.
    proj = (0.5 * (K[0, 0] + K[1, 1]) * diameter / t[:, 2]).unsqueeze(1)
    size = proj * (0.6 + 0.5 * torch.rand(n, 2, generator=g, dtype=dtype))
    centre = (
        torch.stack([u, v], dim=1)
        + (torch.rand(n, 2, generator=g, dtype=dtype) - 0.5) * 0.2 * size
    )
    bbox = torch.cat([centre - size / 2, centre + size / 2], dim=1)
    return t, bbox, diameter


@pytest.mark.parametrize("K_shape", ["single", "per_sample"])
@pytest.mark.parametrize("img_wh", ["tuple", "tensor"])
def test_k_path_matches_legacy_bit_for_bit(K_shape, img_wh):
    t, bbox = speed_batch()
    K = Camera.K if K_shape == "single" else Camera.K.expand(N, 3, 3)
    wh = (1920, 1200) if img_wh == "tuple" else torch.tensor([[1920.0, 1200.0]]).expand(N, 2)

    U_old = translation_to_bbox_relative_translation(t, bbox, Camera)
    U_new = translation_to_bbox_relative_translation(t, bbox, Camera, K=K, img_wh=wh)
    assert U_old.dtype == U_new.dtype == torch.float32
    assert torch.equal(U_new, U_old)

    t_old = bbox_relative_translation_to_translation(U_old, bbox, Camera)
    t_new = bbox_relative_translation_to_translation(U_old, bbox, Camera, K=K, img_wh=wh)
    assert torch.equal(t_new, t_old)


def test_legacy_depth_is_tz_times_mean_box_side_over_height():
    t, bbox = speed_batch()
    U = translation_to_bbox_relative_translation(t, bbox, Camera)
    w, h = bbox[:, 2] - bbox[:, 0], bbox[:, 3] - bbox[:, 1]
    torch.testing.assert_close(U[:, 2], t[:, 2] * (w + h) / (2 * Camera.height))


def test_legacy_round_trip():
    t, bbox = speed_batch()
    t_back = bbox_relative_translation_to_translation(
        translation_to_bbox_relative_translation(t, bbox, Camera), bbox, Camera
    )
    torch.testing.assert_close(t_back, t, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("camera", CAMERAS)
def test_site_round_trip_bop_cameras(camera):
    t, bbox, diameter = bop_batch(CAMERAS[camera])
    K = torch.tensor(CAMERAS[camera], dtype=torch.float64)
    kw = dict(K=K, depth_mode="site", diameter=diameter)
    U = translation_to_bbox_relative_translation(t, bbox, Camera, **kw)
    t_back = bbox_relative_translation_to_translation(U, bbox, Camera, **kw)
    torch.testing.assert_close(t_back, t, rtol=1e-12, atol=1e-9)
    # Dimensionless and ~1 (it is ~1.2 for an object box spanning its full diameter).
    assert 0.4 < U[:, 2].min() and U[:, 2].max() < 1.6, U[:, 2].aminmax()
    # The 2D centre offsets stay small: |U_x|, |U_y| < 0.5 since the jitter is +-10 %.
    assert U[:, :2].abs().max() < 0.5


def test_site_round_trip_mixed_cameras_per_sample():
    """A YCB-V-like batch mixing both cameras, with per-sample K and diameters."""
    t_uw, bbox_uw, d_uw = bop_batch(YCBV_UW_K, n=N // 2, seed=3)
    t_cmu, bbox_cmu, d_cmu = bop_batch(YCBV_CMU_K, n=N // 2, seed=4)
    t, bbox, diameter = (
        torch.cat([t_uw, t_cmu]),
        torch.cat([bbox_uw, bbox_cmu]),
        torch.cat([d_uw, d_cmu]),
    )
    K = torch.cat(
        [
            torch.tensor(YCBV_UW_K, dtype=torch.float64).expand(N // 2, 3, 3),
            torch.tensor(YCBV_CMU_K, dtype=torch.float64).expand(N // 2, 3, 3),
        ]
    )
    kw = dict(K=K, depth_mode="site", diameter=diameter)
    U = translation_to_bbox_relative_translation(t, bbox, Camera, **kw)
    t_back = bbox_relative_translation_to_translation(U, bbox, Camera, **kw)
    torch.testing.assert_close(t_back, t, rtol=1e-12, atol=1e-9)
    # Using the wrong camera for half the batch must break the round trip.
    t_wrong = bbox_relative_translation_to_translation(
        U, bbox, Camera, K=K[0], depth_mode="site", diameter=diameter
    )
    assert (t_wrong[N // 2 :] - t[N // 2 :]).abs().max() > 1.0


def test_site_depth_is_invariant_to_object_size_and_focal_length():
    """Same apparent geometry (box scales with f * D / t_z) -> same U_z."""
    t, bbox, diameter = bop_batch(LM_K)
    K = torch.tensor(LM_K, dtype=torch.float64)
    U = translation_to_bbox_relative_translation(
        t, bbox, Camera, K=K, depth_mode="site", diameter=diameter
    )
    # Twice the diameter at twice the distance: same box, same U_z.
    t2 = t * torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64)
    U2 = translation_to_bbox_relative_translation(
        t2, bbox, Camera, K=K, depth_mode="site", diameter=2 * diameter
    )
    torch.testing.assert_close(U2, U)


def test_float32_site_round_trip():
    t, bbox, diameter = bop_batch(LM_K, dtype=torch.float32)
    K = torch.tensor(LM_K)
    kw = dict(K=K, depth_mode="site", diameter=diameter)
    t_back = bbox_relative_translation_to_translation(
        translation_to_bbox_relative_translation(t, bbox, Camera, **kw), bbox, Camera, **kw
    )
    torch.testing.assert_close(t_back, t, rtol=1e-5, atol=1e-3)  # < 1 um at ~1 m


def test_bad_arguments():
    t, bbox = speed_batch(n=4)
    with pytest.raises(ValueError, match="depth_mode"):
        translation_to_bbox_relative_translation(t, bbox, Camera, K=Camera.K, depth_mode="foo")
    with pytest.raises(ValueError, match="img_wh"):
        translation_to_bbox_relative_translation(t, bbox, Camera, K=Camera.K)
    with pytest.raises(ValueError, match="diameter"):
        translation_to_bbox_relative_translation(t, bbox, Camera, K=Camera.K, depth_mode="site")
