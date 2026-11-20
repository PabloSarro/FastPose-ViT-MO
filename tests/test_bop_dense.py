"""Dense targets of BOPDataset (M18.1): mask and normalised object coordinates (XYZ).

Checked on the real LM PBR / LM-O test files against independent references:
- the denormalised XYZ lie on the model surface (distance to the nearest model vertex), and
  projecting R·X + t lands on the crop pixel the target belongs to;
- the mask is mask_visib sampled at the crop pixels (compared with cv2.warpAffine nearest);
- the black bands outside the crop box carry no target, as in the image;
- depth_scale handles PBR (0.1) and real (1.0) images; points off the model box are not valid.

Skipped when the BOP tree is not available.
"""

import os

import cv2
import numpy as np
import pytest
import torch
from bop_toolkit_lib import inout
from scipy.spatial import cKDTree

from src.bop_dataset import (
    LMO_OBJ_IDS,
    SPLITS,
    BOPDataset,
    Source,
    build_index,
    crop_warp_matrix,
    default_bop_root,
    nearest_rotation,
    object_extents,
)

BOP_ROOT = default_bop_root()
pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(BOP_ROOT, "lmo", "test_targets_bop19.json")),
    reason=f"BOP tree not found at {BOP_ROOT}",
)
SIZE = 64


@pytest.fixture(scope="module")
def pbr_dataset():
    index = build_index(BOP_ROOT, (Source("lm", "train_pbr", (0, 17)),), LMO_OBJ_IDS, "lmo")
    return BOPDataset(
        dataset="lmo", split="val", index=index, img_size=128, dense_size=SIZE, augment=False
    )


@pytest.fixture(scope="module")
def test_dataset():
    index = build_index(BOP_ROOT, SPLITS["lmo"]["test"], LMO_OBJ_IDS, "lmo")
    return BOPDataset(
        dataset="lmo", split="test", index=index, img_size=128, dense_size=SIZE, augment=False
    )


@pytest.fixture(scope="module")
def trees():
    return {
        o: cKDTree(
            inout.load_ply(os.path.join(BOP_ROOT, "lmo", "models_eval", f"obj_{o:06d}.ply"))["pts"]
        )
        for o in LMO_OBJ_IDS
    }


def denormalise(ds, sample):
    """Object coordinates (mm) of the valid pixels of a sample, and their (row, col)."""
    obj = sample["obj_idx"]
    ys, xs = np.nonzero(sample["xyz_valid"][0].numpy() > 0)
    xyz = sample["xyz"].numpy()[:, ys, xs].T  # (n, 3)
    return xyz * ds.xyz_half[obj] + ds.xyz_centre[obj], ys, xs


def test_object_extents_match_models_info():
    centre, half = object_extents(BOP_ROOT, "lmo", LMO_OBJ_IDS)
    assert centre.shape == half.shape == (8, 3)
    assert (half > 0).all()
    # an LM model is centred near its frame origin; the half-extent is half its size, below d/2
    assert np.abs(centre).max() < 150 and half.max() < 200


@pytest.mark.parametrize("which", ["pbr", "real"])
def test_xyz_lies_on_the_model_surface(which, pbr_dataset, test_dataset, trees):
    ds = pbr_dataset if which == "pbr" else test_dataset
    dists, counts = [], []
    for row in range(0, len(ds), max(1, len(ds) // 40)):
        s = ds[row]
        pts, _, _ = denormalise(ds, s)
        counts.append(len(pts))
        if len(pts):
            d, _ = trees[s["obj_id"]].query(pts)
            dists.append(np.median(d))
    assert np.mean(np.array(counts) > 10) > 0.9  # almost every instance has valid pixels
    # PBR: depth is the rendered one, so the points are on the surface up to the vertex
    # spacing (measured 1.2 mm); real LM-O depth + GT pose are less exact (measured 3 mm)
    limit = 3.0 if which == "pbr" else 6.0
    assert np.median(dists) < limit, np.median(dists)


def test_wrong_pose_fails_the_surface_check(pbr_dataset, trees):
    """The check above has power: points with R instead of Rᵀ are far from the model."""
    far = []
    for row in range(0, len(pbr_dataset), len(pbr_dataset) // 30):
        s = pbr_dataset[row]
        pts, _, _ = denormalise(pbr_dataset, s)
        if len(pts) == 0:
            continue
        # pose-inverse mistake: swap the rotation used for the back-projection
        R = nearest_rotation(s["R_m2c"].double().numpy())
        t = s["t_m2c"].double().numpy()
        wrong = (R @ pts.T).T  # not R^T(p - t) of any consistent point
        far.append(np.median(trees[s["obj_id"]].query(wrong + t)[0]))
    assert np.median(far) > 10.0


def test_projection_lands_on_the_crop_pixel(pbr_dataset):
    for row in range(0, len(pbr_dataset), len(pbr_dataset) // 25):
        s = pbr_dataset[row]
        pts, ys, xs = denormalise(pbr_dataset, s)
        if len(pts) == 0:
            continue
        R = nearest_rotation(s["R_m2c"].double().numpy())
        t = s["t_m2c"].double().numpy()
        K = s["K"].double().numpy()
        cam = pts @ R.T + t
        u, v = K[0, 0] * cam[:, 0] / cam[:, 2] + K[0, 2], K[1, 1] * cam[:, 1] / cam[:, 2] + K[1, 2]
        M = crop_warp_matrix(s["bboxes"].double().tolist(), SIZE)
        # crop pixel j holds K coordinate (j - tx) / s + 0.5
        np.testing.assert_allclose(u, (xs - M[0, 2]) / M[0, 0] + 0.5, atol=2e-2)
        np.testing.assert_allclose(v, (ys - M[1, 2]) / M[1, 1] + 0.5, atol=2e-2)


def test_mask_is_mask_visib_sampled_at_the_crop_pixels(pbr_dataset):
    """Independent reference: warpAffine (nearest) of mask_visib with the crop matrix. The two
    differ only where a crop pixel centre falls exactly on a source pixel edge (a few pixels)."""
    ds = pbr_dataset
    for row in range(0, len(ds), len(ds) // 20):
        s = ds[row]
        raw = cv2.imread(ds.mask_path(row), cv2.IMREAD_GRAYSCALE)
        M = crop_warp_matrix(s["bboxes"].double().tolist(), SIZE)
        ref = cv2.warpAffine(raw, M, (SIZE, SIZE), flags=cv2.INTER_NEAREST) > 0
        got = s["mask"][0].numpy() > 0
        # the box only cuts away the bands: inside the box the two agree
        x1, y1, x2, y2 = s["bboxes"].tolist()
        u = (np.arange(SIZE) - M[0, 2]) / M[0, 0] + 0.5
        v = (np.arange(SIZE) - M[1, 2]) / M[1, 1] + 0.5
        inside = ((v >= y1) & (v <= y2))[:, None] & ((u >= x1) & (u <= x2))[None, :]
        assert (ref[inside] != got[inside]).mean() < 0.03
        assert not got[~inside].any()
        assert got.sum() > 0  # bbox_visib of a row with visib_fract >= 0.1 contains the mask


def test_bands_outside_a_shrunk_box_are_empty(pbr_dataset):
    ds = pbr_dataset
    row = len(ds) // 3
    x1, y1, x2, y2 = (float(v) for v in ds.crop_bbox(row))
    shrunk = [x1 + 0.3 * (x2 - x1), y1, x2, y2]  # cut the left 30 % of the box
    mask, xyz, valid = ds.dense_targets(row, shrunk)
    full_mask, _, _ = ds.dense_targets(row, [x1, y1, x2, y2])
    M = crop_warp_matrix(shrunk, SIZE)
    left_of_box = (np.arange(SIZE) - M[0, 2]) / M[0, 0] + 0.5 < shrunk[0]
    assert not mask[:, left_of_box].any()
    assert not valid[:, left_of_box].any() and not xyz[:, :, left_of_box].any()
    assert full_mask.sum() > 0


def test_invalid_pixels_are_zero_and_values_in_range(pbr_dataset):
    for row in range(0, len(pbr_dataset), len(pbr_dataset) // 20):
        s = pbr_dataset[row]
        assert s["mask"].shape == (1, SIZE, SIZE) and s["xyz"].shape == (3, SIZE, SIZE)
        assert s["xyz_valid"].shape == (1, SIZE, SIZE)
        assert s["mask"].dtype == s["xyz"].dtype == s["xyz_valid"].dtype == torch.float32
        valid = s["xyz_valid"][0] > 0
        assert (s["xyz_valid"] <= s["mask"]).all()  # valid pixels are masked pixels
        assert (s["xyz"][:, ~valid] == 0).all()
        assert s["xyz"].abs().max() <= 1.15 + 1e-6


def test_depth_scale_pbr_and_real(pbr_dataset, test_dataset):
    assert pbr_dataset.depth_scale(0) == pytest.approx(0.1)
    assert test_dataset.depth_scale(0) == pytest.approx(1.0)
    # the sampled depth is plausible in mm: close to the object distance t_z
    for ds in (pbr_dataset, test_dataset):
        s = ds[len(ds) // 2]
        pts, _, _ = denormalise(ds, s)
        R = nearest_rotation(s["R_m2c"].double().numpy())
        z = (pts @ R.T + s["t_m2c"].double().numpy())[:, 2]
        assert abs(np.median(z) - s["t_m2c"][2].item()) < s["diameter"].item()


def test_no_dense_keys_by_default(pbr_dataset):
    plain = BOPDataset(split="val", index=pbr_dataset.index, img_size=128, augment=False)
    s = plain[0]
    assert "mask" not in s and "xyz" not in s and "xyz_valid" not in s
    ref = pbr_dataset[0]
    for k in ("images", "translations", "rotations", "bboxes"):
        assert torch.equal(s[k], ref[k]), k  # the dense targets do not change the rest
