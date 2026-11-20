"""Point-splatting renderer (M19.1): analytic sphere, then the real LM models against the PBR
depth images and visible-object masks (the GT pose must reproduce them)."""

import json
import os

import cv2
import numpy as np
import pytest
import torch

from src.bop_dataset import (
    LMO_OBJ_IDS,
    BOPDataset,
    Source,
    build_index,
    crop_warp_matrix,
    default_bop_root,
    nearest_rotation,
    object_extents,
)
from src.render import PointRenderer, crop_intrinsics, load_render_assets

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sphere_assets(radius=50.0, n=65536):  # the production number of points
    rng = np.random.default_rng(0)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return {
        "points": (v * radius)[None].astype(np.float32),
        "normals": v[None].astype(np.float32),
        "colors": np.full((1, n, 3), 0.8, np.float32),
    }


def test_sphere_mask_depth_and_xyz():
    renderer = PointRenderer(sphere_assets(), centre=np.zeros((1, 3)), half=np.full((1, 3), 50.0), device=DEV)
    K = np.array([[500.0, 0, 64], [0, 500.0, 64], [0, 0, 1]])
    t = np.array([0.0, 0.0, 400.0])
    image, depth = renderer.render([0], np.eye(3)[None], t[None], K[None], 128)
    mask = image[0, 6].cpu().numpy() > 0.5
    # analytic silhouette: a circle of radius f * r / sqrt(d^2 - r^2) around the principal point
    d, r = 400.0, 50.0
    radius_px = 500.0 * r / np.sqrt(d * d - r * r)
    yy, xx = np.mgrid[0:128, 0:128] + 0.5
    # one pixel per point: the silhouette is within about half a pixel of the analytic circle
    circle = (xx - 64) ** 2 + (yy - 64) ** 2 <= (radius_px + 0.5) ** 2
    assert (mask & circle).sum() / (mask | circle).sum() > 0.97
    tight = (xx - 64) ** 2 + (yy - 64) ** 2 <= (radius_px - 1.5) ** 2
    assert mask[tight].all()  # no holes inside
    # the nearest surface point is at depth d - r, with object coordinates (0, 0, -1)
    assert depth[0, 0, 64, 64].item() == pytest.approx(d - r, abs=1.0)
    np.testing.assert_allclose(image[0, 3:6, 64, 64].cpu().numpy(), [0, 0, -1], atol=0.05)
    # behind the camera or outside the crop: nothing
    image, _ = renderer.render([0], np.eye(3)[None], np.array([[0.0, 0.0, -400.0]]), K[None], 128)
    assert image[0, 6].sum() == 0
    image, _ = renderer.render([0], np.eye(3)[None], np.array([[5000.0, 0.0, 400.0]]), K[None], 128)
    assert image[0, 6].sum() == 0


def test_nearest_point_wins_and_batch_is_independent():
    """Two instances in one batch do not mix, and a far sphere behind a near one is hidden."""
    renderer = PointRenderer(sphere_assets(), centre=np.zeros((1, 3)), half=np.full((1, 3), 50.0), device=DEV)
    K = np.tile(np.array([[500.0, 0, 64], [0, 500.0, 64], [0, 0, 1]]), (2, 1, 1))
    t = np.array([[0.0, 0.0, 300.0], [0.0, 0.0, 600.0]])
    both, _ = renderer.render([0, 0], np.tile(np.eye(3), (2, 1, 1)), t, K, 128)
    one, _ = renderer.render([0], np.eye(3)[None], t[1:], K[1:], 128)
    assert torch.equal(both[1], one[0])
    assert both[0, 6].sum() > both[1, 6].sum()  # nearer = larger


def test_crop_intrinsics_matches_the_warp():
    K = np.array([[572.4, 0, 325.3], [0, 573.6, 242.0], [0, 0, 1.0]])
    bbox = [100.0, 80.0, 220.0, 190.0]
    M = crop_warp_matrix(bbox, 128, 1.4)
    Kc = crop_intrinsics(K, M)
    X = np.array([[10.0, -20.0, 600.0], [-35.0, 15.0, 700.0]])
    u = (K @ X.T).T
    u = u[:, :2] / u[:, 2:]  # image coordinates (K coords)
    c = (Kc @ X.T).T
    c = c[:, :2] / c[:, 2:]  # crop coordinates, same convention
    for ui, ci in zip(u, c):
        # cv2.warpAffine maps index (u - 0.5) to index M @ (index, 1); crop coordinate = index + 0.5
        index = M @ np.array([ui[0] - 0.5, ui[1] - 0.5, 1.0])
        np.testing.assert_allclose(ci, index + 0.5, atol=1e-9)


BOP_ROOT = default_bop_root()
pytestmark_real = pytest.mark.skipif(
    not os.path.exists(os.path.join(BOP_ROOT, "lm", "models", "obj_000001.ply")), reason="no BOP tree"
)


@pytestmark_real
def test_render_reproduces_pbr_depth_and_mask():
    assets = load_render_assets(BOP_ROOT, "lmo", LMO_OBJ_IDS, 32768)
    centre, half = object_extents(BOP_ROOT, "lmo", LMO_OBJ_IDS)
    renderer = PointRenderer(assets, centre, half, DEV)
    index = build_index(BOP_ROOT, (Source("lm", "train_pbr", (0,)),), LMO_OBJ_IDS, "lmo")
    ds = BOPDataset(dataset="lmo", split="val", index=index, img_size=64, augment=False)
    scene = os.path.join(BOP_ROOT, "lm", "train_pbr", "000000")
    info = json.load(open(os.path.join(scene, "scene_gt_info.json")))
    depth_errors, ious = [], []
    for row in range(len(ds)):
        im, gt = int(index["im_id"][row]), int(index["gt_id"][row])
        if info[str(im)][gt]["visib_fract"] < 0.97:
            continue
        bbox = [float(v) for v in ds.crop_bbox(row)]
        M = crop_warp_matrix(bbox, 160, 1.4)
        Kc = crop_intrinsics(index["K"][row].astype(np.float64), M)
        R = nearest_rotation(index["R"][row].astype(np.float64))
        image, depth = renderer.render([int(index["obj_idx"][row])], R[None], index["t"][row][None], Kc[None], 160)
        mask_png = cv2.imread(os.path.join(scene, "mask_visib", f"{im:06d}_{gt:06d}.png"), 0)
        depth_png = cv2.imread(os.path.join(scene, "depth", f"{im:06d}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32)
        depth_png *= 0.1  # depth_scale of the PBR scenes: mm
        warp = lambda a, flag: cv2.warpAffine(a, M, (160, 160), flags=flag)  # noqa: E731
        ref_mask = warp(mask_png, cv2.INTER_NEAREST) > 0
        ref_depth = warp(depth_png, cv2.INTER_NEAREST)
        mask = image[0, 6].cpu().numpy() > 0.5
        ious.append((mask & ref_mask).sum() / max((mask | ref_mask).sum(), 1))
        both = mask & ref_mask & (ref_depth > 0)
        depth_errors.append(np.median(np.abs(depth[0, 0].cpu().numpy()[both] - ref_depth[both])))
        if len(ious) >= 25:
            break
    assert len(ious) >= 10
    assert np.median(ious) > 0.93, np.median(ious)
    assert np.median(depth_errors) < 2.0, np.median(depth_errors)  # mm (point spacing, splat size)
