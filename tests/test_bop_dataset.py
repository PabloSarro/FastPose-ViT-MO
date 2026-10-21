"""BOP index (src/bop_dataset.py), on the real LM / LM-O files.

- Training rows: exactly the LM-O-object instances with visib_fract >= 0.1 and a valid bbox,
  recounted independently from the raw JSON of one PBR scene.
- Test rows: exactly the 1,445 LM-O BOP19 targets, so the evaluation sees every target.
- Row contents: pose, K, bboxes and image path match the raw files; obj_idx follows obj_ids.
- Cache: written once, reloaded identically, and the file name changes with the settings.

Skipped when the BOP tree (scripts/make_bop_tree.sh) is not available.
"""

import json
import os

import numpy as np
import pytest

from src.bop_dataset import (
    LMO_OBJ_IDS,
    SPLITS,
    Source,
    build_index,
    default_bop_root,
    image_path,
    index_key,
    load_index,
)

BOP_ROOT = default_bop_root()
pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(BOP_ROOT, "lmo", "test_targets_bop19.json")),
    reason=f"BOP tree not found at {BOP_ROOT}",
)
PBR_SCENE = os.path.join(BOP_ROOT, "lm", "train_pbr", "000000")


def load(path):
    with open(path) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def pbr_index():
    return build_index(BOP_ROOT, (Source("lm", "train_pbr", (0,)),), LMO_OBJ_IDS, "lmo")


@pytest.fixture(scope="module")
def test_index():
    return build_index(BOP_ROOT, SPLITS["lmo"]["test"], LMO_OBJ_IDS, "lmo")


def test_pbr_rows_match_raw_annotations(pbr_index):
    gt = load(os.path.join(PBR_SCENE, "scene_gt.json"))
    info = load(os.path.join(PBR_SCENE, "scene_gt_info.json"))
    expected = [
        (int(im), i)
        for im in gt
        for i, (g, n) in enumerate(zip(gt[im], info[im]))
        if g["obj_id"] in LMO_OBJ_IDS and n["visib_fract"] >= 0.1 and min(n["bbox_visib"][2:]) > 0
    ]
    assert len(expected) == 6866  # measured on 2026-10-20 (TASKS.md §5)
    rows = sorted(zip(pbr_index["im_id"].tolist(), pbr_index["gt_id"].tolist()))
    assert rows == sorted(expected)


def test_test_rows_are_exactly_the_bop19_targets(test_index):
    targets = load(os.path.join(BOP_ROOT, "lmo", "test_targets_bop19.json"))
    assert len(test_index["source"]) == sum(t["inst_count"] for t in targets) == 1445
    expected = sorted(
        (t["scene_id"], t["im_id"], t["obj_id"]) for t in targets for _ in range(t["inst_count"])
    )
    got = sorted(
        zip(
            test_index["scene_id"].tolist(),
            test_index["im_id"].tolist(),
            test_index["obj_id"].tolist(),
        )
    )
    assert got == expected


@pytest.mark.parametrize("which", ["pbr", "test"])
def test_row_contents_match_raw_files(which, pbr_index, test_index):
    index = pbr_index if which == "pbr" else test_index
    rng = np.random.default_rng(0)
    for row in rng.choice(len(index["source"]), 20, replace=False):
        scene_dir = os.path.dirname(os.path.dirname(image_path(index, row)))
        im = str(index["im_id"][row])
        gt = load(os.path.join(scene_dir, "scene_gt.json"))[im][index["gt_id"][row]]
        info = load(os.path.join(scene_dir, "scene_gt_info.json"))[im][index["gt_id"][row]]
        cam = load(os.path.join(scene_dir, "scene_camera.json"))[im]
        assert gt["obj_id"] == index["obj_id"][row] == LMO_OBJ_IDS[index["obj_idx"][row]]
        np.testing.assert_array_equal(index["R"][row], np.reshape(gt["cam_R_m2c"], (3, 3)))
        np.testing.assert_array_equal(index["t"][row], gt["cam_t_m2c"])
        np.testing.assert_array_equal(index["K"][row], np.reshape(cam["cam_K"], (3, 3)))
        np.testing.assert_array_equal(index["bbox_obj"][row], info["bbox_obj"])
        np.testing.assert_array_equal(index["bbox_visib"][row], info["bbox_visib"])
        assert os.path.exists(image_path(index, row))
    assert image_path(index, 0).endswith(".jpg" if which == "pbr" else ".png")
    np.testing.assert_array_equal(index["img_wh"], np.tile([640, 480], (len(index["source"]), 1)))


def test_object_table(test_index):
    np.testing.assert_array_equal(test_index["obj_ids"], LMO_OBJ_IDS)
    assert test_index["diameters"][0] == pytest.approx(102.099, abs=1e-3)  # ape
    assert test_index["diameters"][3] == pytest.approx(261.472, abs=1e-3)  # driller
    assert test_index["symmetric"].tolist() == [False] * 5 + [True, True, False]  # eggbox, glue


def test_cache_round_trip(tmp_path):
    first = load_index("lmo", "test", BOP_ROOT, str(tmp_path))
    files = os.listdir(tmp_path)
    assert len(files) == 1 and files[0].startswith("lmo_test_") and files[0].endswith(".npz")
    again = load_index("lmo", "test", BOP_ROOT, str(tmp_path))
    assert first.keys() == again.keys()
    for k in first:
        np.testing.assert_array_equal(first[k], again[k])
    assert index_key("lmo", "train", BOP_ROOT, 0.1) != index_key("lmo", "train", BOP_ROOT, 0.2)
    with pytest.raises(ValueError):
        load_index("lmo", "obj_ids", BOP_ROOT, str(tmp_path))


def test_val_real_excludes_the_lmo_test_scene():
    val = SPLITS["lmo"]["val_real"][0]
    assert val.dataset == "lm" and 2 not in val.scenes


# --- BOPDataset.__getitem__: crop warp, labels, augmentations ---------------------------------

import albumentations as A  # noqa: E402
import torch  # noqa: E402

from src.bop_dataset import BOPDataset, crop_image, crop_warp_matrix, pixel_transform  # noqa: E402
from src.engine import unpack_batch  # noqa: E402


def apply(M, x, y):
    return M[0, 0] * x + M[0, 2], M[1, 1] * y + M[1, 2]


def test_crop_warp_maps_bbox_to_centred_square():
    for bbox in ([100.0, 50.0, 180.0, 250.0], [10.5, 20.25, 310.0, 60.0]):
        M = crop_warp_matrix(bbox, 384)
        x1, y1, x2, y2 = bbox
        np.testing.assert_allclose(apply(M, (x1 + x2) / 2, (y1 + y2) / 2), (191.5, 191.5))
        side = max(x2 - x1, y2 - y1)
        assert M[0, 0] == M[1, 1] == pytest.approx(384 / side)  # same scale on both axes
        # the long side spans the crop exactly
        lo, hi = apply(M, x1, y1), apply(M, x2, y2)
        long_axis = 0 if x2 - x1 >= y2 - y1 else 1
        assert hi[long_axis] - lo[long_axis] == pytest.approx(384)


def test_crop_black_bands_and_context():
    image = np.full((480, 640, 3), 200, np.uint8)
    bbox = [300.0, 100.0, 340.0, 300.0]  # tall: bands left and right
    black = crop_image(image, bbox, 64)
    assert black[:, :20].max() == 0 and black[:, -20:].max() == 0
    assert (
        black[:, 28:36].min() == 200 and black[0, 32].min() == 200 and black[63, 32].min() == 200
    )
    assert crop_image(image, bbox, 64, context=True).min() == 200
    # outside the image is zero in both modes
    assert crop_image(image, [-50.0, -50.0, 50.0, 50.0], 64, context=True)[:10, :10].max() == 0


def test_pixel_transforms_are_photometric_only():
    for name in ("none", "standard", "domain_gap"):
        flat = [type(t).__name__ for t in pixel_transform(name).transforms]
        flat += [
            type(c).__name__
            for t in pixel_transform(name).transforms
            if isinstance(t, A.OneOf)
            for c in t.transforms
        ]
        assert "OpticalDistortion" not in flat and "Resize" not in flat
        assert "Normalize" in flat
    assert [type(t).__name__ for t in pixel_transform("none").transforms] == [
        "Normalize",
        "ToTensorV2",
    ]
    with pytest.raises(ValueError):
        pixel_transform("strong")


@pytest.fixture(scope="module")
def test_dataset(test_index):
    return BOPDataset(split="test", index=test_index, img_size=128)


def test_sample_fields_and_types(test_dataset):
    s = test_dataset[5]
    assert s["images"].shape == (3, 128, 128) and s["images"].dtype == torch.float32
    assert s["translations"].shape == (3,) and s["rotations"].shape == (3, 3)
    for k in ("translations", "rotations", "bboxes", "K", "img_wh", "diameter", "R_m2c", "t_m2c"):
        assert s[k].dtype == torch.float32, k
    idx = test_dataset.index
    x, y, w, h = idx["bbox_visib"][5].tolist()
    assert s["bboxes"].tolist() == [x, y, x + w, y + h]  # no jitter outside training
    assert s["obj_id"] == idx["obj_id"][5] and s["obj_idx"] == idx["obj_idx"][5]
    assert s["diameter"].item() == pytest.approx(idx["diameters"][idx["obj_idx"][5]], rel=1e-6)
    R = s["rotations"].double()
    torch.testing.assert_close(R @ R.T, torch.eye(3, dtype=torch.float64), atol=1e-5, rtol=0)


def test_eval_samples_are_deterministic(test_dataset):
    a, b = test_dataset[17], test_dataset[17]
    for k in ("images", "translations", "rotations", "bboxes"):
        assert torch.equal(a[k], b[k]), k


def test_object_centre_lands_where_U_says(test_dataset):
    """U_x, U_y locate the projected 3D centre relative to the bbox; check it in crop pixels."""
    for row in range(0, len(test_dataset), 97):
        s = test_dataset[row]
        K, t, bbox = s["K"].double(), s["t_m2c"].double(), s["bboxes"].double().tolist()
        u, v = (K @ t / t[2])[:2].tolist()
        M = crop_warp_matrix(bbox, 128)
        cu, cv = apply(M, u, v)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        Ux, Uy = s["translations"][:2].tolist()
        assert (cu - 63.5) / (M[0, 0] * w) == pytest.approx(Ux, abs=1e-5)
        assert (cv - 63.5) / (M[1, 1] * h) == pytest.approx(Uy, abs=1e-5)


def test_site_depth_label():
    index = build_index(BOP_ROOT, SPLITS["lmo"]["test"], LMO_OBJ_IDS, "lmo")
    ds = BOPDataset(split="test", index=index, img_size=64)
    s = ds[0]
    x1, y1, x2, y2 = s["bboxes"].tolist()
    f = 0.5 * (s["K"][0, 0] + s["K"][1, 1]).item()
    expected = s["t_m2c"][2].item() * 0.5 * ((x2 - x1) + (y2 - y1)) / (f * s["diameter"].item())
    assert s["translations"][2].item() == pytest.approx(expected, rel=1e-5)
    legacy = BOPDataset(split="test", index=index, img_size=64, depth_mode="legacy")[0]
    assert legacy["translations"][2].item() == pytest.approx(
        s["t_m2c"][2].item() * ((x2 - x1) + (y2 - y1)) / (2 * 480), rel=1e-5
    )


def test_training_jitter_stays_valid(pbr_index):
    ds = BOPDataset(split="train", index=pbr_index, img_size=64, pixel_aug="none")
    assert ds.augment
    np.random.seed(0)
    changed = 0
    for row in range(0, 400, 4):
        bbox = ds.crop_bbox(row)
        x, y, w, h = pbr_index["bbox_visib"][row].tolist()
        assert 0 <= bbox[0] < bbox[2] <= 640 and 0 <= bbox[1] < bbox[3] <= 480
        assert bbox[2] - bbox[0] >= 0.5 * w and bbox[3] - bbox[1] >= 0.5 * h
        changed += bbox != [x, y, x + w, y + h]
    assert 25 < changed < 75  # jitter applied about half of the time


def test_collated_batch_goes_through_unpack_batch(test_dataset):
    loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, num_workers=0)
    batch = unpack_batch(next(iter(loader)))
    assert batch["images"].shape == (4, 3, 128, 128)
    assert batch["K"].shape == (4, 3, 3) and batch["obj_idx"].dtype == torch.int64
    assert batch["diameter"].shape == (4,) and batch["img_wh"].shape == (4, 2)
