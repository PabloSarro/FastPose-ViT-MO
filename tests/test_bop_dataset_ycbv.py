"""YCB-V index (M8), on the real files: the same checks as for LM-O (tests/test_bop_dataset.py).

- Test rows: exactly the 4,123 YCB-V BOP19 targets.
- Training rows: the instances with visib_fract >= 0.1 and a valid bbox of one PBR scene.
- Row contents (pose, K, bboxes, image path) match the raw files.
- The GT pose, K and bbox_obj agree (reprojected models), on PBR and on the real test images.
- Labels round-trip through the engine's inverse transforms.
- The image size comes from camera_uw.json / camera_cmu.json (YCB-V has no camera.json).

Skipped when the YCB-V tree (bash scripts/make_bop_tree.sh ycbv) is not available.
"""

import json
import os

import numpy as np
import pytest
from bop_toolkit_lib import inout

from src.bop_dataset import (
    SPLITS,
    YCBV_OBJ_IDS,
    BOPDataset,
    Source,
    _image_size,
    bop_bbox_to_xyxy,
    build_index,
    default_bop_root,
    image_path,
)
from tests.test_bop_dataset import (
    ROUND_TRIP_TOL,
    iou,
    projected_box,
    reprojection_sample,
    round_trip_errors,
)

BOP_ROOT = default_bop_root()
pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(BOP_ROOT, "ycbv", "test_targets_bop19.json")),
    reason=f"YCB-V tree not found at {BOP_ROOT}",
)
PBR_SCENE = os.path.join(BOP_ROOT, "ycbv", "train_pbr", "000000")


def load(path):
    with open(path) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def test_index():
    return build_index(BOP_ROOT, SPLITS["ycbv"]["test"], YCBV_OBJ_IDS, "ycbv")


@pytest.fixture(scope="module")
def pbr_index():
    return build_index(
        BOP_ROOT, (Source("ycbv", "train_pbr", (0, 17, 33, 47)),), YCBV_OBJ_IDS, "ycbv"
    )


def test_image_size_from_the_two_camera_files():
    assert not os.path.exists(os.path.join(BOP_ROOT, "ycbv", "camera.json"))
    assert _image_size(os.path.join(BOP_ROOT, "ycbv")) == (640, 480)


def test_splits_use_pbr_only_and_keep_test_out():
    cfg = SPLITS["ycbv"]
    assert cfg["obj_ids"] == tuple(range(1, 22))
    assert cfg["train"][0].scenes == tuple(range(48))
    assert cfg["val"] == cfg["val_pbr"] == (Source("ycbv", "train_pbr", (48, 49)),)
    for split in ("train", "val", "val_pbr"):
        assert all(s.split == "train_pbr" for s in cfg[split])


def test_test_rows_are_exactly_the_bop19_targets(test_index):
    targets = load(os.path.join(BOP_ROOT, "ycbv", "test_targets_bop19.json"))
    assert len(test_index["source"]) == sum(t["inst_count"] for t in targets) == 4123
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


def test_pbr_rows_match_raw_annotations():
    index = build_index(BOP_ROOT, (Source("ycbv", "train_pbr", (0,)),), YCBV_OBJ_IDS, "ycbv")
    gt = load(os.path.join(PBR_SCENE, "scene_gt.json"))
    info = load(os.path.join(PBR_SCENE, "scene_gt_info.json"))
    expected = [
        (int(im), i)
        for im in gt
        for i, n in enumerate(info[im])
        if n["visib_fract"] >= 0.1 and min(n["bbox_visib"][2:]) > 0
    ]
    rows = sorted(zip(index["im_id"].tolist(), index["gt_id"].tolist()))
    assert rows == sorted(expected) and len(rows) > 10000


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
        assert gt["obj_id"] == index["obj_id"][row] == YCBV_OBJ_IDS[index["obj_idx"][row]]
        np.testing.assert_array_equal(index["R"][row], np.reshape(gt["cam_R_m2c"], (3, 3)))
        np.testing.assert_array_equal(index["t"][row], gt["cam_t_m2c"])
        np.testing.assert_array_equal(index["K"][row], np.reshape(cam["cam_K"], (3, 3)))
        np.testing.assert_array_equal(index["bbox_obj"][row], info["bbox_obj"])
        np.testing.assert_array_equal(index["bbox_visib"][row], info["bbox_visib"])
        assert os.path.exists(image_path(index, row))
    assert image_path(index, 0).endswith(".jpg" if which == "pbr" else ".png")
    np.testing.assert_array_equal(index["img_wh"], np.tile([640, 480], (len(index["source"]), 1)))


def test_object_table(test_index):
    np.testing.assert_array_equal(test_index["obj_ids"], YCBV_OBJ_IDS)
    # BOP19 symmetric objects of YCB-V: continuous 13 (bowl), 18 (large marker); discrete 1, 16,
    # 19, 20, 21
    sym = (np.flatnonzero(test_index["symmetric"]) + 1).tolist()
    assert sym == [1, 13, 16, 18, 19, 20, 21]
    assert test_index["diameters"].min() > 50 and test_index["diameters"].max() < 300  # mm


@pytest.mark.parametrize("which", ["pbr", "test"])
def test_reprojected_models_match_bbox_obj(which, pbr_index, test_index):
    index = pbr_index if which == "pbr" else test_index
    good, edges = [], []
    for row in reprojection_sample(index):
        obj = int(index["obj_id"][row])
        points = inout.load_ply(
            os.path.join(BOP_ROOT, "ycbv", "models_eval", f"obj_{obj:06d}.ply")
        )["pts"]
        R, t, K = index["R"][row], index["t"][row], index["K"][row]
        box = bop_bbox_to_xyxy(index["bbox_obj"][row])
        proj = projected_box(points, R, t, K)
        good.append(iou(proj, box))
        edges.append(proj - box)
    good, edges = np.array(good), np.array(edges)
    assert good.min() >= 0.9, good.min()
    assert np.abs(edges.mean(axis=0)).max() < 0.5, edges.mean(axis=0)


def test_label_round_trip_ycbv_test(test_index):
    """All 4,123 YCB-V test targets."""
    ds = BOPDataset(dataset="ycbv", split="test", index=test_index, img_size=32)
    t_err, R_err = round_trip_errors(ds, np.arange(len(ds)), "site")
    assert t_err < ROUND_TRIP_TOL and R_err < ROUND_TRIP_TOL, (t_err, R_err)


def test_label_round_trip_training_crops(pbr_index):
    np.random.seed(0)
    ds = BOPDataset(
        dataset="ycbv",
        index=pbr_index,
        img_size=32,
        augment=True,
        pixel_aug="none",
        bbox_jitter_prob=1.0,
    )
    rows = np.random.default_rng(1).choice(len(ds), 256, replace=False)
    t_err, R_err = round_trip_errors(ds, rows, "site")
    assert t_err < ROUND_TRIP_TOL and R_err < ROUND_TRIP_TOL, (t_err, R_err)
