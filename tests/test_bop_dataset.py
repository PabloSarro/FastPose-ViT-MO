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
