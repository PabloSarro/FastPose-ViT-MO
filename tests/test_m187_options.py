"""M18.7 training options: rotation loss weight, occlusion rectangle settings and pasted occluders.

The defaults must keep the earlier behaviour (the 2x densex5 runs are the baseline). The dataset
checks use real LM PBR files and are skipped when the BOP tree is not available.
"""

import os

import albumentations as A
import numpy as np
import pytest
import torch

from src.bop_dataset import (
    LMO_OBJ_IDS,
    BOPDataset,
    Source,
    build_index,
    default_bop_root,
    pixel_transform,
)
from src.engine import TrainingConfig, train_epoch
from src.train_args import build_parser
from tests.test_dense_engine import TinyDense, batch

BOP_ROOT = default_bop_root()
needs_bop = pytest.mark.skipif(
    not os.path.exists(os.path.join(BOP_ROOT, "lmo", "test_targets_bop19.json")),
    reason=f"BOP tree not found at {BOP_ROOT}",
)
SIZE, DENSE = 128, 64


def dropout_of(transform):
    found = [t for t in transform.transforms if isinstance(t, A.CoarseDropout)]
    assert len(found) == 1
    return found[0]


def test_defaults_keep_the_earlier_recipe():
    args = build_parser().parse_args(["--dataset", "LMO"])
    assert (args.rotation_loss_weight, args.bop_cutout_p) == (1.0, 0.0)
    assert (args.bop_occlusion_p, args.bop_occlusion_max_holes) == (0.5, 4)
    assert args.bop_occlusion_max_size == 0.3
    d = dropout_of(pixel_transform("domain_gap", occlusion=True))
    assert d.p == 0.5 and tuple(d.num_holes_range) == (1, 4)
    assert tuple(d.hole_height_range) == tuple(d.hole_width_range) == (0.1, 0.3)


def test_occlusion_settings_reach_the_transform():
    t = pixel_transform(
        "none", occlusion=True, occlusion_p=0.8, occlusion_max_holes=6, occlusion_max_size=0.4
    )
    d = dropout_of(t)
    assert d.p == 0.8 and tuple(d.num_holes_range) == (1, 6)
    assert tuple(d.hole_height_range) == tuple(d.hole_width_range) == (0.1, 0.4)


@pytest.mark.parametrize("weight", [0.5, 2.0])
def test_rotation_loss_weight_scales_only_the_total(weight):
    def run(w):
        torch.manual_seed(0)
        model = TinyDense()
        config = TrainingConfig(
            rotation_format="matrix",
            rotation_loss_type="6d_relative_frobenius",
            merge_outputs=True,
            max_grad_norm=0.0,
            use_per_iteration_scheduling=False,
            rotation_loss_weight=w,
        )
        return train_epoch(
            model,
            [batch()],
            torch.optim.SGD(model.parameters(), lr=0.0),
            lambda p, t, b: (p - t).abs().mean(),
            lambda p, t: (p - t).abs().mean(),
            torch.device("cpu"),
            0,
            None,
            config,
        )

    total1, trans1, rot1 = run(1.0)
    total, trans, rot = run(weight)
    assert (trans, rot) == pytest.approx((trans1, rot1))  # the reported losses are unweighted
    assert total - total1 == pytest.approx((weight - 1.0) * rot1, rel=1e-5)


@pytest.fixture(scope="module")
def index():
    return build_index(BOP_ROOT, (Source("lm", "train_pbr", (0, 1)),), LMO_OBJ_IDS, "lmo")


def make(index, cutout_p):
    return BOPDataset(
        dataset="lmo",
        split="train",
        index=index,
        img_size=SIZE,
        dense_size=DENSE,
        augment=True,
        bbox_jitter_percent=0.0,
        pixel_aug="none",
        cutout_p=cutout_p,
    )


@needs_bop
def test_covered_dense_samples_the_same_crop_pixels(index):
    ds = make(index, 0.0)
    bbox = ds.crop_bbox(0)
    covered = np.zeros((SIZE, SIZE), dtype=bool)
    covered[:, : SIZE // 2] = True  # left half of the crop
    small = ds.covered_dense(covered, bbox)
    assert small.shape == (DENSE, DENSE)
    assert small[:, : DENSE // 2 - 1].all() and not small[:, DENSE // 2 + 1 :].any()


@needs_bop
def test_pasted_occluder_changes_only_its_pixels_and_leaves_the_targets(index):
    base_ds, cut_ds = make(index, 0.0), make(index, 1.0)
    rows = np.random.default_rng(0).choice(len(base_ds), 24, replace=False)
    changed_images = hidden_pixels = 0
    for k, row in enumerate(int(r) for r in rows):
        base = base_ds[row]
        np.random.seed(k)
        cut = cut_ds[row]
        # replay the same random draws to know which pixels were pasted
        np.random.seed(k)
        np.random.random()  # the probability check of __getitem__
        bbox = cut_ds.crop_bbox(row)
        crop = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
        _, covered = cut_ds.paste_occluder(row, crop, bbox)
        diff = (base["images"] - cut["images"]).abs().sum(0).numpy() > 1e-6
        assert not (diff & ~covered).any()  # nothing outside the occluder changed
        keep = torch.from_numpy(~cut_ds.covered_dense(covered, bbox))
        torch.testing.assert_close(cut["mask"], base["mask"] * keep)
        torch.testing.assert_close(cut["xyz_valid"], base["xyz_valid"] * keep)
        torch.testing.assert_close(cut["xyz"], base["xyz"] * keep)
        for key in ("translations", "rotations", "bboxes", "R_m2c", "t_m2c"):
            torch.testing.assert_close(cut[key], base[key])  # pose labels unchanged
        changed_images += int(diff.any())
        hidden_pixels += int((base["mask"] - cut["mask"]).sum())
    assert changed_images >= 20  # the occluder really lands on the crop
    assert hidden_pixels > 0  # and hides part of some objects


def make_occluded(index, targets, occlusion=True):
    return BOPDataset(
        dataset="lmo",
        split="train",
        index=index,
        img_size=SIZE,
        dense_size=DENSE,
        augment=True,
        bbox_jitter_percent=0.0,
        pixel_aug="none",
        occlusion_aug=occlusion,
        occlusion_p=1.0,
        occlusion_targets=targets,
    )


@needs_bop
@pytest.mark.parametrize("targets", ["hide", "hide_mask"])
def test_targets_follow_the_occlusion_rectangles(index, targets):
    clean_ds = make_occluded(index, "keep", occlusion=False)
    ds = make_occluded(index, targets)
    removed = 0
    for k, row in enumerate(int(r) for r in np.random.default_rng(1).choice(len(ds), 24, replace=False)):
        clean = clean_ds[row]
        ds.transform.set_random_seed(k)
        occ = ds[row]
        # with no other augmentation, the pixels that changed are the erased ones
        changed = ((occ["images"] - clean["images"]).abs().sum(0) > 1e-6).numpy()
        assert changed.any()
        keep = torch.from_numpy(~ds.covered_dense(changed, ds.crop_bbox(row)))
        torch.testing.assert_close(occ["mask"], clean["mask"] * keep)
        if targets == "hide":
            torch.testing.assert_close(occ["xyz_valid"], clean["xyz_valid"] * keep)
            torch.testing.assert_close(occ["xyz"], clean["xyz"] * keep)
        else:  # XYZ stays supervised under the rectangles
            torch.testing.assert_close(occ["xyz_valid"], clean["xyz_valid"])
            torch.testing.assert_close(occ["xyz"], clean["xyz"])
        removed += int((clean["mask"] - occ["mask"]).sum())
    assert removed > 0


@needs_bop
def test_keep_leaves_the_targets_under_the_rectangles(index):
    clean, occ = make_occluded(index, "keep", occlusion=False)[3], make_occluded(index, "keep")[3]
    assert (occ["images"] != clean["images"]).any()
    for key in ("mask", "xyz", "xyz_valid"):
        torch.testing.assert_close(occ[key], clean[key])


@needs_bop
def test_cutout_off_without_augmentation(index):
    ds = BOPDataset(dataset="lmo", split="val", index=index, img_size=SIZE, cutout_p=1.0)
    assert ds.cutout_p == 0.0
