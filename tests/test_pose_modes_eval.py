"""M18.5: bop_evaluate.predict / estimate_image with the dense pose modes.

An oracle replaces the network: it returns the dataset's own labels (pose output) and dense
targets, in the batch order, so each pose mode must give back the GT poses of the rows. That
checks everything around the network: crop boxes, K, object extents, row order, the DR pose
decoding and the refiner. Plus argument checks and a random-weights smoke run of a real model.
"""

import argparse
import os

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

import src.bop_evaluate as be
from src.bop_dataset import (
    SPLITS,
    BOPDataset,
    build_index,
    default_bop_root,
    nearest_rotation,
    select_rows,
)
from src.pnp import POSE_MODES
from src.utils import rotation_matrix_to_6d

BOP_ROOT = default_bop_root()
pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(BOP_ROOT, "lmo", "test_targets_bop19.json")),
    reason="BOP tree not available",
)
N = 24


@pytest.fixture(scope="module")
def dataset():
    index = build_index(BOP_ROOT, SPLITS["lmo"]["test"], list(SPLITS["lmo"]["obj_ids"]), "lmo")
    rows = np.arange(0, len(index["obj_id"]), len(index["obj_id"]) // N)[:N]
    return BOPDataset(
        dataset="lmo", split="test", index=select_rows(index, rows), img_size=64,
        dense_size=64, augment=False,
    )


def args_for(mode):
    return argparse.Namespace(
        batch_size=8, num_workers=0, depth_mode="site", dataset="lmo", pose_mode=mode,
        crop_scale=1.0, no_rotation_compensation=False,
    )


def oracle(dataset, monkeypatch, batch_size=8):
    """forward_dense returning the labels and targets of the batches, in order."""
    batches = iter(DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0))

    def forward_dense(model, images, obj_idx, merge_outputs):
        batch = next(batches)
        logits = (batch["xyz_valid"] * 2 - 1) * 10  # confident only where the XYZ is valid
        dense = torch.cat([logits, batch["xyz"]], dim=1).to(images.device)
        rot6d = rotation_matrix_to_6d(batch["rotations"])
        return batch["translations"].to(images.device), rot6d.to(images.device), dense

    monkeypatch.setattr(be, "forward_dense", forward_dense)
    monkeypatch.setattr(be, "forward_model", lambda m, i, o, merge_outputs: forward_dense(m, i, o, 1)[:2])


def errors(pred, dataset):
    R_gt = np.stack([nearest_rotation(R.astype(np.float64)) for R in dataset.index["R"]])
    t_gt = dataset.index["t"]
    cos = (np.einsum("nij,nij->n", pred["R"], R_gt) - 1) / 2
    return np.degrees(np.arccos(np.clip(cos, -1, 1))), np.linalg.norm(pred["t"] - t_gt, axis=1)


@pytest.mark.parametrize("mode", POSE_MODES)
def test_every_mode_returns_the_gt_with_an_oracle_network(mode, dataset, monkeypatch):
    oracle(dataset, monkeypatch)
    pred = be.predict(None, dataset, args_for(mode), torch.device("cpu"))
    assert pred["R"].shape == (len(dataset), 3, 3) and pred["fallback"].shape == (len(dataset),)
    rot, trans = errors(pred, dataset)
    assert rot.max() < 0.1 and trans.max() < 1.0, (mode, rot.max(), trans.max())
    assert not pred["fallback"].any()


def test_default_mode_has_no_refiner():
    assert be.make_refiner(argparse.Namespace(pose_mode="dr", dataset="lmo"), None) is None
    assert be.make_refiner(argparse.Namespace(dataset="lmo"), None) is None


def test_estimate_image_with_a_refiner(dataset, monkeypatch):
    """The timed per-image path returns the refiner's poses for the rows of one image."""
    # one image = its rows; the oracle serves the rows of that image in order
    index = dataset.index
    key = (int(index["scene_id"][0]), int(index["im_id"][0]))
    rows = [r for r in range(len(dataset)) if (int(index["scene_id"][r]), int(index["im_id"][r])) == key]
    sub = BOPDataset(
        dataset="lmo", split="test", index=select_rows(index, np.array(rows)), img_size=64,
        dense_size=64, augment=False,
    )
    oracle(sub, monkeypatch, batch_size=len(rows))
    args = args_for("pnp")
    refiner = be.make_refiner(args, sub)
    transformer = be.pose_transformer(sub, args)
    image = sub.load_image(0)
    R, t = be.estimate_image(None, sub, transformer, list(range(len(rows))), image, torch.device("cpu"), refiner)
    pred = {"R": R, "t": t}
    rot, trans = errors(pred, sub)
    assert rot.max() < 0.1 and trans.max() < 1.0


def test_random_dense_model_runs_every_mode(dataset):
    """A real (random-weight) dense model through predict: shapes, and a poor model falls back
    or gives finite poses, never crashes."""
    from src.models import FastPoseViT

    torch.manual_seed(0)
    model = FastPoseViT(
        "vit_b_16_no_weights", 0, 0, 3, 6, merge_outputs=True, no_mlp=True,
        num_objects=8, object_conditioning="token", dense_heads=True, dense_size=64,
    ).eval()
    small = BOPDataset(dataset="lmo", split="test", index=select_rows(dataset.index, np.arange(6)),
                       img_size=224, augment=False)
    for mode in POSE_MODES:
        pred = be.predict(model, small, args_for(mode), torch.device("cpu"))
        assert np.isfinite(pred["R"]).all() and np.isfinite(pred["t"]).all()
        assert pred["fallback"].dtype == bool


def test_cli_rejects_pose_modes_without_dense_heads(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "sys.argv",
        ["bop_evaluate", "--out_dir", str(tmp_path), "--model_weights", "x.pt", "--pose_mode", "pnp"],
    )
    with pytest.raises(SystemExit):
        be.main()
