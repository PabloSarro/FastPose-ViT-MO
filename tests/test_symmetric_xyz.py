"""Symmetric XYZ targets (M18.6a): the target of the symmetry the rotation loss picks.

Reference: the dataset's own back-projection run with the symmetric pose (R·S_k, t + R·s_k)
instead of the GT pose (same depth, same mask). SymmetricRotation6DLoss.symmetric_xyz must give
the same normalised XYZ, and choose_symmetry must return k for a prediction equal to the
symmetric pose k. LM-O eggbox / glue (no offsets) and a YCB-V clamp (shifted symmetry).
"""

import os

import numpy as np
import pytest
import torch

from src.bop_dataset import (
    SPLITS,
    BOPDataset,
    build_index,
    default_bop_root,
    object_extents,
    select_rows,
    symmetry_table,
)
from src.losses import SymmetricRotation6DLoss
from src.utils import rotation_matrix_to_6d

BOP_ROOT = default_bop_root()
pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(BOP_ROOT, "ycbv", "models_eval", "models_info.json")),
    reason="BOP tree not available",
)


def setup(dataset_name, obj_id, n_rows=6):
    cfg = SPLITS[dataset_name]
    obj_ids = list(cfg["obj_ids"])
    index = build_index(BOP_ROOT, cfg["test"], obj_ids, cfg["models_dataset"])
    rows = np.flatnonzero(index["obj_id"] == obj_id)[:: max(1, 4 if dataset_name == "lmo" else 40)][:n_rows]
    ds = BOPDataset(
        dataset=dataset_name, split="test", index=select_rows(index, rows), img_size=64,
        dense_size=32, augment=False,
    )
    table, offsets = symmetry_table(BOP_ROOT, cfg["models_dataset"], obj_ids)
    crit = SymmetricRotation6DLoss(table, offsets)
    centre, half = object_extents(BOP_ROOT, cfg["models_dataset"], obj_ids)
    return ds, crit, torch.as_tensor(centre, dtype=torch.float32), torch.as_tensor(half, dtype=torch.float32)


def symmetric_pose_reference(ds, row, R_sym, t_sym):
    """The dataset's targets for the same pixels with the pose replaced (R_sym, t_sym)."""
    index = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in ds.index.items()}
    index["R"][row], index["t"][row] = R_sym, t_sym
    other = BOPDataset(
        dataset="lmo" if ds.xyz_centre.shape[0] == 8 else "ycbv", split="test", index=index,
        img_size=64, dense_size=32, augment=False,
    )
    return other.dense_targets(row, ds.crop_bbox(row))


@pytest.mark.parametrize("dataset_name,obj_id", [("lmo", 10), ("lmo", 11), ("ycbv", 19), ("ycbv", 13)])
def test_symmetric_xyz_matches_the_symmetric_pose(dataset_name, obj_id):
    ds, crit, centre, half = setup(dataset_name, obj_id)
    obj_ids = list(SPLITS[dataset_name]["obj_ids"])
    o = obj_ids.index(obj_id)
    n_sym = int((torch.linalg.matrix_norm(crit.sym_table[o] - torch.eye(3), dim=(-2, -1)) > 1e-4).sum()) + 1
    assert n_sym > 1, "the object must have a non-trivial symmetry"
    checked = 0
    for row in range(len(ds)):
        s = ds[row]
        R, t = s["R_m2c"].double(), s["t_m2c"].double()
        batch = lambda x: x.unsqueeze(0)  # noqa: E731
        obj = torch.tensor([o])
        pose = (batch(R), batch(t), batch(s["K"]), batch(s["bboxes"]), batch(s["img_wh"]), batch(s["diameter"]))
        U, rotations = crit.symmetric_labels(obj, *pose)  # allocentric rotations of every sym pose
        for k in range(1, crit.sym_table.shape[1]):
            S, off = crit.sym_table[o, k].double(), crit.sym_offsets[o, k].double()
            if torch.linalg.matrix_norm(S - torch.eye(3, dtype=torch.float64)) < 1e-4:
                continue  # padding identity
            # the network predicts exactly symmetric pose k: the loss must pick k
            pred6d = rotation_matrix_to_6d(rotations[:, k].float())
            kk = crit.choose_symmetry(pred6d, s["rotations"].unsqueeze(0), obj, pose)
            assert kk.item() == k or torch.allclose(rotations[0, kk.item()], rotations[0, k], atol=1e-5)
            k_used = kk.item()
            target = crit.symmetric_xyz(
                s["xyz"].unsqueeze(0), torch.tensor([k_used]), obj, centre, half
            )[0].numpy()
            R_sym = (R @ crit.sym_table[o, k_used].double()).numpy()
            t_sym = (t + R @ crit.sym_offsets[o, k_used].double()).numpy()
            _, ref, ref_valid = symmetric_pose_reference(ds, row, R_sym, t_sym)
            both = (s["xyz_valid"][0].numpy() > 0) & ref_valid
            assert both.sum() > 20
            np.testing.assert_allclose(target[:, both], ref[:, both], atol=2e-3)
            # control: the untransformed GT target is clearly different (the test has power)
            # (a discretised continuous symmetry has small steps: only for a clear rotation)
            if torch.linalg.matrix_norm(S - torch.eye(3, dtype=torch.float64)) > 1.0:
                assert np.abs(s["xyz"].numpy()[:, both] - ref[:, both]).max() > 0.05
            checked += 1
    assert checked > 0


def test_asymmetric_object_and_identity_are_unchanged():
    ds, crit, centre, half = setup("lmo", 5)  # an asymmetric LM-O object
    s = ds[0]
    obj = torch.tensor([1])
    out = crit.symmetric_xyz(s["xyz"].unsqueeze(0), torch.tensor([0]), obj, centre, half)[0]
    torch.testing.assert_close(out, s["xyz"], atol=1e-4, rtol=0)
    # any k of an asymmetric object is the identity as well
    out = crit.symmetric_xyz(s["xyz"].unsqueeze(0), torch.tensor([crit.sym_table.shape[1] - 1]), obj, centre, half)[0]
    torch.testing.assert_close(out, s["xyz"], atol=1e-4, rtol=0)
