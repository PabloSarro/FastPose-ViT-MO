# Tests of src/bop_evaluate.py's per-instance errors, including the symmetric objects (ADD-S,
# symmetric MSSD/MSPD and rotation error). Need the BOP tree (skipped without it).

import os

import numpy as np
import pytest

from src.bop_dataset import default_bop_root, load_index, rows_for_objects, select_rows
from src.bop_evaluate import instance_errors

pytest.importorskip("bop_toolkit_lib")
from bop_toolkit_lib import inout, misc  # noqa: E402

BOP_ROOT = default_bop_root()
pytestmark = pytest.mark.skipif(
    not os.path.isdir(os.path.join(BOP_ROOT, "lmo")), reason="BOP tree not available"
)


def _object(obj_id, n=5):
    index = load_index("lmo", "test")
    return select_rows(index, rows_for_objects(index, [obj_id])[:n])


def _syms(obj_id):
    info = inout.load_json(
        os.path.join(BOP_ROOT, "lmo", "models_eval", "models_info.json"), keys_to_int=True
    )
    return misc.get_symmetry_transformations(info[obj_id], max_sym_disc_step=0.01)


@pytest.mark.parametrize("obj_id", [10, 11])
def test_symmetric_pose_has_zero_error(obj_id):
    sub = _object(obj_id)
    sym = [s for s in _syms(obj_id) if not np.allclose(s["R"], np.eye(3))][0]
    # x_c = R_gt (S x + t_s) + t_gt: the pose R_gt S, R_gt t_s + t_gt looks identical
    R = sub["R"].astype(np.float64) @ sym["R"]
    t = (sub["R"].astype(np.float64) @ sym["t"])[..., 0] + sub["t"].astype(np.float64)
    errs = instance_errors(sub, {"R": R, "t": t}, obj_id, BOP_ROOT)
    # GT and symmetry matrices are orthonormal to ~1e-4 only (arccos near 1 -> ~0.06 deg), and
    # ADD-S matches vertices to their nearest neighbour (models_eval spacing ~2 mm)
    assert errs["rot_deg"].max() < 0.1
    assert errs["add"].max() < 0.02 * errs["diameter"]
    assert errs["mssd"].max() < 1e-3 and errs["mspd"].max() < 1e-3
    plain = instance_errors(sub, {"R": sub["R"], "t": sub["t"]}, obj_id, BOP_ROOT)
    assert plain["add"].max() < 1e-3 and plain["rot_deg"].max() < 0.1


def test_asymmetric_object_uses_plain_add():
    sub = _object(8)
    R = sub["R"].astype(np.float64)
    flip = np.diag([-1.0, -1.0, 1.0])  # 180 deg about z: a large error for the driller
    errs = instance_errors(sub, {"R": R @ flip, "t": sub["t"]}, 8, BOP_ROOT)
    assert errs["rot_deg"].min() == pytest.approx(180.0, abs=1e-3)
    assert errs["add"].min() > 0.1 * errs["diameter"]


def test_symmetric_id_table_matches_models_info():
    from src.bop_evaluate import LMO_SYMMETRIC_IDS

    index = load_index("lmo", "test")
    assert tuple(index["obj_ids"][index["symmetric"]].tolist()) == LMO_SYMMETRIC_IDS


def test_per_object_table_means():
    from src.bop_evaluate import TABLE_COLUMNS, per_object_table

    keys = [k for k, _, _ in TABLE_COLUMNS if k != "n"]
    per_obj = {"8": {"n": 100, **{k: 1.0 for k in keys}}, "10": {"n": 300, **{k: 0.0 for k in keys}}}
    table, means = per_object_table(per_obj, symmetric_ids={10})
    assert means["mean over objects"]["ADD@0.1d"] == pytest.approx(0.5)
    assert means["mean over instances"]["ADD@0.1d"] == pytest.approx(0.25)
    assert means["mean over objects"]["n"] == 400
    assert "| 10 (sym.) | 300 |" in table and len(table.strip().splitlines()) == 2 + 2 + 2
