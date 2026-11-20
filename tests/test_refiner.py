"""Refiner (M19.2): update rule, symmetric point loss, network wiring, dataset samples."""

import os

import numpy as np
import pytest
import torch

from src.bop_dataset import (
    LMO_OBJ_IDS,
    SPLITS,
    Source,
    build_index,
    default_bop_root,
    object_extents,
    symmetry_table,
)
from src.refiner import (
    IN_CHANNELS,
    NOISE_LEVELS,
    RefineDataset,
    RefineLoss,
    RefinerNet,
    apply_update,
    exact_update,
    make_input,
    perturb_pose,
    pose_crop,
    refine_instances,
    rotation_about,
    rotation_from_6d,
    rotation_to_6d,
)
from src.render import PointRenderer, load_render_assets

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def random_poses(n, seed=0):
    rng = np.random.default_rng(seed)
    R = np.stack([rotation_about(rng.normal(size=3), rng.uniform(0, 180)) for _ in range(n)])
    t = np.stack([[rng.uniform(-100, 100), rng.uniform(-100, 100), rng.uniform(500, 1200)] for _ in range(n)])
    return torch.tensor(R, dtype=torch.float64), torch.tensor(t, dtype=torch.float64)


def test_6d_round_trip():
    R, _ = random_poses(20)
    torch.testing.assert_close(rotation_from_6d(rotation_to_6d(R)), R)
    # any 6D vector gives a proper rotation
    M = rotation_from_6d(torch.randn(50, 6, dtype=torch.float64))
    torch.testing.assert_close(M @ M.transpose(1, 2), torch.eye(3, dtype=torch.float64).expand(50, 3, 3), atol=1e-9, rtol=0)
    assert torch.allclose(torch.linalg.det(M), torch.ones(50, dtype=torch.float64))


def test_zero_output_is_the_identity_update_and_exact_update_reaches_the_target():
    R, t = random_poses(16, 1)
    Rt, tt = random_poses(16, 2)
    K = torch.tensor([[900.0, 0, 128], [0, 900.0, 128], [0, 0, 1]], dtype=torch.float64).expand(16, 3, 3)
    R0, t0 = apply_update(R, t, K, torch.zeros(16, 9, dtype=torch.float64), 256)
    torch.testing.assert_close(R0, R)
    torch.testing.assert_close(t0, t)
    out = exact_update(R, t, Rt, tt, K, 256)
    R2, t2 = apply_update(R, t, K, out, 256)
    torch.testing.assert_close(R2, Rt, atol=1e-9, rtol=0)
    torch.testing.assert_close(t2, tt, atol=1e-6, rtol=0)


BOP_ROOT = default_bop_root()
HAS_BOP = os.path.exists(os.path.join(BOP_ROOT, "lm", "models", "obj_000001.ply"))
needs_bop = pytest.mark.skipif(not HAS_BOP, reason="no BOP tree")


@pytest.fixture(scope="module")
def loss_setup():
    assets = load_render_assets(BOP_ROOT, "lmo", LMO_OBJ_IDS, 8192)
    from src.bop_dataset import load_models_info

    diam = load_models_info(BOP_ROOT, "lmo", LMO_OBJ_IDS)["diameter"]
    table, offsets = symmetry_table(BOP_ROOT, "lmo", LMO_OBJ_IDS)
    return RefineLoss(assets["points"][:, :512], diam, table, offsets)


@needs_bop
def test_loss_zero_at_gt_and_at_symmetric_poses_and_grows_with_error(loss_setup):
    loss = loss_setup
    R, t = random_poses(8, 3)
    R, t = R.float(), t.float()
    obj = torch.tensor([5, 6, 5, 6, 0, 1, 2, 3])  # eggbox (5) and glue (6) are symmetric
    z = loss(R, t, R, t, obj)
    assert max(z[k].item() for k in ("total", "full", "rot", "trans")) < 1e-4
    # a symmetric object: the pose turned by one of its symmetries is as good as the GT
    S = loss.sym_R[5, 1]
    assert (S - torch.eye(3)).abs().sum() > 0.5, "eggbox must have a non-trivial symmetry"
    sym = loss(R[:1] @ S, t[:1] + R[:1, :, :] @ loss.sym_t[5, 1], R[:1], t[:1], obj[:1])
    assert sym["total"].item() < 1e-4
    # an asymmetric object (index 0) is not
    wrong = loss(R[4:5] @ S, t[4:5], R[4:5], t[4:5], obj[4:5])
    assert wrong["rot"].item() > 1e-2
    # monotone in the error
    errs = []
    for deg in (2, 10, 30):
        Rn = torch.tensor(rotation_about(np.array([0.0, 0.0, 1.0]), deg), dtype=torch.float32)[None] @ R[4:5]
        errs.append(loss(Rn, t[4:5], R[4:5], t[4:5], obj[4:5])["total"].item())
    assert errs[0] < errs[1] < errs[2]
    ts = [loss(R[4:5], t[4:5] + torch.tensor([[0.0, 0.0, d]]), R[4:5], t[4:5], obj[4:5])["trans"].item() for d in (5, 20)]
    assert 0 < ts[0] < ts[1]


@needs_bop
def test_loss_gradient(loss_setup):
    R, t = random_poses(4, 4)
    out = torch.zeros(4, 9, requires_grad=True)
    K = torch.tensor([[900.0, 0, 128], [0, 900.0, 128], [0, 0, 1]]).expand(4, 3, 3)
    Rg, tg = random_poses(4, 5)
    R2, t2 = apply_update(R.float(), t.float(), K, out, 256)
    l = loss_setup(R2, t2, Rg.float(), tg.float(), torch.tensor([0, 1, 2, 3]))
    l["total"].backward()
    assert out.grad.abs().sum() > 0 and torch.isfinite(out.grad).all()


def test_network_shape_identity_start_and_input_channels():
    net = RefinerNet("vit_b_16_no_weights", num_objects=3).eval()
    assert net.net.backbone.conv_proj.in_channels == IN_CHANNELS
    x = torch.randn(2, IN_CHANNELS, 224, 224)
    with torch.no_grad():
        out = net(x, torch.tensor([0, 2]))
    assert out.shape == (2, 9) and out.abs().max().item() == 0.0  # zero head: the identity
    image, render = torch.randn(2, 3, 64, 64), torch.rand(2, 7, 64, 64)
    assert make_input(image, render).shape == (2, IN_CHANNELS, 64, 64)
    # the background of the render stays at 0 in the colour channels
    render[:, 6] = 0
    assert make_input(image, render)[:, 3:6].abs().max().item() == 0.0


def test_network_conv_copies_the_pretrained_filters():
    try:
        net = RefinerNet("vit_b_16_256", num_objects=2)
    except Exception as e:  # weights not downloadable here
        pytest.skip(f"pretrained weights unavailable: {e}")
    w = net.net.backbone.conv_proj.weight
    torch.testing.assert_close(w[:, 0:3], w[:, 3:6])
    assert w[:, 6:].abs().max().item() == 0.0 and w[:, 0:3].abs().sum() > 0


def test_perturbation_statistics():
    rng = np.random.default_rng(0)
    K = np.array([[572.0, 0, 320.0], [0, 573.0, 240.0], [0, 0, 1.0]])
    R, t = np.eye(3), np.array([30.0, -20.0, 800.0])
    angles, dz = [], []
    for _ in range(3000):
        R2, t2 = perturb_pose(R, t, K, 180.0, rng)
        angles.append(np.degrees(np.arccos(np.clip((np.trace(R2) - 1) / 2, -1, 1))))
        dz.append(np.log(t2[2] / t[2]))
    assert 3.0 < np.median(angles) < 8.0  # a mixture around our DR errors (median ~4.6 deg)
    assert np.percentile(angles, 95) > 15  # and larger ones
    assert abs(np.mean(dz)) < 0.01 and 0.02 < np.std(dz) < 0.08
    assert sum(lv["p"] for lv in NOISE_LEVELS) == pytest.approx(1.0)


@needs_bop
def test_dataset_sample_and_crop_contains_the_object():
    assets = load_render_assets(BOP_ROOT, "lmo", LMO_OBJ_IDS, 4096)
    index = build_index(BOP_ROOT, (Source("lm", "train_pbr", (0,)),), LMO_OBJ_IDS, "lmo")
    ds = RefineDataset(assets, size=128, dataset="lmo", split="val", index=index, augment=False)
    s = ds[3]
    assert s["image"].shape == (3, 128, 128) and s["K_crop"].shape == (3, 3)
    for k in ("R_init", "t_init", "R_gt", "t_gt"):
        assert s[k].dtype == torch.float32
    assert not torch.equal(s["R_init"], s["R_gt"])
    assert torch.equal(ds[3]["R_init"], s["R_init"])  # deterministic without augmentation
    # the initial pose's own crop contains all its projected points; the GT is close to the crop too
    K = index["K"][3].astype(np.float64)
    bbox, M, Kc = pose_crop(ds.points_box[int(index["obj_idx"][3])], s["R_init"].double().numpy(), s["t_init"].double().numpy(), K, 128, 1.4)
    pts = ds.points_box[int(index["obj_idx"][3])] @ s["R_init"].double().numpy().T + s["t_init"].double().numpy()
    c = (Kc @ pts.T).T
    c = c[:, :2] / c[:, 2:]
    assert c.min() > 128 * 0.1 and c.max() < 128 * 0.9  # 1.4 margin -> inside the middle 71 %
    np.testing.assert_allclose(s["K_crop"].numpy(), Kc, atol=1e-3)  # the sample carries the crop's K


@needs_bop
def test_refine_instances_chains_iterations():
    assets = load_render_assets(BOP_ROOT, "lmo", LMO_OBJ_IDS, 4096)
    centre, half = object_extents(BOP_ROOT, "lmo", LMO_OBJ_IDS)
    renderer = PointRenderer(assets, centre, half, DEV)

    class Shrink(torch.nn.Module):  # a fixed update: depth x 1.01 each iteration
        def forward(self, x, obj):
            out = torch.zeros(len(x), 9, device=x.device)
            out[:, 2] = np.log(1.01)
            return out

    K = np.array([[572.0, 0, 320.0], [0, 573.0, 240.0], [0, 0, 1.0]])
    image = np.zeros((480, 640, 3), np.uint8)
    R = np.tile(np.eye(3), (2, 1, 1))
    t = np.array([[0.0, 0.0, 700.0], [50.0, 20.0, 900.0]])
    res = refine_instances(Shrink(), renderer, assets["points"][:, :2000], image, K, [0, 3], R, t, 3, 128, 1.4, DEV)
    assert len(res) == 3
    for n, (Rn, tn) in enumerate(res, start=1):
        np.testing.assert_allclose(tn[:, 2], t[:, 2] * 1.01**n, rtol=1e-6)
        np.testing.assert_allclose(Rn, R, atol=1e-6)
