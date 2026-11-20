"""Dense losses and the engine with dense heads (M18.3): values on small cases, and one training
and validation epoch on a tiny model, with the dense terms in the total loss."""

import types

import pytest
import torch

from src.engine import TrainingConfig, ValidationConfig, forward_dense, train_epoch, val_epoch
from src.losses import dense_losses


def test_dense_losses_values():
    dense = torch.zeros(1, 4, 2, 2)
    dense[:, 0] = 100.0  # confident "object" logits
    dense[:, 1:4, 0, 0] = 0.5  # XYZ prediction 0.5 at one pixel
    mask = torch.ones(1, 1, 2, 2)
    xyz = torch.zeros(1, 3, 2, 2)
    valid = torch.zeros(1, 1, 2, 2)
    valid[0, 0, 0, 0] = 1.0  # only the first pixel counts for XYZ
    bce, l1 = dense_losses(dense, mask, xyz, valid)
    assert bce.item() == pytest.approx(0.0, abs=1e-6)
    assert l1.item() == pytest.approx(0.5)  # |0.5 - 0| over the 3 coordinates of 1 pixel
    # wrong mask: BCE of a logit of 100 against a 0 target is ~100
    bce_wrong, _ = dense_losses(dense, torch.zeros(1, 1, 2, 2), xyz, valid)
    assert bce_wrong.item() == pytest.approx(100.0, rel=1e-3)
    # nothing valid: finite, zero XYZ loss, gradient flows into the mask only
    dense.requires_grad_()
    bce, l1 = dense_losses(dense, mask, xyz, torch.zeros_like(valid))
    (bce + l1).backward()
    assert l1.item() == 0.0 and torch.isfinite(dense.grad).all()
    assert dense.grad[:, 1:4].abs().sum() == 0


class TinyDense(torch.nn.Module):
    """Same call signature as FastPoseViT with dense heads; merged 9 + 4 dense channels."""

    def __init__(self, size=8):
        super().__init__()
        self.pose = torch.nn.Linear(3, 9)
        self.dense = torch.nn.Conv2d(3, 4, 1)
        self.size = size

    def forward(self, x, obj_idx=None, return_dense=False):
        pose = self.pose(x.mean((2, 3)))
        if not return_dense:
            return pose
        return pose, self.dense(torch.nn.functional.adaptive_avg_pool2d(x, self.size))


def batch(n=4, size=8):
    g = torch.Generator().manual_seed(0)
    R = torch.eye(3).expand(n, 3, 3).clone()
    return {
        "images": torch.randn(n, 3, 16, 16, generator=g),
        "translations": torch.zeros(n, 3),
        "rotations": R,
        "bboxes": torch.tensor([[0.0, 0.0, 16.0, 16.0]] * n),
        "obj_idx": torch.zeros(n, dtype=torch.long),
        "mask": (torch.rand(n, 1, size, size, generator=g) > 0.5).float(),
        "xyz": torch.rand(n, 3, size, size, generator=g) * 2 - 1,
        "xyz_valid": torch.ones(n, 1, size, size),
    }


def test_forward_dense_splits_merged_output():
    t, R, dense = forward_dense(TinyDense(), batch()["images"], None, merge_outputs=True)
    assert t.shape == (4, 3) and R.shape == (4, 6) and dense.shape == (4, 4, 8, 8)


def test_train_epoch_adds_dense_terms_and_trains_the_decoder():
    model = TinyDense()
    before = model.dense.weight.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    translation = lambda p, t, b: (p - t).abs().mean()  # noqa: E731
    rotation = lambda p, t: (p - t).abs().mean()  # noqa: E731
    config = TrainingConfig(
        rotation_format="matrix",
        rotation_loss_type="6d_relative_frobenius",
        merge_outputs=True,
        max_grad_norm=0.0,
        use_per_iteration_scheduling=False,
        dense=True,
        mask_loss_weight=1.0,
        xyz_loss_weight=2.0,
    )
    loader = [batch(), batch()]
    total, trans, rot = train_epoch(
        model, loader, optimizer, translation, rotation, torch.device("cpu"), 0, None, config
    )
    assert total > trans + rot  # the dense terms are in the total, not in the reported pose losses
    assert not torch.equal(before, model.dense.weight)  # the decoder got gradients


def test_dense_without_targets_is_a_clear_error():
    model = TinyDense()
    config = TrainingConfig(
        rotation_format="matrix", merge_outputs=True, use_per_iteration_scheduling=False, dense=True
    )
    b = batch()
    for k in ("mask", "xyz", "xyz_valid"):
        del b[k]
    with pytest.raises(KeyError, match="dense_size"):
        train_epoch(
            model,
            [b],
            torch.optim.SGD(model.parameters(), lr=0.1),
            lambda p, t, bb: p.mean(),
            lambda p, t: p.mean(),
            torch.device("cpu"),
            0,
            None,
            config,
        )
