"""FastPoseViT(num_objects=N): per-object output heads.

- num_objects=1 is the original model: same parameter names and shapes, so the published
  SPEED weights load unchanged and give the same output.
- num_objects>1: the last layer has N times more outputs; forward(x, obj_idx) returns each
  sample's own object block, and forward(x) all blocks as (B, N, D).
"""

import os

import pytest
import torch

from src.models import FastPoseViT

WEIGHTS = os.path.join(os.path.dirname(__file__), "..", "weights", "vit_b_16_384.pth")
BEST_SPEED = dict(  # README best config: --no_mlp --merge_outputs, 9 outputs (3 + 6D)
    num_hidden_layers=0,
    hidden_layer_dim=0,
    out_dim_translation=3,
    out_dim_rotation=6,
    merge_outputs=True,
    no_mlp=True,
)


def small_model(num_objects=1, merge_outputs=True, no_mlp=True, seed=0):
    """ViT-B/16 at 224 px without pretrained weights (no download), for structure tests."""
    torch.manual_seed(seed)
    return FastPoseViT(
        "vit_b_16_no_weights",
        num_hidden_layers=0 if no_mlp else 1,
        hidden_layer_dim=0 if no_mlp else 64,
        out_dim_translation=3,
        out_dim_rotation=6,
        merge_outputs=merge_outputs,
        no_mlp=no_mlp,
        num_objects=num_objects,
    ).eval()


@pytest.mark.parametrize("merge_outputs", [True, False])
@pytest.mark.parametrize("no_mlp", [True, False])
def test_single_object_model_unchanged(merge_outputs, no_mlp):
    """Default num_objects=1: same state dict layout and outputs as before."""
    m1 = small_model(1, merge_outputs, no_mlp)
    shapes = {k: v.shape for k, v in m1.state_dict().items()}
    if merge_outputs and no_mlp:
        assert shapes["output.weight"] == (9, 768)
    elif not merge_outputs and no_mlp:
        assert shapes["translation.weight"] == (3, 768)
        assert shapes["rotation.weight"] == (6, 768)
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = m1(x)
    if merge_outputs:
        assert out.shape == (2, 9)
    else:
        assert out[0].shape == (2, 3) and out[1].shape == (2, 6)


@pytest.mark.skipif(not os.path.exists(WEIGHTS), reason="weights/vit_b_16_384.pth not present")
def test_published_speed_weights_load_into_default_model():
    model = FastPoseViT("vit_b_16_384_no_weights", **BEST_SPEED)
    model.load_from_pretrained(WEIGHTS)  # strict load_state_dict: names and shapes must match
    assert model.num_objects == 1


@pytest.mark.parametrize("merge_outputs", [True, False])
@pytest.mark.parametrize("no_mlp", [True, False])
def test_per_object_heads_select_the_sample_object(merge_outputs, no_mlp):
    n_obj = 8
    model = small_model(n_obj, merge_outputs, no_mlp)
    if merge_outputs and no_mlp:
        assert model.output.weight.shape == (9 * n_obj, 768)
    x = torch.randn(5, 3, 224, 224)
    obj_idx = torch.tensor([0, 3, 7, 3, 1])
    with torch.no_grad():
        all_objects = model(x)  # no obj_idx: every object's outputs
        selected = model(x, obj_idx)
    if merge_outputs:
        assert all_objects.shape == (5, n_obj, 9)
        assert selected.shape == (5, 9)
        torch.testing.assert_close(selected, all_objects[torch.arange(5), obj_idx])
    else:
        assert all_objects[0].shape == (5, n_obj, 3) and all_objects[1].shape == (5, n_obj, 6)
        assert selected[0].shape == (5, 3) and selected[1].shape == (5, 6)
        torch.testing.assert_close(selected[0], all_objects[0][torch.arange(5), obj_idx])
        torch.testing.assert_close(selected[1], all_objects[1][torch.arange(5), obj_idx])


def test_per_object_gradients_only_reach_the_sample_object_head():
    n_obj = 4
    model = small_model(n_obj).train()
    x = torch.randn(2, 3, 224, 224)
    model(x, torch.tensor([2, 2])).sum().backward()
    grad = model.output.weight.grad.view(n_obj, 9, -1).abs().sum(dim=(1, 2))
    assert grad[2] > 0
    assert torch.all(grad[[0, 1, 3]] == 0)


def test_invalid_num_objects():
    with pytest.raises(ValueError, match="num_objects"):
        small_model(0)
