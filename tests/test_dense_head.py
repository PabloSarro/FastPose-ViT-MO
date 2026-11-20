"""Dense heads of FastPoseViT (M18.2): shapes, per-object selection, and no change without them."""

import pytest
import torch

from src.models import FastPoseViT

VIT = "vit_b_16_no_weights"  # 224 px, 14 x 14 patches: small enough for the CPU
OBJECTS = 3


def build(conditioning="token", dense=True, **kw):
    torch.manual_seed(0)
    return FastPoseViT(
        vit_model=VIT,
        num_hidden_layers=0,
        hidden_layer_dim=0,
        out_dim_translation=3,
        out_dim_rotation=6,
        merge_outputs=True,
        no_mlp=True,
        num_objects=OBJECTS,
        object_conditioning=conditioning,
        dense_heads=dense,
        dense_size=32,
        **kw,
    ).eval()


@pytest.fixture(scope="module")
def images():
    return torch.randn(2, 3, 224, 224, generator=torch.Generator().manual_seed(1))


@pytest.mark.parametrize("conditioning", ["token", "heads", "both"])
def test_dense_output_shape_and_unchanged_pose(conditioning, images):
    model = build(conditioning)
    obj = torch.tensor([2, 0])
    with torch.no_grad():
        plain = model(images, obj)
        pose, dense = model(images, obj, return_dense=True)
    assert plain.shape == (2, 9) and dense.shape == (2, 4, 32, 32)
    assert torch.equal(plain, pose)  # the pose output does not depend on the flag


def test_per_object_heads_select_the_samples_object(images):
    model = build("heads")
    with torch.no_grad():
        _, both = model(images, torch.tensor([2, 0]), return_dense=True)
        _, first = model(images[:1], torch.tensor([2]), return_dense=True)
        _, other = model(images[:1], torch.tensor([0]), return_dense=True)
    torch.testing.assert_close(both[0], first[0], atol=1e-5, rtol=1e-4)
    assert not torch.allclose(first, other, atol=1e-3)  # a different object's output set


def test_state_dict_is_unchanged_without_dense_heads():
    base = set(build(dense=False).state_dict())
    dense = set(build(dense=True).state_dict())
    assert base < dense and all(k.startswith("dense.") for k in dense - base)
    # same weights without the flag: a checkpoint of an existing model loads
    build(dense=False).load_state_dict(build(dense=False).state_dict())


def test_return_dense_needs_dense_heads(images):
    with pytest.raises(ValueError, match="dense_heads"):
        build(dense=False)(images, torch.tensor([0, 1]), return_dense=True)


def test_convnext_has_no_dense_heads():
    with pytest.raises(ValueError, match="ViT"):
        FastPoseViT(
            vit_model="convnext_base",
            num_hidden_layers=0,
            hidden_layer_dim=0,
            out_dim_translation=3,
            out_dim_rotation=6,
            merge_outputs=True,
            no_mlp=True,
            dense_heads=True,
        )


def test_dense_loss_reaches_decoder_and_backbone(images):
    model = build("token").train()
    _, dense = model(images, torch.tensor([1, 2]), return_dense=True)
    dense.square().mean().backward()
    for name, p in model.dense.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, name
    assert model.backbone.conv_proj.weight.grad.abs().sum() > 0  # patch tokens reach the backbone
    assert model.object_token.grad.abs().sum() > 0  # and the object token conditions them


def test_wrong_grid_is_rejected():
    from src.models import DenseHead

    head = DenseHead(in_dim=32, grid=4, size=8, hidden=32)
    assert head(torch.randn(2, 16, 32)).shape == (2, 4, 8, 8)
    with pytest.raises(ValueError, match="patch tokens"):
        head(torch.randn(2, 25, 32))
