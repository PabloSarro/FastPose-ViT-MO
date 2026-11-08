# M9: the ONNX export of a multi-object FastPoseViT has a dynamic batch and matches PyTorch.

import numpy as np
import pytest
import torch

from optimization.bop_export import export_onnx
from src.models import FastPoseViT

ort = pytest.importorskip("onnxruntime")


@pytest.mark.parametrize("conditioning", ["token", "heads"])
def test_onnx_dynamic_batch_matches_torch(tmp_path, conditioning):
    torch.manual_seed(0)
    model = FastPoseViT(
        vit_model="vit_b_16_no_weights",
        num_hidden_layers=0,
        hidden_layer_dim=0,
        out_dim_translation=3,
        out_dim_rotation=6,
        merge_outputs=True,
        no_mlp=True,
        num_objects=3,
        object_conditioning=conditioning,
    ).eval()
    if conditioning == "token":
        # Non-zero object embeddings, so that a wrong obj_idx would change the output
        with torch.no_grad():
            model.object_token.normal_(std=0.5)
    path = str(tmp_path / "model.onnx")
    export_onnx(model, path, img_size=224)

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    for n in (1, 5):  # 5 != the export example's batch of 2
        images = torch.randn(n, 3, 224, 224)
        obj_idx = torch.arange(n, dtype=torch.int64) % 3
        with torch.no_grad():
            ref = model(images, obj_idx).numpy()
        out = sess.run(["pose"], {"images": images.numpy(), "obj_idx": obj_idx.numpy()})[0]
        assert out.shape == (n, 9)
        np.testing.assert_allclose(out, ref, atol=1e-4, rtol=0)
        # The object index matters (the graph did not freeze it)
        other = sess.run(["pose"], {"images": images.numpy(), "obj_idx": (obj_idx + 1).numpy() % 3})
        assert np.abs(other[0] - out).max() > 1e-3
