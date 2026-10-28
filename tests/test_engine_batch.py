"""Engine plumbing for SPEED tuples and BOP dict batches.

- unpack_batch: SPEED tuple and BOP dict give the same keys; missing BOP fields are None.
- forward_model: same split as the original inline code, and obj_idx reaches the model.
- CoordinateTransformer: without K it is the original SPEED path; with K = Camera.K it gives
  the same result; with per-sample K and "site" depth it inverts the dataset-side transform.
"""

import torch

from src.camera import Camera
from src.engine import CoordinateTransformer, forward_model, unpack_batch
from src.utils import translation_to_bbox_relative_translation

LM_K = torch.tensor([[572.4114, 0.0, 325.2611], [0.0, 573.57043, 242.04899], [0.0, 0.0, 1.0]])


def speed_batch(n=6):
    torch.manual_seed(0)
    t = torch.stack([torch.randn(n) * 0.5, torch.randn(n) * 0.5, torch.rand(n) * 20 + 5], 1)
    centre = torch.stack([960 + 3003 * t[:, 0] / t[:, 2], 600 + 3003 * t[:, 1] / t[:, 2]], 1)
    size = (3000 / t[:, 2]).unsqueeze(1).expand(n, 2)
    bbox = torch.cat([centre - size / 2, centre + size / 2], 1)
    q = torch.nn.functional.normalize(torch.randn(n, 4), dim=1)
    return t, q, bbox


def test_unpack_speed_tuple():
    images, t, R, bbox = (
        torch.zeros(2, 3, 8, 8),
        torch.zeros(2, 3),
        torch.eye(3).expand(2, 3, 3),
        torch.zeros(2, 4),
    )
    batch = unpack_batch((images, t, R, bbox))
    assert batch["images"] is images and batch["translations"] is t
    assert batch["rotations"] is R and batch["bboxes"] is bbox
    assert all(batch[k] is None for k in ("K", "img_wh", "obj_idx", "diameter"))


def test_unpack_bop_dict_keeps_extra_fields():
    raw = {
        "images": torch.zeros(2, 3, 8, 8),
        "translations": torch.zeros(2, 3),
        "rotations": torch.eye(3).expand(2, 3, 3),
        "bboxes": torch.zeros(2, 4),
        "K": LM_K.expand(2, 3, 3),
        "obj_idx": torch.tensor([0, 5]),
        "diameter": torch.tensor([102.0, 262.0]),
        "scene_id": torch.tensor([1, 1]),
    }
    batch = unpack_batch(raw)
    assert batch["obj_idx"] is raw["obj_idx"] and batch["diameter"] is raw["diameter"]
    assert batch["img_wh"] is None
    assert batch["scene_id"] is raw["scene_id"]  # other fields pass through
    assert "img_wh" not in raw  # input not modified


class RecordingModel(torch.nn.Module):
    def __init__(self, merged):
        super().__init__()
        self.merged, self.calls = merged, []

    def forward(self, x, *args):
        self.calls.append(args)
        out = torch.arange(x.shape[0] * 9, dtype=torch.float32).view(-1, 9)
        return out if self.merged else (out[:, :3], out[:, 3:])


def test_forward_model_split_and_obj_idx():
    x = torch.zeros(4, 3, 8, 8)
    model = RecordingModel(merged=True)
    t, r = forward_model(model, x, None, merge_outputs=True)
    assert model.calls == [()]  # SPEED: called exactly as model(images)
    assert (
        t.shape == (4, 3)
        and r.shape == (4, 6)
        and torch.equal(t, torch.arange(36.0).view(4, 9)[:, :3])
    )
    obj_idx = torch.tensor([1, 0, 2, 2])
    forward_model(model, x, obj_idx, merge_outputs=True)
    assert model.calls[-1][0] is obj_idx
    t, r = forward_model(RecordingModel(merged=False), x, None, merge_outputs=False)
    assert t.shape == (4, 3) and r.shape == (4, 6)


def test_coordinate_transformer_with_speed_K_equals_legacy():
    t, q, bbox = speed_batch()
    U = translation_to_bbox_relative_translation(t, bbox, Camera)
    ct = CoordinateTransformer("quaternion", Camera, no_rotation_compensation=False)
    legacy = ct.transform_to_absolute(U, q, U, q, bbox)
    with_k = ct.transform_to_absolute(
        U,
        q,
        U,
        q,
        bbox,
        K=Camera.K.expand(len(t), 3, 3),
        img_wh=torch.tensor([[1920.0, 1200.0]]).expand(len(t), 2),
    )
    for a, b in zip(legacy, with_k):
        assert torch.equal(a, b)
    torch.testing.assert_close(legacy[0], t, rtol=1e-5, atol=1e-5)


def test_coordinate_transformer_site_mode_inverts_dataset_transform():
    torch.manual_seed(1)
    n = 8
    t = torch.stack([torch.randn(n) * 80, torch.randn(n) * 60, torch.rand(n) * 1000 + 500], 1)
    diameter = torch.rand(n) * 160 + 100
    K = LM_K.expand(n, 3, 3)
    centre = torch.stack(
        [LM_K[0, 0] * t[:, 0] / t[:, 2] + LM_K[0, 2], LM_K[1, 1] * t[:, 1] / t[:, 2] + LM_K[1, 2]],
        1,
    )
    size = (573 * diameter / t[:, 2]).unsqueeze(1).expand(n, 2)
    bbox = torch.cat([centre - size / 2, centre + size / 2], 1)
    U = translation_to_bbox_relative_translation(
        t, bbox, Camera, K=K, depth_mode="site", diameter=diameter
    )
    q = torch.nn.functional.normalize(torch.randn(n, 4), dim=1)
    ct = CoordinateTransformer(
        "quaternion", Camera, no_rotation_compensation=True, depth_mode="site"
    )
    abs_t = ct.transform_to_absolute(U, q, U, q, bbox, K=K, diameter=diameter)[0]
    torch.testing.assert_close(abs_t, t, rtol=1e-5, atol=1e-2)


def test_progress_tracker_shows_the_item_values():
    # update_losses copies the losses to the CPU only on displayed batches (every 10th), with
    # the same values as calling .item() on each
    from src.engine import ProgressTracker, smooth_loss

    tracker = ProgressTracker(smooth_window=5, desc="t")
    tracker.setup_progress_bar(40)
    losses = [torch.rand(3) for _ in range(40)]
    for i, (a, b, c) in enumerate(losses):
        tracker.update_losses(a, b, c, i)
    shown = [losses[i] for i in range(0, 40, 10)][-5:]
    assert list(tracker.loss_deque) == [x[0].item() for x in shown]
    assert list(tracker.loss_rotation_deque) == [x[2].item() for x in shown]
    assert tracker.pbar.postfix.startswith(f"loss={smooth_loss([x[0].item() for x in shown]):.4f}")
    tracker.close()
