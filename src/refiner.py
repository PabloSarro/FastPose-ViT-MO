"""Render-and-compare pose refiner (M19 R2): given an image, the camera, an object and a pose
estimate, it predicts a pose update from the image crop and a render of the model at that pose.

Network: our FastPoseViT (a ViT-B with an object token, one linear head) on 10 channels: the image
crop (3), the render's shaded colour (3), its normalised object coordinates (3) and its mask (1),
all cropped around the projection of the object at the current pose (box x margin, as CosyPose).
Output (9): image-space shift of the object origin (du, dv, in crop sides), log depth ratio, and a
6D rotation update that is applied in the camera frame. The update:
    z' = z * exp(dz);   (u', v') = (u, v) + (du, dv) * side;   X' = (u' - cx) z' / fx, ...;   R' = dR R
with (u, v) the projection of the model origin in the crop. It is the parametrisation of CosyPose.

Loss: point matching on model points, L1 in units of the object diameter, minimum over the
object's symmetries (the one chosen by the full pose is used for all three terms), disentangled:
total = full pose + rotation only (translation of the chosen symmetry) + translation only.

Poses are camera-from-model, translations in mm, K in BOP convention (pixel i centred at i + 0.5).
"""

import numpy as np
import torch
import torch.nn as nn

from src.bop_dataset import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    BOPDataset,
    crop_image,
    crop_warp_matrix,
    nearest_rotation,
)
from src.models import FastPoseViT
from src.render import PointRenderer, crop_intrinsics

IN_CHANNELS = 10  # image 3 + render colour 3 + render xyz 3 + render mask 1
IDENTITY_6D = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)  # first two columns of the identity


def rotation_from_6d(v: torch.Tensor) -> torch.Tensor:
    """Rotation matrices (B, 3, 3) from 6D vectors (B, 6): Gram-Schmidt on the two columns."""
    b1 = nn.functional.normalize(v[:, 0:3], dim=-1)
    a2 = v[:, 3:6]
    b2 = nn.functional.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def rotation_to_6d(R: torch.Tensor) -> torch.Tensor:
    return torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)


def apply_update(
    R: torch.Tensor, t: torch.Tensor, K_crop: torch.Tensor, out: torch.Tensor, side: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The refined pose (R', t') from the current (R (B, 3, 3), t (B, 3) mm) and the network
    output (B, 9), in the crop of `side` pixels with intrinsics K_crop (B, 3, 3)."""
    identity = torch.tensor(IDENTITY_6D, dtype=out.dtype, device=out.device)
    dR = rotation_from_6d(out[:, 3:9] + identity)
    fx, fy = K_crop[:, 0, 0], K_crop[:, 1, 1]
    cx, cy = K_crop[:, 0, 2], K_crop[:, 1, 2]
    z = t[:, 2]
    u = fx * t[:, 0] / z + cx + out[:, 0] * side
    v = fy * t[:, 1] / z + cy + out[:, 1] * side
    z2 = z * torch.exp(out[:, 2])
    t2 = torch.stack([(u - cx) * z2 / fx, (v - cy) * z2 / fy, z2], dim=-1)
    return dR @ R, t2


def exact_update(
    R: torch.Tensor, t: torch.Tensor, R_target: torch.Tensor, t_target: torch.Tensor,
    K_crop: torch.Tensor, side: int,
) -> torch.Tensor:
    """The network output (B, 9) that apply_update turns (R, t) into the target pose."""
    fx, fy = K_crop[:, 0, 0], K_crop[:, 1, 1]
    du = (fx * t_target[:, 0] / t_target[:, 2] - fx * t[:, 0] / t[:, 2]) / side
    dv = (fy * t_target[:, 1] / t_target[:, 2] - fy * t[:, 1] / t[:, 2]) / side
    dz = torch.log(t_target[:, 2] / t[:, 2])
    dR = R_target @ R.transpose(1, 2)
    identity = torch.tensor(IDENTITY_6D, dtype=R.dtype, device=R.device)
    return torch.cat([du[:, None], dv[:, None], dz[:, None], rotation_to_6d(dR) - identity], dim=1)


class RefinerNet(nn.Module):
    """FastPoseViT (a ViT with an object token, or a ConvNeXt with per-object heads) on IN_CHANNELS
    channels with a zero-initialised head (= the identity update).

    The first convolution takes the pretrained RGB filters for the image; the render's colour
    filters start at zero (default), negated or copied (render_init) and the xyz / mask channels
    at zero. forward(x (B, 10, S, S), obj_idx (B,)) -> (B, 9).
    """

    def __init__(self, vit_model: str, num_objects: int, render_init: str = "zero"):
        super().__init__()
        if render_init not in ("zero", "copy", "neg"):
            raise ValueError(f"render_init must be zero, copy or neg, got {render_init!r}")
        cnn = vit_model.startswith("convnext")
        self.net = FastPoseViT(
            vit_model=vit_model, num_hidden_layers=0, hidden_layer_dim=0, out_dim_translation=3,
            out_dim_rotation=6, merge_outputs=True, nb_class_tokens=1, no_mlp=True,
            num_objects=num_objects, object_conditioning="heads" if cnn else "token",
        )
        old = self.net.backbone.features[0][0] if cnn else self.net.backbone.conv_proj
        new = nn.Conv2d(IN_CHANNELS, old.out_channels, old.kernel_size, old.stride, bias=old.bias is not None)
        with torch.no_grad():
            new.weight.zero_()
            new.weight[:, 0:3] = old.weight
            # the render's colour filters: 0 = start as the pretrained image model and learn to use
            # the render; neg = the first layer computes image - render (a built-in comparison);
            # copy = image + render (entangled: kept for the record, it did not learn)
            new.weight[:, 3:6] = {"zero": 0.0, "copy": 1.0, "neg": -1.0}[render_init] * old.weight
            if old.bias is not None:
                new.bias.copy_(old.bias)
            self.net.output.weight.zero_()
            self.net.output.bias.zero_()
        if cnn:
            self.net.backbone.features[0][0] = new
        else:
            self.net.backbone.conv_proj = new

    def forward(self, x: torch.Tensor, obj_idx: torch.Tensor) -> torch.Tensor:
        return self.net(x, obj_idx)


def make_input(image: torch.Tensor, render: torch.Tensor) -> torch.Tensor:
    """(B, 10, S, S) network input from the normalised image crop (B, 3, S, S) and the render
    (B, 7, S, S): the render's colour is normalised like the image, xyz and mask are kept."""
    mean = torch.tensor(IMAGENET_MEAN, device=render.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=render.device).view(1, 3, 1, 1)
    colour = (render[:, 0:3] - mean) / std * render[:, 6:7]  # background stays 0
    return torch.cat([image, colour, render[:, 3:7]], dim=1)


class RefineLoss(nn.Module):
    """Symmetric disentangled point-matching loss (see the module docstring).

    Args:
        loss_points: (n_obj, Np, 3) model points in mm.
        diameters: (n_obj,) mm.
        sym_R, sym_t: (n_obj, K, 3, 3), (n_obj, K, 3) the symmetries padded with the identity
            (bop_dataset.symmetry_table).
    """

    def __init__(self, loss_points, diameters, sym_R, sym_t):
        super().__init__()
        self.register_buffer("points", torch.as_tensor(loss_points, dtype=torch.float32))
        self.register_buffer("diameters", torch.as_tensor(diameters, dtype=torch.float32))
        self.register_buffer("sym_R", torch.as_tensor(sym_R, dtype=torch.float32))
        self.register_buffer("sym_t", torch.as_tensor(sym_t, dtype=torch.float32))

    def forward(self, R, t, R_gt, t_gt, obj_idx) -> dict:
        P = self.points[obj_idx]  # (B, Np, 3)
        diameter = self.diameters[obj_idx]
        S, s = self.sym_R[obj_idx], self.sym_t[obj_idx]  # (B, K, 3, 3), (B, K, 3)
        R_k = R_gt[:, None] @ S  # equivalent GT poses (R_gt S_k, t_gt + R_gt s_k)
        t_k = t_gt[:, None] + torch.einsum("bij,bkj->bki", R_gt, s)
        gt_pts = torch.einsum("bkij,bnj->bkni", R_k, P) + t_k[:, :, None]  # (B, K, Np, 3)
        pred_pts = torch.einsum("bij,bnj->bni", R, P) + t[:, None]  # (B, Np, 3)
        full_k = (pred_pts[:, None] - gt_pts).abs().mean((2, 3))  # (B, K)
        k = full_k.argmin(dim=1)
        batch = torch.arange(len(k), device=k.device)
        norm = diameter
        full = full_k[batch, k] / norm
        rot_pts = torch.einsum("bij,bnj->bni", R, P) + t_k[batch, k][:, None]
        rot = (rot_pts - gt_pts[batch, k]).abs().mean((1, 2)) / norm
        trans = (t - t_k[batch, k]).abs().mean(1) / norm
        return {
            "total": (full + rot + trans).mean(),
            "full": full.mean(), "rot": rot.mean(), "trans": trans.mean(),
        }


# Noise levels of the initial poses (the pose the refiner starts from): rotation angle (degrees,
# half-normal about a random axis), image shift of the origin (crop sides) and log depth ratio.
# Mixture: errors like our direct-regression ones (median ~4.6 deg, ~16 mm), and larger ones.
NOISE_LEVELS = (
    {"p": 0.45, "rot": 3.0, "xy": 0.01, "z": 0.015},
    {"p": 0.35, "rot": 8.0, "xy": 0.03, "z": 0.04},
    {"p": 0.20, "rot": 20.0, "xy": 0.08, "z": 0.10},
)


def rotation_about(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    a = np.radians(angle_deg)
    A = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * A + (1 - np.cos(a)) * A @ A


def perturb_pose(R, t, K, side_px: float, rng: np.random.Generator, levels=NOISE_LEVELS):
    """A noisy copy of the pose (R, t): camera-frame rotation, image shift of the origin and depth
    ratio, drawn from one of the noise levels. `side_px` converts crop sides to image pixels."""
    level = levels[rng.choice(len(levels), p=[lv["p"] for lv in levels])]
    dR = rotation_about(rng.normal(size=3), abs(rng.normal()) * level["rot"])
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    z = t[2] * np.exp(rng.normal() * level["z"])
    u = fx * t[0] / t[2] + cx + rng.normal() * level["xy"] * side_px
    v = fy * t[1] / t[2] + cy + rng.normal() * level["xy"] * side_px
    return dR @ R, np.array([(u - cx) * z / fx, (v - cy) * z / fy, z])


def pose_crop(points: np.ndarray, R, t, K, size: int, margin: float):
    """Crop of the object at a pose: (bbox xyxy in K coordinates, warp M (2, 3), K_crop (3, 3)).

    The box is that of the projected `points` (n, 3), widened to a square of side margin x max(w, h)
    by crop_warp_matrix, as CosyPose's 1.4. Needs the object in front of the camera."""
    cam = points @ R.T + t
    uv = cam[:, :2] / cam[:, 2:3] * np.array([K[0, 0], K[1, 1]]) + np.array([K[0, 2], K[1, 2]])
    lo, hi = uv.min(0), uv.max(0)
    bbox = [lo[0], lo[1], hi[0], hi[1]]
    M = crop_warp_matrix(bbox, size, margin)
    return bbox, M, crop_intrinsics(K, M)


class RefineDataset(BOPDataset):
    """Training / validation samples of the refiner: one object instance, a perturbed pose and the
    image crop around the object at that pose. The render is made on the GPU from the sample.

    Item (dict): image (3, S, S) normalised crop, K_crop (3, 3), R_init, t_init, R_gt, t_gt,
    obj_idx, diameter, row. With augment=False the perturbation is deterministic per row.
    """

    def __init__(
        self, assets: dict, size: int = 256, margin: float = 1.4, noise_seed: int = 0,
        noise_levels=NOISE_LEVELS, **kwargs,
    ):
        kwargs.setdefault("img_size", size)
        kwargs.setdefault("crop_context", True)
        super().__init__(**kwargs)
        self.size, self.margin, self.noise_seed = size, margin, noise_seed
        self.noise_levels = noise_levels
        self.points_box = np.asarray(assets["points"][:, :2000], dtype=np.float64)

    def __getitem__(self, row: int) -> dict:
        index = self.index
        obj = int(index["obj_idx"][row])
        K = index["K"][row].astype(np.float64)
        R_gt = nearest_rotation(index["R"][row].astype(np.float64))
        t_gt = index["t"][row].astype(np.float64)
        rng = np.random.default_rng() if self.augment else np.random.default_rng((self.noise_seed, row))
        R, t = perturb_pose(R_gt, t_gt, K, self.size / self.margin, rng, self.noise_levels)
        image = self.load_image(row)
        if self.real_background and index["source"][row] > 0:
            image = self.replace_background(row, image)
        if t[2] < 100 or t_gt[2] < 100:  # degenerate (never for BOP): keep the GT
            R, t = R_gt, t_gt
        bbox, M, K_crop = pose_crop(self.points_box[obj], R, t, K, self.size, self.margin)
        crop = crop_image(image, bbox, self.size, True, self.margin)
        tensor = self.transform(image=crop.astype(np.float32) / 255.0)["image"]
        f32 = lambda a: torch.from_numpy(np.asarray(a, dtype=np.float32))  # noqa: E731
        return {
            "image": tensor, "K_crop": f32(K_crop), "R_init": f32(R), "t_init": f32(t),
            "R_gt": f32(R_gt), "t_gt": f32(t_gt), "obj_idx": obj,
            "diameter": torch.tensor(index["diameters"][obj], dtype=torch.float32), "row": row,
        }


@torch.no_grad()
def refine_instances(
    model: nn.Module, renderer: PointRenderer, points_box: np.ndarray, image: np.ndarray,
    K: np.ndarray, obj_idx, R, t, n_iterations: int, size: int, margin: float, device,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Refine the poses of the instances of one image, iterating the network.

    Args:
        image: (H, W, 3) uint8 RGB. K: (3, 3). obj_idx (N,), R (N, 3, 3), t (N, 3) mm.
    Returns:
        [(R, t)] after iteration 1..n_iterations, each (N, 3, 3), (N, 3) numpy.
    """
    obj_idx = np.asarray(obj_idx)
    R, t = np.array(R, dtype=np.float64), np.array(t, dtype=np.float64)
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    results = []
    for _ in range(n_iterations):
        crops, K_crops = [], []
        for i in range(len(obj_idx)):
            bbox, _, K_crop = pose_crop(points_box[obj_idx[i]], R[i], t[i], K, size, margin)
            crops.append((crop_image(image, bbox, size, True, margin).astype(np.float32) / 255.0 - mean) / std)
            K_crops.append(K_crop)
        x_img = torch.from_numpy(np.stack(crops)).permute(0, 3, 1, 2).contiguous().to(device)
        K_t = torch.tensor(np.stack(K_crops), dtype=torch.float32, device=device)
        R_t = torch.tensor(R, dtype=torch.float32, device=device)
        t_t = torch.tensor(t, dtype=torch.float32, device=device)
        o_t = torch.as_tensor(obj_idx, device=device).long()
        render, _ = renderer.render(o_t, R_t, t_t, K_t, size)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            out = model(make_input(x_img, render), o_t)
        R2, t2 = apply_update(R_t, t_t, K_t, out.float(), size)
        R, t = R2.cpu().numpy().astype(np.float64), t2.cpu().numpy().astype(np.float64)
        results.append((R.copy(), t.copy()))
    return results
