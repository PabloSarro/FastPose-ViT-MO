"""Batched GPU point-splatting renderer for the render-and-compare refiner (M19 R2).

No mesh rasteriser and no extra dependency: every object is a fixed cloud of points sampled on
its surface (with colour, normal and normalised object coordinates). A render projects the points
with the pose and the crop intrinsics, keeps the nearest point per pixel with a z-buffer
(`scatter_reduce` "amin"), optionally splatting each point on a (2r+1)^2 neighbourhood, and closes
the few holes at grazing angles from the neighbouring pixels. It returns, per instance, a 7-channel image: headlight-shaded colour (3), normalised object
coordinates (3) and the mask (1), plus the depth in mm. Nothing is differentiable (the refiner
does not need it): the render is an input.

Coordinates and units: models in mm in the BOP model frame; a pose is (R, t) camera-from-model;
K is the intrinsics of the image being rendered, with pixel i centred at i + 0.5 (BOP, as
src/bop_dataset.crop_warp_matrix).
"""

import os

import numpy as np
import torch

# Where each dataset's render models live under the BOP root: LM (LM-O objects) and YCB-V models
# with vertex colours / texture, T-LESS CAD models (grey; the headlight shading gives the shape)
RENDER_MODEL_DIRS = {"lmo": "lm/models", "ycbv": "ycbv/models", "tless": "tless/models_cad"}
GREY = 0.6


def _sample_texture(texture: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Bilinear colours (n, 3) in [0, 1] of a texture image (H, W, 3 uint8) at uv (n, 2).

    BOP ply files give (u, v) with v counted from the bottom of the image."""
    h, w = texture.shape[:2]
    x = np.clip(uv[:, 0], 0, 1) * (w - 1)
    y = np.clip(1.0 - uv[:, 1], 0, 1) * (h - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    tex = texture.astype(np.float64) / 255.0
    top = tex[y0, x0] * (1 - fx) + tex[y0, x1] * fx
    bottom = tex[y1, x0] * (1 - fx) + tex[y1, x1] * fx
    return top * (1 - fy) + bottom * fy


def sample_model_points(ply_path: str, n_points: int, seed: int = 0) -> dict:
    """n_points on the surface of a BOP model, area-weighted over its triangles.

    Returns {"points": (n, 3) mm, "normals": (n, 3) unit, "colors": (n, 3) in [0, 1]}. Colours
    come from the vertex colours, else from the texture image next to the ply (YCB-V), else grey.
    """
    from bop_toolkit_lib import inout

    model = inout.load_ply(ply_path)
    pts, faces = np.asarray(model["pts"], np.float64), np.asarray(model["faces"], np.int64)
    tri = pts[faces]  # (F, 3, 3)
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(cross, axis=1)
    rng = np.random.default_rng(seed)
    face = rng.choice(len(faces), size=n_points, p=area / area.sum())
    r1, r2 = rng.random(n_points), rng.random(n_points)
    root = np.sqrt(r1)
    bary = np.stack([1 - root, root * (1 - r2), root * r2], axis=1)  # (n, 3)
    points = np.einsum("nk,nkd->nd", bary, tri[face])
    face_normal = cross / np.maximum(np.linalg.norm(cross, axis=1, keepdims=True), 1e-12)
    if "normals" in model:  # smooth vertex normals, interpolated
        vn = np.asarray(model["normals"], np.float64)[faces[face]]
        normals = np.einsum("nk,nkd->nd", bary, vn)
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    else:
        normals = face_normal[face]
    if "colors" in model:
        vc = np.asarray(model["colors"], np.float64)[faces[face]]
        colors = np.einsum("nk,nkd->nd", bary, vc) / (255.0 if vc.max() > 1.5 else 1.0)
    elif "texture_uv" in model:
        from PIL import Image

        texture = np.array(
            Image.open(os.path.join(os.path.dirname(ply_path), model["texture_file"])).convert("RGB")
        )
        uv = np.einsum("nk,nkd->nd", bary, np.asarray(model["texture_uv"], np.float64)[faces[face]])
        colors = _sample_texture(texture, uv)
    else:
        colors = np.full((n_points, 3), GREY)
    return {
        "points": points.astype(np.float32),
        "normals": normals.astype(np.float32),
        "colors": np.clip(colors, 0, 1).astype(np.float32),
    }


def load_render_assets(
    bop_root: str, dataset: str, obj_ids, n_points: int = 65536, cache_dir: str | None = None
) -> dict:
    """Render clouds of all objects of a dataset: arrays (n_obj, n_points, 3) "points", "normals",
    "colors", cached as one .npz (sampling is deterministic, the cache only saves time)."""
    cache = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache = os.path.join(cache_dir, f"render_{dataset}_{n_points}_{len(list(obj_ids))}.npz")
        if os.path.exists(cache):
            with np.load(cache) as f:
                return {k: f[k] for k in f.files}
    folder = os.path.join(bop_root, RENDER_MODEL_DIRS[dataset])
    clouds = [
        sample_model_points(os.path.join(folder, f"obj_{int(o):06d}.ply"), n_points, seed=int(o))
        for o in obj_ids
    ]
    assets = {k: np.stack([c[k] for c in clouds]) for k in ("points", "normals", "colors")}
    if cache:
        np.savez(cache, **assets)
    return assets


def crop_intrinsics(K: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Intrinsics of a crop: K (3, 3) of the image and M (2, 3) the crop warp
    (bop_dataset.crop_warp_matrix: crop index = scale * image index + shift).

    A point at image coordinate u (K coordinates, pixel i centred at i + 0.5) lands at crop
    coordinate c = scale * u + (shift + 0.5 - 0.5 * scale), also with pixel j centred at j + 0.5.
    """
    K_crop = np.eye(3)
    for axis, centre in ((0, 2), (1, 2)):
        scale, shift = M[axis, axis], M[axis, 2]
        K_crop[axis, axis] = scale * K[axis, axis]
        K_crop[axis, centre] = scale * K[axis, 2] + shift + 0.5 - 0.5 * scale
    return K_crop


class PointRenderer:
    """Renders the point clouds of several objects at given poses (see the module docstring).

    Args:
        assets: load_render_assets output (numpy, (n_obj, N, 3) each).
        centre, half: (n_obj, 3) model-box centre / half-extent in mm (bop_dataset.object_extents):
            the XYZ channels are (X - centre) / half, as the dense targets of M18.
        device: Where the clouds live and the renders are made.
        radius: Splat radius in pixels (0 = one pixel per point; small holes are closed afterwards;
            1 = 3 x 3 splats, which fatten the silhouette by ~1.7 px: measured on a sphere).
    """

    def __init__(self, assets: dict, centre, half, device, radius: int = 0):
        self.points = torch.as_tensor(assets["points"], device=device)
        self.normals = torch.as_tensor(assets["normals"], device=device)
        self.colors = torch.as_tensor(assets["colors"], device=device)
        self.centre = torch.as_tensor(centre, dtype=torch.float32, device=device)
        self.half = torch.as_tensor(half, dtype=torch.float32, device=device)
        self.device = device
        offsets = [(dx, dy) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)]
        self.offsets = torch.tensor(offsets, device=device)  # (n_offsets, 2)

    @torch.no_grad()
    def render(self, obj_idx, R, t, K, size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Render B instances into size x size crops.

        Args:
            obj_idx: (B,) object index. R: (B, 3, 3). t: (B, 3) mm. K: (B, 3, 3) the crop
            intrinsics (image K composed with the crop, pixel i centred at i + 0.5).
            size: Output side in pixels.

        Returns:
            image (B, 7, size, size): shaded colour, normalised XYZ, mask; depth (B, 1, size, size)
            in mm (0 where empty). Deterministic: the same input gives the same render.
        """
        dev = self.device
        obj_idx = torch.as_tensor(obj_idx, device=dev).long()
        R = torch.as_tensor(R, dtype=torch.float32, device=dev)
        t = torch.as_tensor(t, dtype=torch.float32, device=dev)
        K = torch.as_tensor(K, dtype=torch.float32, device=dev)
        B, N = len(obj_idx), self.points.shape[1]
        cam = torch.einsum("bij,bnj->bni", R, self.points[obj_idx]) + t[:, None]
        z = cam[..., 2]
        safe_z = z.clamp(min=1e-3)
        u = K[:, 0, 0, None] * cam[..., 0] / safe_z + K[:, 0, 2, None]
        v = K[:, 1, 1, None] * cam[..., 1] / safe_z + K[:, 1, 2, None]
        ix, iy = torch.floor(u).long(), torch.floor(v).long()  # the pixel containing the point
        # every point lands on the (2r+1)^2 pixels around its own
        px = ix[None] + self.offsets[:, 0, None, None]  # (O, B, N)
        py = iy[None] + self.offsets[:, 1, None, None]
        valid = (z > 1e-3)[None] & (px >= 0) & (px < size) & (py >= 0) & (py < size)
        batch = torch.arange(B, device=dev)[None, :, None].expand_as(px)
        flat = ((batch * size + py) * size + px)[valid]  # (n_valid,)
        zz = z[None].expand_as(px)[valid]
        n_pix = B * size * size
        zbuf = torch.full((n_pix,), float("inf"), device=dev)
        zbuf.scatter_reduce_(0, flat, zz, reduce="amin", include_self=True)
        # the points within a tolerance of the nearest compete; the lowest index wins, so the
        # render does not depend on the order in which the GPU handles the points
        front = zz <= zbuf[flat] * 1.002 + 0.5
        which = torch.arange(N, device=dev)[None, None].expand_as(px)[valid][front]
        winner = torch.full((n_pix,), N, dtype=torch.long, device=dev)
        winner.scatter_reduce_(0, flat[front], which, reduce="amin", include_self=True)
        pix = torch.nonzero(winner < N)[:, 0]
        owner = pix // (size * size)
        point = winner[pix]
        obj = obj_idx[owner]
        xyz = (self.points[obj, point] - self.centre[obj]) / self.half[obj]
        normal_cam = torch.einsum("nij,nj->ni", R[owner], self.normals[obj, point])
        shade = 0.25 + 0.75 * normal_cam[:, 2].abs()  # two-sided headlight: mesh normals may be flipped
        attrs = torch.zeros(n_pix, 8, device=dev)  # shaded colour 3, xyz 3, mask 1, depth 1
        attrs[pix] = torch.cat(
            [self.colors[obj, point] * shade[:, None], xyz, torch.ones_like(shade)[:, None], zbuf[pix, None]],
            dim=1,
        )
        image = attrs.view(B, size, size, 8).permute(0, 3, 1, 2).contiguous()
        image = self._close_holes(self._close_holes(image))  # two passes: ~1 point per pixel on large crops
        return image[:, :7].contiguous(), image[:, 7:8].contiguous()

    @staticmethod
    def _close_holes(image: torch.Tensor) -> torch.Tensor:
        """Fill empty pixels that have at least 5 of their 8 neighbours filled with the mean of
        those neighbours. The splats leave small holes where the surface is seen at a grazing
        angle (few points per pixel); this closes them without growing the silhouette."""
        mask = image[:, 6:7]
        weight = torch.nn.functional.avg_pool2d(mask, 3, stride=1, padding=1, count_include_pad=True) * 9.0
        mean = torch.nn.functional.avg_pool2d(image * mask, 3, stride=1, padding=1, count_include_pad=True) * 9.0
        fill = (mask < 0.5) & (weight >= 4.5)
        filled = mean / weight.clamp(min=1.0)
        out = torch.where(fill, filled, image)
        out[:, 6:7] = torch.where(fill, torch.ones_like(mask), mask)
        return out
