# This file contains the BOP dataset: an index with one row per object instance, and BOPDataset,
# which turns a row into a square crop with pose labels
#
# The index is built from the standard BOP tree (scripts/make_bop_tree.sh) and cached as a single
# .npz of numpy arrays, so DataLoader workers share it without per-sample JSON parsing.

import argparse
import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, Sampler

from src.camera import Camera
from src.datasets import random_bbox_augmentation
from src.utils import (
    DEPTH_MODES,
    get_apparent_orientation,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
    translation_to_bbox_relative_translation,
)

# Fix OpenCV threading conflicts with PyTorch DataLoader workers (as src/datasets.py)
cv2.setNumThreads(0)

INDEX_VERSION = 1

# Objects of LM-O (the 8 LM objects that appear in the LM-O test scene)
LMO_OBJ_IDS = (1, 5, 6, 8, 9, 10, 11, 12)


@dataclass(frozen=True)
class Source:
    """One BOP split folder (or part of it) that contributes rows to an index.

    Attributes:
        dataset: BOP dataset folder, e.g. "lm".
        split: Split folder inside it, e.g. "train_pbr" or "test".
        scenes: Scene ids to use, or None for all scenes in the folder.
        targets: If True, keep only the instances listed in the dataset's test_targets_bop19.json
            (the official evaluation targets), without the visibility filter.
    """

    dataset: str
    split: str
    scenes: tuple | None = None
    targets: bool = False


# Splits per target dataset (see TASKS.md §5). LM-O has no training images of its own: the BOP
# practice is to train on LM's PBR images. LM test scene 2 is the LM-O test sequence, so it is
# left out of val.
SPLITS = {
    "lmo": {
        "obj_ids": LMO_OBJ_IDS,
        "models_dataset": "lmo",
        "train": (Source("lm", "train_pbr", tuple(range(0, 48))),),
        "val_pbr": (Source("lm", "train_pbr", (48, 49)),),
        "val_real": (Source("lm", "test", LMO_OBJ_IDS, targets=True),),
        "val": (
            Source("lm", "train_pbr", (48, 49)),
            Source("lm", "test", LMO_OBJ_IDS, targets=True),
        ),
        "test": (Source("lmo", "test", targets=True),),
    },
}

INT_COLUMNS = ("source", "scene_id", "im_id", "gt_id", "obj_id", "obj_idx")
# Per-source and per-object tables (not per row)
TABLE_KEYS = ("source_dirs", "source_exts", "obj_ids", "diameters", "symmetric")


def select_rows(index: dict, rows) -> dict:
    """Index restricted to the given rows (per-object and per-source tables are kept)."""
    rows = np.asarray(rows)
    n = len(index["source"])
    return {
        k: (v[rows] if v.ndim and len(v) == n and k not in TABLE_KEYS else v)
        for k, v in index.items()
    }


def rows_for_objects(index: dict, obj_ids) -> np.ndarray:
    """Rows of the given BOP object ids."""
    return np.flatnonzero(np.isin(index["obj_id"], np.asarray(obj_ids)))


class ObjectBalancedSampler(Sampler):
    """Virtual epochs with the same number of samples of every object.

    Each epoch draws samples_per_epoch // n_objects instances of every object, then shuffles them
    all. Every object walks through its own random permutations, so an instance comes back only
    after all instances of its object were seen. The order depends only on (seed, epoch): call
    set_epoch before each epoch, and a resumed run continues with the same samples.
    """

    def __init__(self, obj_idx, samples_per_epoch: int, seed: int = 0):
        obj_idx = np.asarray(obj_idx)
        self.groups = [np.flatnonzero(obj_idx == o) for o in np.unique(obj_idx)]
        self.per_object = samples_per_epoch // len(self.groups)
        if self.per_object < 1:
            raise ValueError(f"samples_per_epoch {samples_per_epoch} < {len(self.groups)} objects")
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.per_object * len(self.groups)

    def _draw(self, g: int, rows: np.ndarray) -> np.ndarray:
        """This epoch's slice of object g's endless sequence of permutations."""
        n, pos, parts = len(rows), self.epoch * self.per_object, []
        end = pos + self.per_object
        while pos < end:
            p, off = divmod(pos, n)
            take = min(n - off, end - pos)
            perm = np.random.default_rng((self.seed, 1 + g, p)).permutation(n)
            parts.append(rows[perm[off : off + take]])
            pos += take
        return np.concatenate(parts)

    def __iter__(self):
        idx = np.concatenate([self._draw(g, rows) for g, rows in enumerate(self.groups)])
        np.random.default_rng((self.seed, 0, self.epoch)).shuffle(idx)
        return iter(idx.tolist())


def default_bop_root() -> str:
    """BOP tree root: $BOP_PATH, else $NAS_HOME/datasets/bop."""
    if os.environ.get("BOP_PATH"):
        return os.environ["BOP_PATH"]
    return os.path.join(os.environ.get("NAS_HOME", "."), "datasets", "bop")


def default_cache_dir() -> str:
    """Index cache folder: $NAS_HOME/cache/bop_index."""
    return os.path.join(os.environ.get("NAS_HOME", "."), "cache", "bop_index")


def _load_json(path: str):
    with open(path) as f:
        return json.load(f)


def _rgb_extension(scene_dir: str) -> str:
    """Image extension of a scene ("jpg" for PBR, "png" for real test images)."""
    for name in sorted(os.listdir(os.path.join(scene_dir, "rgb"))):
        return name.rsplit(".", 1)[1]
    raise FileNotFoundError(f"No images in {scene_dir}/rgb")


def _valid_bbox(bbox) -> bool:
    return bbox[2] > 0 and bbox[3] > 0


def load_models_info(bop_root: str, dataset: str, obj_ids) -> dict:
    """Diameters (mm) and symmetry flags of the given objects, from models_eval/models_info.json."""
    info = _load_json(os.path.join(bop_root, dataset, "models_eval", "models_info.json"))
    return {
        "diameter": np.array([info[str(o)]["diameter"] for o in obj_ids], dtype=np.float64),
        "symmetric": np.array(
            [
                "symmetries_discrete" in info[str(o)] or "symmetries_continuous" in info[str(o)]
                for o in obj_ids
            ]
        ),
    }


def symmetry_table(
    bop_root: str, dataset: str, obj_ids, max_sym_disc_step: float = 0.01
) -> np.ndarray:
    """Rotations of every object's symmetry set, padded with the identity: (n_obj, K, 3, 3).

    From models_eval/models_info.json via bop_toolkit (the BOP19 set: identity first, continuous
    symmetries discretised at max_sym_disc_step of the diameter). An asymmetric object has the
    identity only. A repeated identity does not change a min over the set, so no mask is needed.
    The symmetries' translation offsets (< 1 mm on LM-O) are not kept.
    """
    from bop_toolkit_lib import misc  # only needed for symmetric training

    info = _load_json(os.path.join(bop_root, dataset, "models_eval", "models_info.json"))
    sets = [
        [s["R"] for s in misc.get_symmetry_transformations(info[str(o)], max_sym_disc_step)]
        for o in obj_ids
    ]
    table = np.tile(np.eye(3), (len(sets), max(len(s) for s in sets), 1, 1))
    for i, rots in enumerate(sets):
        table[i, : len(rots)] = rots
    return table


def build_index(
    bop_root: str, sources, obj_ids, models_dataset: str, min_visib_fract: float = 0.1
) -> dict:
    """Read the BOP annotations and return the index as a dict of numpy arrays.

    A row is one object instance (a ground-truth pose in one image):
    - Sources without targets keep the instances of the given objects with
      visib_fract >= min_visib_fract and a valid (non-empty) bbox_visib.
    - Sources with targets keep exactly the instances of the (scene, image, object) triples in
      test_targets_bop19.json, so the test index matches what the BOP evaluation scores.

    Args:
        bop_root: Root of the standard BOP tree.
        sources: Iterable of Source.
        obj_ids: BOP object ids to keep; obj_idx is the position in this list (the output head).
        models_dataset: Dataset whose models_info.json gives the diameters.
        min_visib_fract: Visibility threshold for sources without targets.

    Returns:
        Dict of arrays, one entry per row (see the keys below), plus per-source and per-object
        tables (source_dirs, source_exts, obj_ids, diameters, symmetric).
    """
    obj_ids = tuple(int(o) for o in obj_ids)
    obj_to_idx = {o: i for i, o in enumerate(obj_ids)}
    cols = {k: [] for k in INT_COLUMNS}
    cols.update(
        R=[], t=[], K=[], img_wh=[], bbox_obj=[], bbox_visib=[], visib_fract=[], px_count_visib=[]
    )
    source_dirs, source_exts = [], []

    for s_idx, src in enumerate(sources):
        split_dir = os.path.join(bop_root, src.dataset, src.split)
        camera = _load_json(os.path.join(bop_root, src.dataset, "camera.json"))
        img_wh = (camera["width"], camera["height"])
        scenes = src.scenes
        if scenes is None:
            scenes = tuple(int(d) for d in sorted(os.listdir(split_dir)) if d.isdigit())
        targets = None
        if src.targets:
            targets = {}
            for tgt in _load_json(os.path.join(bop_root, src.dataset, "test_targets_bop19.json")):
                if tgt["obj_id"] in obj_to_idx and tgt["scene_id"] in scenes:
                    key = (tgt["scene_id"], tgt["im_id"], tgt["obj_id"])
                    targets[key] = targets.get(key, 0) + tgt["inst_count"]
            scenes = tuple(sc for sc in scenes if any(k[0] == sc for k in targets))
        source_dirs.append(os.path.abspath(split_dir))
        source_exts.append(_rgb_extension(os.path.join(split_dir, f"{scenes[0]:06d}")))

        for scene_id in scenes:
            scene_dir = os.path.join(split_dir, f"{scene_id:06d}")
            scene_gt = _load_json(os.path.join(scene_dir, "scene_gt.json"))
            scene_gt_info = _load_json(os.path.join(scene_dir, "scene_gt_info.json"))
            scene_camera = _load_json(os.path.join(scene_dir, "scene_camera.json"))
            for im_key in sorted(scene_gt, key=int):
                im_id = int(im_key)
                for gt_id, (gt, info) in enumerate(zip(scene_gt[im_key], scene_gt_info[im_key])):
                    obj_id = gt["obj_id"]
                    if obj_id not in obj_to_idx:
                        continue
                    if targets is not None:
                        if (scene_id, im_id, obj_id) not in targets:
                            continue
                    elif info["visib_fract"] < min_visib_fract or not _valid_bbox(
                        info["bbox_visib"]
                    ):
                        continue
                    cols["source"].append(s_idx)
                    cols["scene_id"].append(scene_id)
                    cols["im_id"].append(im_id)
                    cols["gt_id"].append(gt_id)
                    cols["obj_id"].append(obj_id)
                    cols["obj_idx"].append(obj_to_idx[obj_id])
                    cols["R"].append(gt["cam_R_m2c"])
                    cols["t"].append(gt["cam_t_m2c"])
                    cols["K"].append(scene_camera[im_key]["cam_K"])
                    cols["img_wh"].append(img_wh)
                    cols["bbox_obj"].append(info["bbox_obj"])
                    cols["bbox_visib"].append(info["bbox_visib"])
                    cols["visib_fract"].append(info["visib_fract"])
                    cols["px_count_visib"].append(info["px_count_visib"])

        if targets is not None:
            # Every target instance must be found, otherwise the evaluation would count misses.
            found = {}
            for sc, im, ob, si in zip(
                cols["scene_id"], cols["im_id"], cols["obj_id"], cols["source"]
            ):
                if si == s_idx:
                    found[(sc, im, ob)] = found.get((sc, im, ob), 0) + 1
            missing = {k: n for k, n in targets.items() if found.get(k, 0) != n}
            if missing:
                raise ValueError(
                    f"{len(missing)} targets of {src} don't match scene_gt, e.g. {next(iter(missing))}"
                )

    n = len(cols["source"])
    index = {k: np.array(cols[k], dtype=np.int32).reshape(n) for k in INT_COLUMNS}
    index["R"] = np.array(cols["R"], dtype=np.float64).reshape(n, 3, 3)  # row-major cam_R_m2c
    index["t"] = np.array(cols["t"], dtype=np.float64).reshape(n, 3)  # mm
    index["K"] = np.array(cols["K"], dtype=np.float64).reshape(n, 3, 3)
    index["img_wh"] = np.array(cols["img_wh"], dtype=np.int32).reshape(n, 2)
    index["bbox_obj"] = np.array(cols["bbox_obj"], dtype=np.int32).reshape(n, 4)  # x, y, w, h
    index["bbox_visib"] = np.array(cols["bbox_visib"], dtype=np.int32).reshape(n, 4)
    index["visib_fract"] = np.array(cols["visib_fract"], dtype=np.float32).reshape(n)
    index["px_count_visib"] = np.array(cols["px_count_visib"], dtype=np.int32).reshape(n)
    index["source_dirs"] = np.array(source_dirs)
    index["source_exts"] = np.array(source_exts)
    index["obj_ids"] = np.array(obj_ids, dtype=np.int32)
    models = load_models_info(bop_root, models_dataset, obj_ids)
    index["diameters"] = models["diameter"]
    index["symmetric"] = models["symmetric"]
    return index


def index_key(dataset: str, split: str, bop_root: str, min_visib_fract: float) -> str:
    """Short hash of everything that defines an index, used in the cache file name."""
    cfg = SPLITS[dataset]
    spec = {
        "version": INDEX_VERSION,
        "bop_root": os.path.abspath(bop_root),
        "sources": [asdict(s) for s in cfg[split]],
        "obj_ids": cfg["obj_ids"],
        "models_dataset": cfg["models_dataset"],
        "min_visib_fract": min_visib_fract,
    }
    return hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:10]


def load_index(
    dataset: str,
    split: str,
    bop_root: str | None = None,
    cache_dir: str | None = None,
    min_visib_fract: float = 0.1,
    rebuild: bool = False,
) -> dict:
    """Return the index of a split, building and caching it on first use.

    Args:
        dataset: Target dataset in SPLITS, e.g. "lmo".
        split: "train", "val", "val_pbr", "val_real" or "test".
        bop_root: Root of the standard BOP tree (default: default_bop_root()).
        cache_dir: Where the .npz files live (default: default_cache_dir()). None of the source
            data is ever written to.
        min_visib_fract: Visibility threshold for training-type sources.
        rebuild: Ignore an existing cache file.

    Returns:
        Dict of numpy arrays, as returned by build_index.
    """
    bop_root = bop_root or default_bop_root()
    cache_dir = cache_dir or default_cache_dir()
    if split not in SPLITS[dataset] or split in ("obj_ids", "models_dataset"):
        raise ValueError(f"Unknown split {split!r} for {dataset}")
    path = os.path.join(
        cache_dir, f"{dataset}_{split}_{index_key(dataset, split, bop_root, min_visib_fract)}.npz"
    )
    if os.path.exists(path) and not rebuild:
        with np.load(path) as data:
            return {k: data[k] for k in data.files}

    start = time.time()
    cfg = SPLITS[dataset]
    index = build_index(
        bop_root, cfg[split], cfg["obj_ids"], cfg["models_dataset"], min_visib_fract
    )
    os.makedirs(cache_dir, exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "wb") as f:  # a file object keeps np.savez from renaming it
        np.savez(f, **index)
    os.replace(tmp_path, path)
    logging.info(
        f"Built {dataset}/{split} index: {len(index['source'])} rows in {time.time() - start:.0f} s -> {path}"
    )
    return index


def image_path(index: dict, row: int) -> str:
    """Path of the RGB image of an index row."""
    src = int(index["source"][row])
    return os.path.join(
        str(index["source_dirs"][src]),
        f"{int(index['scene_id'][row]):06d}",
        "rgb",
        f"{int(index['im_id'][row]):06d}.{index['source_exts'][src]}",
    )


def summarize(index: dict) -> str:
    """Human-readable counts: rows, images, rows per object."""
    n = len(index["source"])
    images = len(
        set(zip(index["source"].tolist(), index["scene_id"].tolist(), index["im_id"].tolist()))
    )
    per_obj = np.bincount(index["obj_idx"], minlength=len(index["obj_ids"]))
    per_obj_txt = ", ".join(
        f"{o}: {c}" for o, c in zip(index["obj_ids"].tolist(), per_obj.tolist())
    )
    return f"{n} rows from {images} images; per object {{{per_obj_txt}}}"


PIXEL_AUGS = ("none", "standard", "domain_gap")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def pixel_transform(pixel_aug: str) -> A.Compose:
    """Photometric augmentations + ImageNet normalization, for crops already at model size.

    "standard" and "domain_gap" are the SPEEDDataset pipelines without Resize (the crop warp
    already outputs the model size) and, for "domain_gap", without OpticalDistortion: it is a
    geometric warp, so it would make the pose labels wrong.
    """
    if pixel_aug not in PIXEL_AUGS:
        raise ValueError(f"pixel_aug must be one of {PIXEL_AUGS}, got {pixel_aug!r}")
    augs = []
    if pixel_aug == "standard":
        augs = [
            A.OneOf(
                [A.RandomBrightnessContrast(p=1.0), A.RandomGamma(p=1.0), A.CLAHE(p=1.0)], p=0.5
            ),
            A.OneOf(
                [
                    A.ImageCompression(quality_range=(40, 90), p=1.0),
                    A.Downscale(scale_range=(0.7, 0.95), p=1.0),
                ],
                p=0.5,
            ),
        ]
    elif pixel_aug == "domain_gap":
        augs = [
            A.ToGray(p=0.5),
            A.RandomToneCurve(p=0.5),
            A.RandomGamma(p=0.5),
            A.CLAHE(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Solarize(p=0.5),
            A.RandomSunFlare(flare_roi=(0, 0, 1, 1), p=0.5),
            A.ISONoise(p=0.5),
            A.MotionBlur(p=0.5),
            A.Defocus(p=0.5),
            A.OneOf([A.ImageCompression(p=1.0), A.Downscale(p=1.0)], p=0.5),
        ]
    return A.Compose(
        augs
        + [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=1.0), ToTensorV2()]
    )


def nearest_rotation(R: np.ndarray) -> np.ndarray:
    """Closest rotation matrix (Frobenius norm) to a 3x3 matrix, via SVD.

    LM-O test GT rotations are slightly non-orthonormal (|RRᵀ - I| up to 0.01). The quaternion
    conversion behind the labels would project them onto a rotation up to ~0.2° away from the
    nearest one; snapping first keeps labels and GT within 1e-6. PBR and LM GT change by ~1e-6.
    """
    U, _, Vt = np.linalg.svd(R)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def bop_bbox_to_xyxy(bbox) -> list[float]:
    """BOP [x, y, w, h] (first pixel, last - first) to [x1, y1, x2, y2] in K coordinates.

    In BOP, pixel i covers [i, i + 1) in the coordinates K projects to (its centre is at
    i + 0.5): projecting a model with its GT pose gives a box whose left/top edges match x, y and
    whose right/bottom edges match x + w + 1, y + h + 1 (measured on LM PBR and LM-O test, see
    tests/test_bop_dataset.py). So the box covers [x, x + w + 1] x [y, y + h + 1].
    """
    x, y, w, h = (float(v) for v in bbox)
    return [x, y, x + w + 1.0, y + h + 1.0]


def crop_warp_matrix(bbox, out_size: int) -> np.ndarray:
    """2x3 affine map (cv2 pixel indices) from the image to a square crop of side out_size.

    The square has side max(w, h) and the same centre as bbox, so the object keeps its aspect
    ratio (as SPEEDDataset.crop_image + padding + resize, in one interpolation step). bbox is in
    K coordinates, where image pixel i is centred at i + 0.5: point u maps to crop coordinate
    scale * (u - bbox centre) + out_size / 2, also with crop pixel j centred at j + 0.5.

    Args:
        bbox: [x1, y1, x2, y2] in K coordinates.
        out_size: Side of the output crop in pixels.
    """
    x1, y1, x2, y2 = (float(v) for v in bbox)
    scale = out_size / max(x2 - x1, y2 - y1)
    # pixel index i -> u = i + 0.5 -> crop coordinate -> crop index (- 0.5)
    tx = 0.5 * out_size - 0.5 + scale * (0.5 - 0.5 * (x1 + x2))
    ty = 0.5 * out_size - 0.5 + scale * (0.5 - 0.5 * (y1 + y2))
    return np.array([[scale, 0.0, tx], [0.0, scale, ty]], dtype=np.float64)


def crop_image(image: np.ndarray, bbox, out_size: int, context: bool = False) -> np.ndarray:
    """Square crop of bbox at out_size, as one warpAffine; zeros outside the image.

    Args:
        image: H x W x C image.
        bbox: [x1, y1, x2, y2] in K coordinates (see bop_bbox_to_xyxy).
        out_size: Output side in pixels.
        context: If False (default, as SPEEDDataset), the bands of the square outside bbox are
            black; if True they show the surrounding image.
    """
    M = crop_warp_matrix(bbox, out_size)
    crop = cv2.warpAffine(
        image,
        M,
        (out_size, out_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    if not context:
        x1, y1, x2, y2 = (float(v) for v in bbox)
        scale, half = M[0, 0], 0.5 * out_size
        # bbox spans [half - r, half + r] in crop coordinates; keep pixels whose centre is inside
        rx, ry = 0.5 * scale * (x2 - x1), 0.5 * scale * (y2 - y1)
        lo_x, hi_x = int(np.ceil(half - rx - 0.5)), int(np.floor(half + rx - 0.5))
        lo_y, hi_y = int(np.ceil(half - ry - 0.5)), int(np.floor(half + ry - 0.5))
        crop[:, : max(lo_x, 0)] = 0
        crop[:, hi_x + 1 :] = 0
        crop[: max(lo_y, 0)] = 0
        crop[hi_y + 1 :] = 0
    return crop


class BOPDataset(Dataset):
    """One sample per index row: the object's square crop and its pose labels.

    Returns a dict (src/engine.py unpack_batch) with:
    - images: (3, S, S) normalized crop.
    - translations: U = (U_x, U_y, U_z) relative to the crop bbox, U_z in `depth_mode`.
    - rotations: allocentric rotation R_relᵀ·R (3x3, or a quaternion), as SPEEDDataset.
    - bboxes: the crop bbox [x1, y1, x2, y2] in K coordinates (bop_bbox_to_xyxy, after
      jitter), needed to undo U.
    - K, img_wh, obj_idx, diameter: per-sample camera and object, for the inverse transforms.
    - R_m2c, t_m2c: the original BOP pose (egocentric, t in mm), and obj_id, scene_id, im_id,
      gt_id, row: to trace a sample back to its files.
    """

    def __init__(
        self,
        dataset: str = "lmo",
        split: str = "train",
        img_size: int = 384,
        rotation_format: str = "matrix",
        depth_mode: str = "site",
        bbox_source: str = "visib",
        augment: bool | None = None,
        bbox_jitter_percent: float = 10.0,
        bbox_jitter_prob: float = 0.5,
        pixel_aug: str = "standard",
        crop_context: bool = False,
        no_rotation_compensation: bool = False,
        index: dict | None = None,
        bop_root: str | None = None,
        cache_dir: str | None = None,
    ):
        """
        Args:
            dataset, split: Index to use (see SPLITS), unless `index` is given.
            img_size: Side of the square crop (the model input size).
            rotation_format: "matrix" or "quaternion".
            depth_mode: "site" (BOP default) or "legacy"; see src/utils.py.
            bbox_source: "visib" (bbox_visib, what a detector sees) or "obj" (bbox_obj).
            augment: Bbox jitter + pixel augmentations; default True for split "train" only.
            bbox_jitter_percent, bbox_jitter_prob: As SPEED's --bbox_crop_percent, applied with
                this probability (SPEED: 0.5).
            pixel_aug: "none", "standard" or "domain_gap" (used only when augmenting).
                "domain_gap" (SPEED+ lighting) makes many LM crops unrecognizable: ablate in M3.
            crop_context: Fill the square around the bbox with image instead of black.
            no_rotation_compensation: Regress the egocentric rotation instead.
            index: A prebuilt index (dict of arrays), e.g. for tests.
            bop_root, cache_dir: Passed to load_index.
        """
        if rotation_format not in ("matrix", "quaternion"):
            raise ValueError(
                f"rotation_format must be matrix or quaternion, got {rotation_format!r}"
            )
        if depth_mode not in DEPTH_MODES:
            raise ValueError(f"depth_mode must be one of {DEPTH_MODES}, got {depth_mode!r}")
        if bbox_source not in ("visib", "obj"):
            raise ValueError(f"bbox_source must be visib or obj, got {bbox_source!r}")
        self.index = (
            index if index is not None else load_index(dataset, split, bop_root, cache_dir)
        )
        self.img_size = img_size
        self.rotation_format = rotation_format
        self.depth_mode = depth_mode
        self.bbox_key = f"bbox_{bbox_source}"
        self.augment = (split == "train") if augment is None else augment
        self.bbox_jitter = bbox_jitter_percent / 100.0
        self.bbox_jitter_prob = bbox_jitter_prob
        self.crop_context = crop_context
        self.no_rotation_compensation = no_rotation_compensation
        self.transform = pixel_transform(pixel_aug if self.augment else "none")

    # The engine reads val_dataset.camera; BOP samples carry their own K, so it is only a fallback
    camera = Camera

    def __len__(self) -> int:
        return len(self.index["source"])

    def load_image(self, row: int) -> np.ndarray:
        """RGB uint8 image of a row."""
        path = image_path(self.index, row)
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def crop_bbox(self, row: int) -> list[float]:
        """The [x1, y1, x2, y2] bbox the crop and labels use, jittered when augmenting."""
        bbox = bop_bbox_to_xyxy(self.index[self.bbox_key][row])
        if self.augment and self.bbox_jitter > 0 and np.random.random() < self.bbox_jitter_prob:
            width, height = (int(v) for v in self.index["img_wh"][row])
            bbox = [
                float(v) for v in random_bbox_augmentation(bbox, width, height, self.bbox_jitter)
            ]
        return bbox

    def labels(self, row: int, bbox) -> tuple[torch.Tensor, torch.Tensor]:
        """(U, rotation) for a row and crop bbox, computed in float64."""
        R = torch.from_numpy(nearest_rotation(self.index["R"][row])).unsqueeze(0)
        t = torch.from_numpy(self.index["t"][row]).unsqueeze(0)
        K = torch.from_numpy(self.index["K"][row])
        U = translation_to_bbox_relative_translation(
            t,
            torch.tensor([bbox], dtype=torch.float64),
            Camera,
            K=K,
            img_wh=tuple(int(v) for v in self.index["img_wh"][row]),
            depth_mode=self.depth_mode,
            diameter=float(self.index["diameters"][self.index["obj_idx"][row]]),
        )
        q = get_apparent_orientation(
            t,
            rotation_matrix_to_quaternion(R),
            no_rotation_compensation=self.no_rotation_compensation,
        )
        rotation = quaternion_to_rotation_matrix(q) if self.rotation_format == "matrix" else q
        return U.squeeze(0), rotation.squeeze(0)

    def __getitem__(self, row: int) -> dict:
        index = self.index
        bbox = self.crop_bbox(row)
        crop = crop_image(self.load_image(row), bbox, self.img_size, self.crop_context)
        image = self.transform(image=crop.astype(np.float32) / 255.0)["image"]
        U, rotation = self.labels(row, bbox)
        obj_idx = int(index["obj_idx"][row])
        return {
            "images": image,
            "translations": U.float(),
            "rotations": rotation.float(),
            "bboxes": torch.tensor(bbox, dtype=torch.float32),
            "K": torch.from_numpy(index["K"][row]).float(),
            "img_wh": torch.from_numpy(index["img_wh"][row]).float(),
            "obj_idx": obj_idx,
            "diameter": torch.tensor(index["diameters"][obj_idx], dtype=torch.float32),
            "R_m2c": torch.from_numpy(index["R"][row]).float(),
            "t_m2c": torch.from_numpy(index["t"][row]).float(),
            "obj_id": int(index["obj_id"][row]),
            "scene_id": int(index["scene_id"][row]),
            "im_id": int(index["im_id"][row]),
            "gt_id": int(index["gt_id"][row]),
            "row": row,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build (or load) and summarize BOP index caches")
    parser.add_argument("--dataset", default="lmo", choices=sorted(SPLITS))
    parser.add_argument(
        "--splits", nargs="+", default=["train", "val_pbr", "val_real", "val", "test"]
    )
    parser.add_argument("--bop_root", default=None)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--min_visib_fract", type=float, default=0.1)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    for split in args.splits:
        start = time.time()
        idx = load_index(
            args.dataset, split, args.bop_root, args.cache_dir, args.min_visib_fract, args.rebuild
        )
        print(f"{args.dataset}/{split}: {summarize(idx)} ({time.time() - start:.1f} s)")
