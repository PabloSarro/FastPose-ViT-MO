# This file contains the BOP dataset index: one row per object instance used for training or evaluation
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

import numpy as np

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
