# This file contains the datasets used in this project

import json
import logging
import mmap
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import time
from object_detector.constants import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    MAX_ANNOTATION_CACHE_SIZE_MB,
)
from PIL import Image

# No StyleAugmentor import needed - handled at training level like in src/

# Fix OpenCV threading conflicts with PyTorch DataLoader workers
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

# Global cache for annotations to avoid repeated JSON loading
_ANNOTATION_CACHE = {}


class SPEEDDataset(Dataset):
    """
    Simplified SPEED Dataset for satellite detection

    Expects JSON format:
    {
        "img007040.jpg": {
            "x1": 969, "y1": 555, "x2": 1153, "y2": 716
        }
    }
    """

    def __init__(
        self,
        images_dir: str,
        annotations_file: str,
        img_size: Tuple[int, int] = (384, 384),
        split: str = "train",
        no_pixel_augmentation: bool = False,
        no_spatial_augmentation: bool = False,
        domain_gap_pixel_augmentation: bool = False,
        do_style_aug: bool = False,
        enable_caching: bool = True,
        convert_to_grayscale: bool = False,
    ):
        """
        Initialize SPEED dataset

        Args:
            images_dir: Directory containing images
            annotations_file: Path to JSON annotations file
            img_size: Target image size (height, width)
            split: Dataset split ("train" or "val")
            no_pixel_augmentation: Don't use pixel data augmentation
            no_spatial_augmentation: Don't use spatial data augmentation
            domain_gap_pixel_augmentation: Use domain gap pixel augmentation with sun flare, blur, noise, compression, and dropout
            do_style_aug: Enable neural style augmentation (requires domain_gap_pixel_augmentation)
            enable_caching: Enable annotation caching for faster loading
            convert_to_grayscale: Convert images to grayscale (replicated as 3-channel for model compatibility)
        """
        # Handle directory structure
        if Path(images_dir).name == "images" and split is not None:
            self.images_dir = Path(images_dir) / split
        else:
            self.images_dir = Path(images_dir)

        self.annotations_file = Path(annotations_file)
        self.img_size = img_size
        self.split = split
        self.no_pixel_augmentation = no_pixel_augmentation
        self.no_spatial_augmentation = no_spatial_augmentation
        self.domain_gap_pixel_augmentation = domain_gap_pixel_augmentation
        self.do_style_aug = do_style_aug
        self.enable_caching = enable_caching

        # Remove redundant tensor pre-allocation - use direct tensor creation for better memory efficiency

        # Cache frequently used values for better performance
        self.is_train_split = split == "train"
        self.convert_to_grayscale = convert_to_grayscale

        # Pre-cache constants for bbox conversion optimization
        self.half_constant = 0.5
        self.zero_constant = 0.0
        self.one_constant = 1.0

        # Pre-compute normalization tensors to avoid repeated tensor creation
        self._norm_mean = None
        self._norm_std = None

        # Pre-compute boolean combinations to avoid repeated calculations
        self._needs_spatial_aug = self.is_train_split and not self.no_spatial_augmentation
        self._has_pixel_aug = self.is_train_split and not self.no_pixel_augmentation

        # Load annotations with caching
        self.annotations = self._load_annotations_cached()

        # Get valid image files
        self.image_files = self._get_valid_images()

        # Style augmentation handled at training level - no initialization needed here

        # Pre-select transform pipeline to avoid runtime conditionals (like src/)
        self._selected_transform = None
        self._use_style_aug_path = False
        self.train_transform_pre_style = None
        self.normalize_transform = None
        self._select_transform_pipeline()

        self.val_transform = self._get_val_transforms()

        # Pre-warm augmentation transforms for GPU power stability
        self._prewarm_transforms()

        logging.info(f"SPEED Dataset ({split}): {len(self.image_files)} images")
        if self.enable_caching:
            logging.info("Annotation caching enabled for faster loading")

    def _prewarm_transforms(self) -> None:
        """Pre-warm augmentation transforms to reduce initial GPU power spikes.

        Runs the transform pipeline multiple times with dummy data to stabilize
        GPU power consumption and warm up any JIT compilation.
        """
        if self._selected_transform:
            # Create dummy image for prewarming
            dummy_image = np.random.rand(self.img_size[0], self.img_size[1], 3).astype(np.float32)
            dummy_bbox = [0.4, 0.4, 0.6, 0.6]  # Normalized YOLO format

            try:
                # Pre-warm the selected transform pipeline
                for _ in range(3):  # Multiple runs to stabilize
                    if self._use_style_aug_path:
                        _ = self._selected_transform(
                            image=dummy_image, bboxes=[dummy_bbox], labels=["satellite"]
                        )
                        if hasattr(self, "normalize_transform") and self.normalize_transform:
                            _ = self.normalize_transform(
                                image=dummy_image, bboxes=[dummy_bbox], labels=["satellite"]
                            )
                    else:
                        _ = self._selected_transform(
                            image=dummy_image, bboxes=[dummy_bbox], labels=["satellite"]
                        )
                logging.info("Augmentation transforms pre-warmed for power stability")
            except Exception as e:
                logging.warning(f"Transform pre-warming failed: {e}")

    def _select_transform_pipeline(self) -> None:
        """Pre-select the appropriate transform pipeline to avoid runtime conditionals.

        Determines which augmentation pipeline to use based on configuration
        and caches the selection for efficient __getitem__ calls.
        """
        if self._has_pixel_aug:
            if self.domain_gap_pixel_augmentation and self.do_style_aug:
                # Style augmentation path - use pre-style transforms (no normalization)
                self._selected_transform = self._get_train_transforms_pre_style()
                self.normalize_transform = self._get_normalize_transform()
                self._use_style_aug_path = True
            elif self.domain_gap_pixel_augmentation:
                # Domain gap without style aug
                self._selected_transform = self._get_train_transforms()
                self._use_style_aug_path = False
            else:
                # Standard augmentations
                self._selected_transform = self._get_train_transforms()
                self._use_style_aug_path = False
        else:
            # No augmentation - basic transform only
            self._selected_transform = self._get_val_transforms()
            self._use_style_aug_path = False

    def _load_image_optimized(self, image_path: Path) -> np.ndarray:
        """Load image with optimized path for grayscale/RGB.

        Args:
            image_path: Path to the image file.

        Returns:
            Loaded image as float32 numpy array normalized to [0, 1] range.

        Raises:
            Exception: If image loading fails.
        """
        try:
            if self.convert_to_grayscale:
                # Load as grayscale and replicate to 3 channels
                image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    image = np.array(Image.open(image_path).convert("L"))
                image = np.stack([image, image, image], axis=-1)
            else:
                # Standard RGB loading
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    image = np.array(Image.open(image_path).convert("RGB"))
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            return image.astype(np.float32) / 255.0
        except Exception as e:
            logging.error(f"Failed to load image {image_path}: {e}")
            raise

    def _load_annotations_cached(self) -> dict:
        """Load annotations with caching for better performance.

        Uses a global cache to avoid repeated JSON loading for the same
        annotation file across multiple dataset instances.

        Returns:
            Dictionary mapping image filenames to bounding box annotations.
        """
        if not self.enable_caching:
            return self._load_annotations()

        # Create cache key based on file path and modification time
        cache_key = str(self.annotations_file.resolve())
        file_mtime = self.annotations_file.stat().st_mtime

        # Check if we have a valid cached version
        if cache_key in _ANNOTATION_CACHE:
            cached_data, cached_mtime = _ANNOTATION_CACHE[cache_key]
            if cached_mtime == file_mtime:
                logging.info(f"Using cached annotations for {self.annotations_file.name}")
                return self._filter_annotations_by_split(cached_data)

        # Load and cache annotations
        start_time = time.time()
        all_annotations = self._load_annotations_raw()
        _ANNOTATION_CACHE[cache_key] = (all_annotations, file_mtime)

        # Filter for current split
        filtered_annotations = self._filter_annotations_by_split(all_annotations)
        load_time = time.time() - start_time
        logging.info(f"Loaded and cached annotations in {load_time:.2f}s")
        return filtered_annotations

    def _load_annotations_raw(self) -> dict:
        """Load raw annotations without split filtering.

        Handles large annotation files by using memory-mapped loading when
        file size exceeds the configured threshold.

        Returns:
            Dictionary of all annotations from the JSON file.
        """
        # For large annotation files, consider memory-mapped loading
        file_size = self.annotations_file.stat().st_size
        threshold_bytes = MAX_ANNOTATION_CACHE_SIZE_MB * 1024 * 1024

        if file_size > threshold_bytes:
            logging.info(
                f"Large annotation file detected ({file_size / 1024 / 1024:.1f}MB), using memory-mapped loading"
            )
            return self._load_annotations_mmap()
        else:
            with open(self.annotations_file, "r") as f:
                return json.load(f)

    def _load_annotations_mmap(self) -> dict:
        """Load annotations using memory mapping for large files.

        Returns:
            Dictionary of annotations loaded via memory-mapped file access.
        """
        try:
            with open(self.annotations_file, "r") as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmapped_file:
                    return json.loads(mmapped_file.read().decode("utf-8"))
        except Exception as e:
            logging.warning(f"Memory-mapped loading failed: {e}, falling back to standard loading")
            with open(self.annotations_file, "r") as f:
                return json.load(f)

    def _filter_annotations_by_split(self, all_annotations: dict) -> dict:
        """Filter annotations by split.

        Args:
            all_annotations: Complete annotation dictionary.

        Returns:
            Filtered dictionary containing only annotations for the current split.
        """
        if self.split is None:
            return all_annotations

        # Load train/val split information
        dataset_root = self.annotations_file.parent
        split_file = dataset_root / f"{self.split}.json"

        if split_file.exists():
            with open(split_file, "r") as f:
                split_data = json.load(f)

            # Extract just the filenames from the split data
            if isinstance(split_data, list) and len(split_data) > 0:
                if isinstance(split_data[0], dict) and "filename" in split_data[0]:
                    split_images = {item["filename"] for item in split_data}
                else:
                    split_images = set(split_data)
            else:
                split_images = set()

            # Filter annotations to only include images in this split
            filtered_annotations = {
                img_name: bbox
                for img_name, bbox in all_annotations.items()
                if img_name in split_images
            }

            logging.info(
                f"Filtered {len(filtered_annotations)} annotations for {self.split} split"
            )
            return filtered_annotations
        else:
            logging.info(f"Warning: Split file {split_file} not found, using all annotations")
            return all_annotations

    def _load_annotations(self) -> dict:
        """Load annotations from JSON file and filter by split.

        Returns:
            Dictionary mapping image filenames to bounding box annotations.

        Raises:
            FileNotFoundError: If annotations file does not exist.
            ValueError: If annotations file contains invalid JSON.
        """
        try:
            # Load all annotations
            with open(self.annotations_file, "r") as f:
                all_annotations = json.load(f)

            # If split is None, return all annotations
            if self.split is None:
                logging.info(f"Loaded {len(all_annotations)} annotations (no split)")
                return all_annotations

            # Load train/val split information
            dataset_root = self.annotations_file.parent
            split_file = dataset_root / f"{self.split}.json"

            if split_file.exists():
                # Load the specific split file to get the image list
                with open(split_file, "r") as f:
                    split_data = json.load(f)

                # Extract just the filenames from the split data
                if isinstance(split_data, list) and len(split_data) > 0:
                    # Handle SPEED format where each entry has a "filename" field
                    if isinstance(split_data[0], dict) and "filename" in split_data[0]:
                        split_images = {item["filename"] for item in split_data}
                    else:
                        # Handle simple list of filenames
                        split_images = set(split_data)
                else:
                    split_images = set()

                # Filter annotations to only include images in this split
                filtered_annotations = {
                    img_name: bbox
                    for img_name, bbox in all_annotations.items()
                    if img_name in split_images
                }

                logging.info(
                    f"Loaded {len(filtered_annotations)} annotations for {self.split} split"
                )
                return filtered_annotations
            else:
                logging.info(f"Warning: Split file {split_file} not found, using all annotations")
                return all_annotations

        except FileNotFoundError:
            raise FileNotFoundError(f"Annotations file not found: {self.annotations_file}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in annotations file: {e}")

    def _get_valid_images(self) -> list[str]:
        """Get list of images that exist and have annotations.

        Returns:
            List of valid image filenames that have both annotations and exist on disk.
        """
        valid_images = []
        for image_name in self.annotations.keys():
            image_path = self.images_dir / image_name
            if image_path.exists():
                valid_images.append(image_name)

        return valid_images

    def _get_train_transforms(self) -> A.Compose:
        """Build optimized combined training transforms in single pipeline.

        Returns:
            Albumentations Compose object with training augmentations.
        """
        transforms = []

        # Add spatial augmentations if enabled (BEFORE resize to ensure consistent output size)
        if not self.no_spatial_augmentation:
            transforms.extend(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    # Removed RandomScale as it causes different output sizes
                ]
            )

        # Always resize AFTER spatial transforms to ensure consistent output size
        transforms.append(A.Resize(height=self.img_size[0], width=self.img_size[1]))

        # Add pixel augmentations based on type
        if self.domain_gap_pixel_augmentation and not self.do_style_aug:
            # Domain gap augmentation pipeline
            transforms.extend(
                [
                    # global tone & LF illumination
                    A.ToGray(p=0.5),
                    A.RandomToneCurve(p=0.5),
                    A.RandomGamma(p=0.5),
                    A.CLAHE(p=0.5),
                    # lens effects
                    A.OpticalDistortion(p=0.5),
                    # clipped highlights / glare
                    A.RandomBrightnessContrast(p=0.5),
                    A.Solarize(p=0.5),
                    A.RandomSunFlare(flare_roi=(0, 0, 1, 1), p=0.5),
                    # realistic noise/blur
                    A.ISONoise(p=0.5),
                    A.MotionBlur(p=0.5),
                    A.Defocus(p=0.5),
                    # compression + downscale
                    A.OneOf(
                        [
                            A.ImageCompression(p=1.0),
                            A.Downscale(p=1.0),
                        ],
                        p=0.5,
                    ),
                ]
            )
        elif not self.no_pixel_augmentation:
            # Standard pixel augmentations (removed expensive HueSaturationValue)
            transforms.append(
                A.OneOf(
                    [
                        A.RandomBrightnessContrast(
                            brightness_limit=0.2, contrast_limit=0.2, p=1.0
                        ),
                        A.RandomGamma(gamma_limit=(80, 120), p=1.0),
                    ],
                    p=0.3,
                )
            )

        # Always add normalization and tensor conversion
        transforms.extend(
            [
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=1.0),
                ToTensorV2(),
            ]
        )

        return A.Compose(
            transforms, bbox_params=A.BboxParams(format="yolo", label_fields=["labels"])
        )

    def _get_train_transforms_pre_style(self) -> A.Compose:
        """Build training transforms before style augmentation (no normalization).

        Returns:
            Albumentations Compose object without final normalization step.
        """
        transforms = []

        # Add spatial augmentations if enabled (BEFORE resize to ensure consistent output size)
        if not self.no_spatial_augmentation:
            transforms.extend(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                ]
            )

        # Always resize AFTER spatial transforms to ensure consistent output size
        transforms.append(A.Resize(height=self.img_size[0], width=self.img_size[1]))

        # Domain gap augmentations (no normalization for style aug pipeline)
        if self.domain_gap_pixel_augmentation:
            transforms.extend(
                [
                    A.RandomSunFlare(
                        flare_roi=(0, 0, 1, 0.5),
                        angle_range=(0, 1),
                        num_flare_circles_range=(6, 10),
                        src_radius=400,
                        p=0.75,
                    ),
                    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
                    A.RandomGamma(gamma_limit=(80, 120), p=0.5),
                    A.OneOf(
                        [
                            A.MotionBlur(p=1.0),
                            A.GaussianBlur(p=1.0),
                            A.GaussNoise(p=1.0),
                            A.ISONoise(p=1.0),
                        ],
                        p=0.5,
                    ),
                    A.ImageCompression(quality_range=(75, 100), p=0.5),
                    A.CoarseDropout(
                        num_holes_range=(1, 16),
                        hole_height_range=(0.05, 0.2),
                        hole_width_range=(0.05, 0.2),
                        p=0.3,
                    ),
                ]
            )

        return A.Compose(
            transforms, bbox_params=A.BboxParams(format="yolo", label_fields=["labels"])
        )

    def _get_normalize_transform(self) -> A.Compose:
        """Build final normalization and tensor conversion transform.

        Returns:
            Albumentations Compose object for ImageNet normalization and tensor conversion.
        """
        return A.Compose(
            [
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=1.0),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="yolo", label_fields=["labels"]),
        )

    def _get_val_transforms(self) -> A.Compose:
        """Build simple validation transforms without augmentation.

        Returns:
            Albumentations Compose object with resize, normalize, and tensor conversion.
        """
        return A.Compose(
            [
                A.Resize(height=self.img_size[0], width=self.img_size[1]),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=1.0),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="yolo", label_fields=["labels"]),
        )

    def _convert_bbox_to_yolo(
        self, bbox: dict, inv_orig_width: float, inv_orig_height: float
    ) -> list[float]:
        """Convert bbox from pixel coordinates to normalized YOLO format.

        Args:
            bbox: Dictionary with x1, y1, x2, y2 keys in pixel coordinates.
            inv_orig_width: Precomputed 1.0 / original_width for efficiency.
            inv_orig_height: Precomputed 1.0 / original_height for efficiency.

        Returns:
            List of [center_x, center_y, width, height] in normalized [0, 1] range.
        """
        # Extract coordinates as numpy array for vectorized operations
        coords = np.array([bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]], dtype=np.float32)
        x1, y1, x2, y2 = coords

        # Vectorized conversion to center_x, center_y, width, height format
        center_x = (x1 + x2) * self.half_constant * inv_orig_width
        center_y = (y1 + y2) * self.half_constant * inv_orig_height
        width = (x2 - x1) * inv_orig_width
        height = (y2 - y1) * inv_orig_height

        # Vectorized clamping to [0, 1] range
        result = np.array([center_x, center_y, width, height], dtype=np.float32)
        result = np.clip(result, self.zero_constant, self.one_constant)

        return result.tolist()

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample from the dataset.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Tuple of (image_tensor, bbox_tensor) where image is [C, H, W]
            and bbox is [4] in YOLO format (cx, cy, w, h).
        """
        image_name = self.image_files[idx]
        image_path = self.images_dir / image_name

        # Load image with optimized I/O
        image = self._load_image_optimized(image_path)
        orig_height, orig_width = image.shape[:2]

        # Precompute inverse dimensions to avoid division in bbox conversion
        inv_orig_width = 1.0 / orig_width
        inv_orig_height = 1.0 / orig_height

        # Get annotation and convert to YOLO format with optimized conversion
        bbox_data = self.annotations[image_name]
        bbox_yolo = self._convert_bbox_to_yolo(bbox_data, inv_orig_width, inv_orig_height)

        # Apply pre-selected transform pipeline using cached split check
        if self.is_train_split:
            if self._use_style_aug_path:
                # Style augmentation pipeline (like src/)
                # Step 1: Apply pre-style transforms
                transformed = self._selected_transform(
                    image=image, bboxes=[bbox_yolo], labels=["satellite"]
                )
                image = transformed["image"]

                # Style augmentation will be applied at training level (like in src/)

                # Step 3: Apply final normalization and tensor conversion
                transformed = self.normalize_transform(
                    image=image, bboxes=transformed["bboxes"], labels=transformed["labels"]
                )
                image = transformed["image"]
            else:
                # Standard transform pipeline
                transformed = self._selected_transform(
                    image=image, bboxes=[bbox_yolo], labels=["satellite"]
                )
                image = transformed["image"]
        else:
            transformed = self.val_transform(image=image, bboxes=[bbox_yolo], labels=["satellite"])
            image = transformed["image"]

        # Optimized bbox handling with direct tensor creation
        if len(transformed["bboxes"]) > 0:
            final_bbox = torch.as_tensor(transformed["bboxes"][0], dtype=torch.float32)
        else:
            final_bbox = torch.as_tensor(bbox_yolo, dtype=torch.float32)

        return image, final_bbox

    def finalize_style_augmented_batch(self, image_batch: torch.Tensor) -> torch.Tensor:
        """
        Apply final ImageNet normalization to batched style-augmented image tensors (like src/)

        Args:
            image_batch: Batch of unnormalized image tensors [B, C, H, W] in range [0, 1]

        Returns:
            Batch of normalized image tensors ready for model
        """
        # Cache normalization tensors on first use
        if self._norm_mean is None or self._norm_mean.device != image_batch.device:
            self._norm_mean = torch.tensor(IMAGENET_MEAN, device=image_batch.device).view(
                1, 3, 1, 1
            )
            self._norm_std = torch.tensor(IMAGENET_STD, device=image_batch.device).view(1, 3, 1, 1)

        # Normalize: (x - mean) / std
        normalized_batch = (image_batch - self._norm_mean) / self._norm_std

        return normalized_batch


def create_dataloaders(
    images_dir: str,
    annotations_file: str,
    batch_size: int = 8,
    img_size: Tuple[int, int] = (384, 384),
    num_workers: int = 4,
    no_pixel_augmentation: bool = False,
    no_spatial_augmentation: bool = False,
    domain_gap_pixel_augmentation: bool = False,
    do_style_aug: bool = False,
    enable_caching: bool = True,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    convert_to_grayscale: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders

    Args:
        images_dir: Directory containing images
        annotations_file: Path to JSON annotations file
        batch_size: Batch size for training
        img_size: Target image size
        num_workers: Number of data loading workers
        no_pixel_augmentation: Don't use pixel data augmentation
        no_spatial_augmentation: Don't use spatial data augmentation
        domain_gap_pixel_augmentation: Use domain gap pixel augmentation with sun flare, blur, noise, compression, and dropout
        do_style_aug: Enable neural style augmentation (requires domain_gap_pixel_augmentation)
        enable_caching: Enable annotation caching for faster loading
        prefetch_factor: How many batches to prefetch per worker
        persistent_workers: Keep workers alive between epochs
        convert_to_grayscale: Convert images to grayscale (replicated as 3-channel for model compatibility)

    Returns:
        train_loader, val_loader
    """

    # Create datasets with optimized I/O
    train_dataset = SPEEDDataset(
        images_dir=images_dir,
        annotations_file=annotations_file,
        img_size=img_size,
        split="train",
        no_pixel_augmentation=no_pixel_augmentation,
        no_spatial_augmentation=no_spatial_augmentation,
        domain_gap_pixel_augmentation=domain_gap_pixel_augmentation,
        do_style_aug=do_style_aug,
        enable_caching=enable_caching,
        convert_to_grayscale=convert_to_grayscale,
    )

    val_dataset = SPEEDDataset(
        images_dir=images_dir,
        annotations_file=annotations_file,
        img_size=img_size,
        split="val",
        no_pixel_augmentation=True,  # Always disable augmentation for validation
        no_spatial_augmentation=True,
        domain_gap_pixel_augmentation=False,  # Always disable domain gap augmentation for validation
        do_style_aug=False,  # Always disable style augmentation for validation
        enable_caching=enable_caching,
        convert_to_grayscale=convert_to_grayscale,
    )

    # Check if validation dataset is empty
    if len(val_dataset) == 0:
        logging.info("Warning: Validation split is empty, creating split from all data")
        raise ValueError("Empty validation split")
    if len(train_dataset) == 0:
        logging.info("Warning: Training split is empty, creating split from all data")
        raise ValueError("Empty training split")

    # Optimize DataLoader settings for stable GPU utilization and faster loading
    logging.info(
        f"DataLoader optimization: Using {num_workers} workers for stable GPU utilization"
    )

    # Create optimized dataloaders with src/-style settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,  # Keep pin_memory for performance
        persistent_workers=persistent_workers,  # Use parameter from function
        prefetch_factor=prefetch_factor
        if num_workers > 0
        else None,  # Use parameter from function
        drop_last=True,
        multiprocessing_context="fork" if num_workers > 0 else None,  # Use fork (faster on Linux)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,  # Keep pin_memory for performance
        persistent_workers=persistent_workers,  # Use parameter from function
        prefetch_factor=prefetch_factor
        if num_workers > 0
        else None,  # Use parameter from function
        multiprocessing_context="fork" if num_workers > 0 else None,  # Use fork (faster on Linux)
    )

    logging.info(
        f"DataLoader config: {num_workers} workers, prefetch={prefetch_factor}, persistent={persistent_workers}, context=fork"
    )

    return train_loader, val_loader


def verify_dataset(images_dir: str, annotations_file: str, split: str = "train") -> bool:
    """Verify that a dataset is properly configured and accessible.

    Args:
        images_dir: Directory containing images (may be parent with split subdirs).
        annotations_file: Path to JSON annotations file.
        split: Dataset split to verify ("train", "val", or "test").

    Returns:
        True if all annotated images exist, False otherwise.
    """
    try:
        base_images_dir = Path(images_dir)
        annotations_file = Path(annotations_file)

        # Determine the actual images directory for this split
        if base_images_dir.name == "images":
            actual_images_dir = base_images_dir / split
        else:
            actual_images_dir = base_images_dir

        # Load all annotations
        with open(annotations_file, "r") as f:
            all_annotations = json.load(f)

        # Load split-specific annotations
        dataset_root = annotations_file.parent
        split_file = dataset_root / f"{split}.json"

        if split_file.exists():
            # Load the specific split file to get the image list
            with open(split_file, "r") as f:
                split_data = json.load(f)

            # Extract just the filenames from the split data
            if isinstance(split_data, list) and len(split_data) > 0:
                # Handle SPEED format where each entry has a "filename" field
                if isinstance(split_data[0], dict) and "filename" in split_data[0]:
                    split_images = {item["filename"] for item in split_data}
                else:
                    # Handle simple list of filenames
                    split_images = set(split_data)
            else:
                split_images = set()

            # Filter annotations to only include images in this split
            annotations = {
                img_name: bbox
                for img_name, bbox in all_annotations.items()
                if img_name in split_images
            }
            logging.info(f"Using {len(annotations)} annotations for {split} split")
        else:
            logging.info(f"Warning: Split file {split_file} not found, using all annotations")
            annotations = all_annotations

        # Check if images exist in the correct directory
        missing_images = []
        for image_name in annotations.keys():
            image_path = actual_images_dir / image_name
            if not image_path.exists():
                missing_images.append(image_name)

        if missing_images:
            logging.info(f"Warning: {len(missing_images)} images not found in {actual_images_dir}")
            logging.info(f"First few missing: {missing_images[:5]}")

        valid_images = len(annotations) - len(missing_images)
        logging.info(
            f"Dataset verification: {valid_images}/{len(annotations)} images found in {actual_images_dir}"
        )

        return len(missing_images) == 0

    except Exception as e:
        logging.info(f"Dataset verification failed: {e}")
        return False
