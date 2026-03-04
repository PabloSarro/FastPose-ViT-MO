# This file contains the datasets used in this project

import json
import logging
import os
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.utils import (
    quaternion_to_rotation_matrix,
    translation_to_bbox_relative_translation,
    rotation_matrix_to_quaternion,
    get_apparent_orientation,
)
from src.camera import Camera
import cv2
import argparse

# Fix OpenCV threading conflicts with PyTorch DataLoader workers
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

# Global cache for annotations to avoid repeated JSON loading
_ANNOTATION_CACHE = {}


# Dataset configuration mapping
DATASET_CONFIGS = {
    "SPEED": {
        "folder": "SPEED_FIXED",
        "quaternion_field": "q_vbs2tango",
        "translation_field": "r_Vo2To_vbs_true",
        "bbox_annotations": "speed_bbox_annotations.json",
    },
    "SPEED_PLUS_SYNTHETIC": {
        "folder": "SPEED_PLUS_SYNTHETIC_FIXED",
        "quaternion_field": "q_vbs2tango_true",
        "translation_field": "r_Vo2To_vbs_true",
        "bbox_annotations": "speed_plus_synthetic_bbox_annotations.json",
    },
    "SPEED_PLUS": {
        "folder": "SPEED_PLUS_FIXED",
        "quaternion_field": "q_vbs2tango_true",
        "translation_field": "r_Vo2To_vbs_true",
        "bbox_annotations": "speed_plus_bbox_annotations.json",
    },
}


def get_dataset_config(dataset_name: str) -> dict:
    """
    Get dataset configuration for the specified dataset

    Args:
        dataset_name (str): Name of the dataset

    Returns:
        dict: Dataset configuration

    Raises:
        ValueError: If dataset name is not supported
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. Supported datasets: {list(DATASET_CONFIGS.keys())}"
        )
    return DATASET_CONFIGS[dataset_name]


class SPEEDDataset(Dataset):
    """
    Dataset class for the SPEED dataset
    """

    def __init__(
        self,
        dataset_root_dir: str,
        split: str = "train",
        rotation_format: str = "quaternion",
        img_size: tuple[int, int] = (224, 224),
        bbox_json_path: str = None,
        args: argparse.Namespace = None,
        dataset_name: str = "SPEED",
    ):
        """
        Initialize the dataset

        Args:
            dataset_root_dir (str): Path to the root directory of the dataset
            split (str): Dataset split to use (train, val, test)
            rotation_format (str): Format of the rotation representation (quaternion, matrix)
            img_size (tuple): Size of the images to use
            bbox_json_path (str): Path to the JSON file containing bounding box information
            args (argparse.Namespace): Arguments from the command line
            dataset_name (str): Name of the dataset to use for field mapping
        """
        self.dataset_root_dir = dataset_root_dir
        self.split = split
        self.rotation_format = rotation_format
        self.json_path = os.path.join(self.dataset_root_dir, f"{self.split}.json")
        self.bbox_json_path = bbox_json_path
        self.dataset_name = dataset_name

        # Get dataset configuration for field mapping
        self.dataset_config = get_dataset_config(dataset_name)

        # Cache frequently used values for better performance
        self.quaternion_field = self.dataset_config["quaternion_field"]
        self.translation_field = self.dataset_config["translation_field"]
        self.img_dir_path = os.path.join(self.dataset_root_dir, "images", self.split)
        self.is_train_split = split == "train"
        self.is_matrix_format = rotation_format == "matrix"

        # Cache Camera class reference to avoid repeated imports in __getitem__
        self.Camera = Camera

        # Pre-compute normalization tensors to avoid repeated tensor creation
        self._norm_mean = None
        self._norm_std = None

        # Pre-allocate tensor templates for hot path optimization
        self._tensor_template_4 = torch.zeros(4, dtype=torch.float32)
        self._tensor_template_3 = torch.zeros(3, dtype=torch.float32)

        # Pre-select transform pipeline to avoid conditional logic in __getitem__
        self._selected_transform = None
        self._use_style_aug_path = False

        # Experiments arguments with optimized attribute access
        if args:
            self.no_pixel_augmentation = args.no_pixel_augmentation
            self.no_spatial_augmentation = args.no_spatial_augmentation
            self.domain_gap_pixel_augmentation = args.domain_gap_pixel_augmentation
            self.do_style_aug = getattr(args, "do_style_aug", False)
            self.no_rotation_compensation = args.no_rotation_compensation
            self.use_absolute_translation = getattr(args, "use_absolute_translation", False)
            self.use_full_image = getattr(args, "use_full_image", False)
            self.no_bbox_augmentation = getattr(args, "no_bbox_augmentation", False)
            self.bbox_crop_percent = getattr(args, "bbox_crop_percent", 10.0)
            self.no_crop_padding = getattr(args, "no_crop_padding", False)
            self.convert_to_grayscale = getattr(args, "convert_to_grayscale", False)
        else:
            self.no_pixel_augmentation = False
            self.no_spatial_augmentation = False
            self.domain_gap_pixel_augmentation = False
            self.do_style_aug = False
            self.no_rotation_compensation = False
            self.use_absolute_translation = False
            self.use_full_image = False
            self.no_bbox_augmentation = False
            self.bbox_crop_percent = 10.0
            self.no_crop_padding = False
            self.convert_to_grayscale = False

        self.args = args

        # Pre-compute boolean combinations to avoid repeated calculations
        self._needs_spatial_aug = self.is_train_split and not self.no_spatial_augmentation
        self._has_pixel_aug = self.is_train_split and not self.no_pixel_augmentation

        # Set up camera data class with caching optimizations
        self.camera = Camera()

        # Cache camera constants for performance
        self._camera_fx = self.camera.fx
        self._camera_fy = self.camera.fy
        self._camera_cx = self.camera.cx
        self._camera_cy = self.camera.cy

        # Initialize spatial transformer
        self.spatial_transformer = SpatialDataAugmenter(
            camera=self.camera,
            no_bbox_augmentation=self.no_bbox_augmentation,
            bbox_crop_percent=self.bbox_crop_percent,
        )

        # Set up image transformations
        # Basic resize and normalization
        self.base_transform = A.Compose(
            [
                A.Resize(height=img_size[0], width=img_size[1], interpolation=cv2.INTER_AREA),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=1.0
                ),
                ToTensorV2(),
            ]
        )

        # Standard augmentations
        self.standard_augment_transform = A.Compose(
            [
                A.Resize(height=img_size[0], width=img_size[1], interpolation=cv2.INTER_AREA),
                A.OneOf(
                    [
                        A.RandomBrightnessContrast(p=1.0),
                        A.RandomGamma(p=1.0),
                        A.CLAHE(p=1.0),
                    ],
                    p=0.5,
                ),
                A.OneOf(
                    [
                        A.ImageCompression(quality_range=(40, 90), p=1.0),
                        A.Downscale(scale_range=(0.7, 0.95), p=1.0),
                    ],
                    p=0.5,
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=1.0
                ),
                ToTensorV2(),
            ]
        )

        # Domain gap augmentations (for style aug: returns unnormalized)
        self.domain_gap_pre_style = A.Compose(
            [
                A.Resize(height=img_size[0], width=img_size[1], interpolation=cv2.INTER_AREA),
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

        # Domain gap augmentations (complete with normalization)
        self.domain_gap_transform = A.Compose(
            [
                A.Resize(height=img_size[0], width=img_size[1], interpolation=cv2.INTER_AREA),
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
                A.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=1.0
                ),
                ToTensorV2(),
            ]
        )

        # Pre-select transform pipeline based on training configuration
        self._select_transform_pipeline()

        # Load only the file paths and labels, not the actual images with caching
        raw_data = self._load_annotations_cached()
        self.data = self._normalize_annotation_data(raw_data)
        if not self.data:
            raise ValueError(f"No annotations with 'filename' found in {self.json_path}.")

        # Pre-compute full image paths to avoid os.path.join in __getitem__
        self._precompute_image_paths()

        if self.bbox_json_path is not None:
            self.bbox_data = self._load_bbox_annotations_cached()
            self.has_bbox = True
        else:
            self.bbox_data = None
            self.has_bbox = False

    def _process_bbox(self, filename: str) -> list[float] | None:
        """Process bounding box from dict/list to standardized list format.

        Args:
            filename: Name of the image file to get bounding box for.

        Returns:
            Bounding box coordinates as [x1, y1, x2, y2] or None if not available.

        Raises:
            ValueError: If bbox_info has an unexpected type.
        """
        if not self.has_bbox:
            return None

        bbox_info = self.bbox_data.get(filename)
        if bbox_info is None:
            return None

        if isinstance(bbox_info, dict):
            return [bbox_info["x1"], bbox_info["y1"], bbox_info["x2"], bbox_info["y2"]]
        elif isinstance(bbox_info, (list, tuple)):
            return list(bbox_info)
        else:
            raise ValueError(f"Unexpected bbox_info type: {type(bbox_info)} for {filename}")

    def _precompute_image_paths(self) -> None:
        """Pre-compute full image paths to avoid os.path.join in __getitem__."""
        self._image_paths = {}
        for item in self.data:
            filename = item["filename"]
            self._image_paths[filename] = os.path.join(self.img_dir_path, filename)

    def _normalize_annotation_data(self, data: dict | list) -> list[dict]:
        """Recursively normalize annotation data to a flat list of dicts with filenames.

        Args:
            data: Annotation data which can be a dict or list of annotations.

        Returns:
            Flat list of annotation dictionaries, each containing a 'filename' key.
        """
        if isinstance(data, dict):
            if "filename" in data:
                return [data]
            normalized = []
            for value in data.values():
                normalized.extend(self._normalize_annotation_data(value))
            return normalized
        if isinstance(data, list):
            normalized = []
            for item in data:
                normalized.extend(self._normalize_annotation_data(item))
            return normalized
        # Ignore unsupported primitive types quietly (e.g., metadata integers)
        return []

    def _select_transform_pipeline(self) -> None:
        """Pre-select the appropriate transform pipeline to avoid runtime conditionals."""
        if self._has_pixel_aug:
            if self.domain_gap_pixel_augmentation:
                if self.do_style_aug:
                    # Style augmentation path - use pre-style transforms
                    self._selected_transform = self.domain_gap_pre_style
                    self._use_style_aug_path = True
                else:
                    # Domain gap without style aug
                    self._selected_transform = self.domain_gap_transform
                    self._use_style_aug_path = False
            else:
                # Standard augmentations
                self._selected_transform = self.standard_augment_transform
                self._use_style_aug_path = False
        else:
            # No augmentation - basic transform only
            self._selected_transform = self.base_transform
            self._use_style_aug_path = False

    def __len__(self) -> int:
        """
        Get the length of the dataset

        Args:
            None

        Returns:
            int: Length of the dataset
        """
        return len(self.data)

    def crop_image(self, image: np.ndarray, bbox: list) -> np.ndarray:
        """
        Crop image according to bbox coordinates

        Args:
            image (np.ndarray): Image to crop (H x W x C)
            bbox (list): Bounding box coordinates [x1, y1, x2, y2]

        Returns:
            np.ndarray: Cropped image
        """
        # Convert to integers
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        cropped_image = image[y1:y2, x1:x2]

        # Pad the image to be square (only if padding is not disabled)
        if not self.no_crop_padding:
            h, w = cropped_image.shape[:2]
            if h != w:
                if h > w:
                    pad = (h - w) // 2
                    cropped_image = np.pad(
                        cropped_image, ((0, 0), (pad, pad), (0, 0)), mode="constant"
                    )
                else:
                    pad = (w - h) // 2
                    cropped_image = np.pad(
                        cropped_image, ((pad, pad), (0, 0), (0, 0)), mode="constant"
                    )

        return cropped_image

    def __getitem__(self, idx: int) -> tuple:
        """
        Get an item from the dataset (simplified and optimized)

        Args:
            idx (int): Index of the item to retrieve

        Returns:
            tuple: Image, translation, rotation, bbox
        """
        item = self.data[idx]
        filename = item["filename"]

        # Load image using pre-computed path
        image = self._load_image_optimized(self._image_paths[filename])

        # Convert pose data on-demand
        quaternion = torch.tensor(item[self.quaternion_field], dtype=torch.float32)
        translation = torch.tensor(item[self.translation_field], dtype=torch.float32)

        # Fix quaternion orientation (use batch format with size 1)
        quaternion = get_apparent_orientation(
            translation=translation.unsqueeze(0),
            centered_rotation_quat=quaternion.unsqueeze(0),
            no_rotation_compensation=self.no_rotation_compensation,
        ).squeeze(0)

        # Get bbox using optimized conversion method
        bbox = self._process_bbox(filename)

        # Apply spatial augmentations
        if self._needs_spatial_aug:
            image, bbox, translation, rotation = self.spatial_transformer(
                image, bbox, translation, quaternion
            )
        else:
            rotation = quaternion

        # Process bbox if it exists
        if self.has_bbox and bbox is not None:
            # Crop image only if not using full image mode
            if not self.use_full_image:
                image = self.crop_image(image, bbox)
            # Convert bbox to tensor for translation computation
            bbox_tensor = torch.tensor(bbox, dtype=torch.float32)
            # Convert to relative translation unless using absolute mode
            if not self.use_absolute_translation:
                translation = translation_to_bbox_relative_translation(
                    translation.unsqueeze(0),
                    bbox_tensor.unsqueeze(0),
                    self.Camera,
                ).squeeze(0)
            # Keep final bbox as tensor
            bbox = bbox_tensor
        else:
            bbox = torch.zeros(4, dtype=torch.float32)

        # Convert rotation format if needed
        if self.is_matrix_format:
            rotation = quaternion_to_rotation_matrix(rotation.unsqueeze(0)).squeeze(0)

        # Apply pre-selected transform pipeline
        image = self._selected_transform(image=image)["image"]

        # Handle style augmentation special case
        if self._use_style_aug_path:
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            return image, translation, rotation, bbox

        return image, translation, rotation, bbox

    def finalize_style_augmented_batch(self, image_batch: torch.Tensor) -> torch.Tensor:
        """
        Apply final ImageNet normalization to batched style-augmented image tensors

        Args:
            image_batch: Batch of unnormalized image tensors [B, C, H, W] in range [0, 1]

        Returns:
            Batch of normalized image tensors ready for ViT model
        """
        # Cache normalization tensors on first use
        if self._norm_mean is None or self._norm_mean.device != image_batch.device:
            self._norm_mean = torch.tensor([0.485, 0.456, 0.406], device=image_batch.device).view(
                1, 3, 1, 1
            )
            self._norm_std = torch.tensor([0.229, 0.224, 0.225], device=image_batch.device).view(
                1, 3, 1, 1
            )

        # Normalize: (x - mean) / std
        normalized_batch = (image_batch - self._norm_mean) / self._norm_std

        return normalized_batch

    def _load_annotations_cached(self) -> list[dict]:
        """Load annotations from JSON file with caching.

        Returns:
            List of annotation dictionaries loaded from the JSON file.
        """
        cache_key = str(Path(self.json_path).resolve())
        file_mtime = Path(self.json_path).stat().st_mtime

        # Check cache
        if cache_key in _ANNOTATION_CACHE:
            cached_data, cached_mtime = _ANNOTATION_CACHE[cache_key]
            if cached_mtime == file_mtime:
                return cached_data

        # Load annotations
        with open(self.json_path, "r") as f:
            data = json.load(f)

        # Cache the data
        _ANNOTATION_CACHE[cache_key] = (data, file_mtime)
        return data

    def _load_bbox_annotations_cached(self) -> dict[str, dict | list]:
        """Load bounding box annotations from JSON file with caching.

        Returns:
            Dictionary mapping filenames to bounding box data.
        """
        cache_key = str(Path(self.bbox_json_path).resolve())
        file_mtime = Path(self.bbox_json_path).stat().st_mtime

        # Check cache
        if cache_key in _ANNOTATION_CACHE:
            cached_data, cached_mtime = _ANNOTATION_CACHE[cache_key]
            if cached_mtime == file_mtime:
                return cached_data

        # Load bbox annotations
        with open(self.bbox_json_path, "r") as f:
            data = json.load(f)

        # Cache the data
        _ANNOTATION_CACHE[cache_key] = (data, file_mtime)
        return data

    def _load_image_optimized(self, image_path: str) -> np.ndarray:
        """Load and preprocess image with optional grayscale conversion.

        Args:
            image_path: Path to the image file.

        Returns:
            Image array as float32 with values in [0, 1] range.

        Raises:
            FileNotFoundError: If the image file cannot be loaded.
        """
        if self.convert_to_grayscale:
            # Optimized: Load directly as grayscale to avoid unnecessary RGB conversion
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(f"Could not load image: {image_path}")
            # Replicate to 3 channels for model compatibility
            image = np.stack([image, image, image], axis=-1)
        else:
            # Standard RGB loading path
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"Could not load image: {image_path}")
            # Convert to RGB efficiently
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image.astype(np.float32) / 255.0  # Scale to [0, 1]


class SpatialDataAugmenter:
    """Applies spatial augmentations to images and adjusts pose labels accordingly."""

    def __init__(
        self, camera: Camera, no_bbox_augmentation: bool = False, bbox_crop_percent: float = 10.0
    ) -> None:
        self.camera = camera
        self.no_bbox_augmentation = no_bbox_augmentation
        # Convert percentage to decimal (e.g., 10.0% -> 0.1)
        self.max_crop_percent = bbox_crop_percent / 100.0
        self.spatial_transform = A.ReplayCompose(
            [
                A.OneOf(
                    [
                        A.Rotate(limit=(-20, 20), border_mode=cv2.BORDER_CONSTANT, fill=0, p=0.5),
                        A.Rotate(limit=(160, 200), border_mode=cv2.BORDER_CONSTANT, fill=0, p=0.5),
                    ],
                    p=0.5,
                ),
            ],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
        )
        self.spatial_transform_no_bbox = A.ReplayCompose(
            [
                A.OneOf(
                    [
                        A.Rotate(limit=(-20, 20), border_mode=cv2.BORDER_CONSTANT, fill=0, p=0.5),
                        A.Rotate(limit=(160, 200), border_mode=cv2.BORDER_CONSTANT, fill=0, p=0.5),
                    ],
                    p=0.5,
                ),
            ]
        )

    def __call__(
        self,
        image: np.ndarray,
        bbox: list[float] | None,
        translation: torch.Tensor,
        rotation: torch.Tensor,
    ) -> tuple[np.ndarray, list[float] | None, torch.Tensor, torch.Tensor]:
        """Apply spatial augmentation to image and adjust pose labels.

        Args:
            image: Input image array (H, W, C).
            bbox: Bounding box coordinates [x1, y1, x2, y2] or None.
            translation: Translation vector tensor.
            rotation: Rotation quaternion tensor.

        Returns:
            Tuple of (transformed_image, transformed_bbox, adjusted_translation, adjusted_rotation).
        """
        # First apply random bbox cropping (50% chance) if enabled
        if not self.no_bbox_augmentation:
            if np.random.random() < 0.5:
                bbox = random_bbox_augmentation(
                    bbox, image.shape[1], image.shape[0], self.max_crop_percent
                )

        # Prepare bbox and label for albumentations format
        if bbox is not None:
            bbox_list = [[bbox[0], bbox[1], bbox[2], bbox[3]]]
            labels = ["satellite"]

            # Apply spatial transforms and get transform parameters
            transformed = self.spatial_transform(image=image, bboxes=bbox_list, labels=labels)

            # Get transformed image and bbox
            transformed_image = transformed["image"]
            # Handle case where spatial transforms filter out the bbox
            if len(transformed["bboxes"]) > 0:
                transformed_bbox = transformed["bboxes"][0]
            else:
                # If bbox was filtered out, return original bbox
                transformed_bbox = bbox_list[0]
        else:
            transformed = self.spatial_transform_no_bbox(image=image)
            transformed_image = transformed["image"]
            transformed_bbox = None

        # Check if replay exists
        if "replay" not in transformed:
            return transformed_image, transformed_bbox, translation, rotation

        # Process each transform's parameters
        for transform in transformed["replay"]["transforms"]:
            if transform["applied"]:
                # Handle OneOf transforms
                if "transforms" in transform:
                    for one_of_transform in transform["transforms"]:
                        if one_of_transform["applied"]:
                            name = one_of_transform.get("__class_fullname__", "")
                            if "Rotate" in name and one_of_transform["params"] is not None:
                                translation, rotation = self._adjust_rotation(
                                    translation, rotation, one_of_transform["params"]
                                )
                # Something handled wrong
                else:
                    logging.error("Error when processing the spatial augmentation")

        return transformed_image, transformed_bbox, translation, rotation

    def _adjust_transformation(
        self, translation: torch.Tensor, rotation_quat: torch.Tensor, transform: dict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Adjust translation and rotation for spatial augmentations.

        Args:
            translation: Translation vector tensor.
            rotation_quat: Rotation quaternion tensor.
            transform: Dictionary containing transformation matrix from albumentations.

        Returns:
            Tuple of (adjusted_translation, adjusted_rotation).
        """
        # Extract transformation matrix from replay
        if "matrix" not in transform:
            return translation, rotation_quat

        # Get the transformation matrix (O) applied by Albumentations
        matrix = torch.tensor(transform["matrix"], dtype=torch.float32)

        # Compute adjusted rotation matrix
        R_change = self.camera.K_inv @ matrix @ self.camera.K
        # The relative orientation does not change for ShiftScaleRotate
        rotation = rotation_quat

        # Compute adjusted translation vector
        translation = R_change @ translation
        return translation, rotation

    def _adjust_rotation(
        self, translation: torch.Tensor, rotation_quat: torch.Tensor, transform: dict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Adjust translation and rotation for rotation augmentations.

        Args:
            translation: Translation vector tensor.
            rotation_quat: Rotation quaternion tensor.
            transform: Dictionary containing rotation matrix from albumentations.

        Returns:
            Tuple of (adjusted_translation, adjusted_rotation).
        """
        # Get rotation matrix from the albumentations transformation
        M = torch.tensor(transform["matrix"], dtype=torch.float32)
        rot_mat_orig = quaternion_to_rotation_matrix(rotation_quat.unsqueeze(0)).squeeze(0)
        R_change = self.camera.K_inv @ M @ self.camera.K
        matrix_rotation = R_change @ rot_mat_orig
        rotation = rotation_matrix_to_quaternion(matrix_rotation)

        translation = R_change @ translation
        return translation, rotation


def random_bbox_augmentation(
    bbox: list[float], img_width: int, img_height: int, max_crop_percent: float = 0.2
) -> list[float]:
    """Apply random cropping/expansion to a bounding box.

    Args:
        bbox: Bounding box coordinates [x1, y1, x2, y2].
        img_width: Image width in pixels.
        img_height: Image height in pixels.
        max_crop_percent: Maximum percentage of bbox dimension to crop/expand.

    Returns:
        Augmented bounding box coordinates [x1, y1, x2, y2].
    """
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1

    # Generate all random values in one call for better performance
    crop_range_w = max_crop_percent * width
    crop_range_h = max_crop_percent * height
    crops = np.random.uniform(-1, 1, 4)  # Single call for 4 values

    # Apply crops/expansions using vectorized operations
    new_x1 = x1 + crops[0] * crop_range_w
    new_x2 = x2 - crops[1] * crop_range_w
    new_y1 = y1 + crops[2] * crop_range_h
    new_y2 = y2 - crops[3] * crop_range_h

    # Clamp to image bounds using numpy operations
    new_x1 = np.clip(new_x1, 0, img_width)
    new_y1 = np.clip(new_y1, 0, img_height)
    new_x2 = np.clip(new_x2, 0, img_width)
    new_y2 = np.clip(new_y2, 0, img_height)

    # Fast validity check with early return
    new_width = new_x2 - new_x1
    new_height = new_y2 - new_y1
    if new_width < width * 0.5 or new_height < height * 0.5 or new_width <= 0 or new_height <= 0:
        return [x1, y1, x2, y2]

    return [new_x1, new_y1, new_x2, new_y2]
