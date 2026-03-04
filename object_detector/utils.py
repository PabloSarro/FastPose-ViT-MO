# This file contains utility functions for the object detection module

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont

from object_detector.constants import EPSILON, FOUR_OVER_PI_SQUARED
from object_detector.LWDETR.lwdetr_utils import NestedTensor

# Optimized tensor creation with device-specific caching
_tensor_cache = {}


def get_cached_tensor(
    value: float, device: torch.device, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Get or create cached tensor for common values to reduce repeated tensor creation"""
    key = (value, str(device), dtype)
    if key not in _tensor_cache:
        _tensor_cache[key] = torch.tensor(value, device=device, dtype=dtype)
    return _tensor_cache[key]


def create_half_tensor(device: torch.device) -> torch.Tensor:
    """Create half tensor compatible with torch.compile.

    Args:
        device: Device to create the tensor on.

    Returns:
        Tensor containing the value 0.5.
    """
    return get_cached_tensor(0.5, device)


def create_epsilon_tensor(device: torch.device) -> torch.Tensor:
    """Create epsilon tensor compatible with torch.compile.

    Args:
        device: Device to create the tensor on.

    Returns:
        Tensor containing the EPSILON constant value.
    """
    return get_cached_tensor(EPSILON, device)


def create_four_over_pi_sq_tensor(device: torch.device) -> torch.Tensor:
    """Create four_over_pi_sq tensor compatible with torch.compile.

    Args:
        device: Device to create the tensor on.

    Returns:
        Tensor containing the FOUR_OVER_PI_SQUARED constant value.
    """
    return get_cached_tensor(FOUR_OVER_PI_SQUARED, device)


def setup_logging(
    args: argparse.Namespace | None = None,
    initialized: bool = False,
    rank: int = 0,
    log_file: str | None = None,
    base_filename: str = "train",
) -> tuple[logging.Logger, str]:
    """Set up logging for the application.

    Args:
        args: Command line arguments with log_dir attribute. If None, uses basic logging.
        initialized: Whether the logger has already been initialized.
        rank: Rank of the process in the distributed environment.
        log_file: Path to the log file.
        base_filename: Base name for the log file.

    Returns:
        Tuple of (Logger object, path to the log file).
    """
    # For backward compatibility - if no args provided, use simple logging
    if args is None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        return logging.getLogger(__name__), ""

    # Create log directory if it doesn't exist
    if not initialized:
        log_dir = getattr(args, "log_dir", "./runs")
        os.makedirs(log_dir, exist_ok=True)

        # Create a unique log file name
        log_file = f"{log_dir}/{base_filename}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"

        # Create a logger
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(f"Rank{rank}")
        log_file = log_file

    # Clear any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.INFO)
    # Prevent the logger from propagating messages to the root logger
    logger.propagate = False

    # Create handlers
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()

    # Set level for handlers
    file_handler.setLevel(logging.INFO)
    console_handler.setLevel(logging.INFO)

    # Create a formatting configuration
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Set formatter for handlers
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger, log_file


def clamp_bbox_coordinates(
    bbox_normalized: torch.Tensor | list[float], tensor_format: bool = False
) -> torch.Tensor | list[float]:
    """Clamp bounding box coordinates to [0,1] range.

    Optimized vectorized version that ensures boxes stay within valid bounds.

    Args:
        bbox_normalized: Bounding box as [center_x, center_y, width, height],
            potentially outside [0,1] range.
        tensor_format: If True, input/output is torch.Tensor, else list/tuple.

    Returns:
        Clamped bounding box [center_x, center_y, width, height] in valid range.
    """
    if tensor_format:
        # Optimized PyTorch tensor version for batch processing
        original_shape = bbox_normalized.shape
        if bbox_normalized.dim() == 1:
            bbox_normalized = bbox_normalized.unsqueeze(0)

        # Create torch.compile compatible tensors
        half_tensor = create_half_tensor(bbox_normalized.device)

        # Vectorized clamping - clamp all coordinates to [0,1] first
        clamped_boxes = torch.clamp(bbox_normalized, 0.0, 1.0)

        # Extract coordinates efficiently
        cx, cy, w, h = clamped_boxes.unbind(-1)

        # Vectorized center adjustment using cached half tensor
        half_w = w * half_tensor
        half_h = h * half_tensor

        # Adjust centers to ensure bbox stays within bounds (vectorized)
        cx = torch.clamp(cx, half_w, 1.0 - half_w)
        cy = torch.clamp(cy, half_h, 1.0 - half_h)

        # Reconstruct with original shape
        result = torch.stack([cx, cy, w, h], dim=-1)
        if len(original_shape) == 1:
            result = result.squeeze(0)
        return result
    else:
        # Optimized Python version with fewer operations
        cx, cy, w, h = bbox_normalized

        # Vectorized clamping approach
        w = max(0.0, min(1.0, w))
        h = max(0.0, min(1.0, h))

        # Optimized center clamping
        half_w, half_h = w * 0.5, h * 0.5
        cx = max(half_w, min(1.0 - half_w, cx))
        cy = max(half_h, min(1.0 - half_h, cy))

        return [cx, cy, w, h]


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert bounding boxes from cxcywh to xyxy format (optimized with cached constants)

    Args:
        boxes: Tensor of shape (N, 4) in cxcywh format

    Returns:
        Tensor of shape (N, 4) in xyxy format
    """
    # Create torch.compile compatible tensor
    half_tensor = create_half_tensor(boxes.device)

    # Optimized vectorized conversion
    cx, cy, w, h = boxes.unbind(-1)
    half_w = w * half_tensor
    half_h = h * half_tensor

    # More efficient coordinate computation
    x1 = cx - half_w
    y1 = cy - half_h
    x2 = cx + half_w
    y2 = cy + half_h

    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute standard IoU between two sets of boxes with batched input support

    Args:
        boxes1: Tensor of shape (N, 4) or (B, N, 4) in xyxy format
        boxes2: Tensor of shape (M, 4) or (B, M, 4) in xyxy format

    Returns:
        Tensor of shape (N, M) or (B, N, M) with IoU values in range [0, 1]
    """
    # Use constant epsilon for JIT compatibility
    epsilon_val = 1e-7

    # Detect if inputs are batched
    is_batched = boxes1.dim() == 3 and boxes2.dim() == 3

    if is_batched:
        # Batched computation: (B, N, 4) and (B, M, 4) -> (B, N, M)

        # Pre-compute areas for efficiency (batched)
        area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])  # [B, N]
        area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])  # [B, M]

        # Compute intersection with proper batched broadcasting
        # boxes1: [B, N, 4] -> [B, N, 1, 4], boxes2: [B, M, 4] -> [B, 1, M, 4]
        boxes1_expanded = boxes1.unsqueeze(2)  # [B, N, 1, 4]
        boxes2_expanded = boxes2.unsqueeze(1)  # [B, 1, M, 4]

        # Extract coordinates for intersection computation
        lt = torch.maximum(boxes1_expanded[..., :2], boxes2_expanded[..., :2])  # [B, N, M, 2]
        rb = torch.minimum(boxes1_expanded[..., 2:], boxes2_expanded[..., 2:])  # [B, N, M, 2]

        # More efficient intersection computation
        wh = (rb - lt).clamp_min(0)  # [B, N, M, 2]
        inter = wh.prod(dim=-1)  # [B, N, M]

        # Vectorized union computation with proper broadcasting
        union = area1.unsqueeze(2) + area2.unsqueeze(1) - inter  # [B, N, M]

        # Compute IoU
        iou = inter / (union + epsilon_val)

        return iou
    else:
        # Regular computation: (N, 4) and (M, 4) -> (N, M)
        # Pre-compute areas for efficiency (vectorized)
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # [N]
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # [M]

        # Compute intersection (optimized broadcasting)
        lt = torch.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
        rb = torch.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

        # More efficient intersection computation
        wh = (rb - lt).clamp_min(0)  # [N,M,2]
        inter = wh.prod(dim=2)  # [N,M] - use prod instead of manual multiplication

        # Vectorized union computation
        union = area1.unsqueeze(1) + area2.unsqueeze(0) - inter  # [N,M]

        # Optimized IoU computation with constant epsilon
        iou = inter / (union + epsilon_val)

        return iou


def smooth_loss(loss_values: List[float], window_size: int = 100) -> float:
    """
    Compute smoothed loss over a window (optimized with early returns)

    Args:
        loss_values: List of loss values
        window_size: Window size for smoothing

    Returns:
        Smoothed loss value
    """
    if not loss_values:  # More efficient empty check
        return 0.0

    # Optimized windowing with slice operation
    window = loss_values[-window_size:] if len(loss_values) > window_size else loss_values

    # More efficient mean calculation
    return sum(window) / len(window) if window else 0.0


def create_output_dir(output_dir: str) -> Path:
    """
    Create output directory for saving models and logs

    Args:
        output_dir: Output directory path

    Returns:
        Path object
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model.

    Args:
        model: PyTorch model to count parameters for.

    Returns:
        Total number of trainable parameters.
    """
    param_counts = torch.tensor([p.numel() for p in model.parameters()])
    trainable_mask = torch.tensor([p.requires_grad for p in model.parameters()])
    return param_counts[trainable_mask].sum().item()


def format_number(num: int) -> str:
    """Format large numbers with appropriate suffixes.

    Args:
        num: Number to format.

    Returns:
        Formatted string with K/M suffix for thousands/millions.
    """
    # Use integer division for better performance
    if num >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.1f}K"
    else:
        return str(num)


def scale_boxes_to_original(
    boxes: torch.Tensor, original_size: Tuple[int, int], model_size: Tuple[int, int] = (384, 384)
) -> torch.Tensor:
    """
    Scale normalized bounding boxes to original image size (optimized vectorized version)

    Args:
        boxes: Tensor of shape (N, 4) with normalized boxes in cxcywh format [0, 1]
        original_size: (width, height) of original image
        model_size: (width, height) of model input size

    Returns:
        Tensor of shape (N, 4) with boxes scaled to original image coordinates
    """
    if boxes.numel() == 0:
        return boxes

    # Optimized scaling with vectorized operations
    orig_w, orig_h = original_size

    # Create scale factors tensor for efficient broadcasting
    scale_factors = torch.tensor(
        [orig_w, orig_h, orig_w, orig_h], device=boxes.device, dtype=boxes.dtype
    )

    # Vectorized scaling operation (no clone needed, creates new tensor)
    scaled_boxes = boxes * scale_factors

    return scaled_boxes


def postprocess_predictions(
    predictions: dict[str, torch.Tensor],
    original_sizes: list[tuple[int, int]],
    model_size: tuple[int, int] = (384, 384),
) -> list[dict[str, torch.Tensor]]:
    """Post-process model predictions to original image coordinates.

    Performs vectorized batch processing to scale predictions.

    Args:
        predictions: Dict with 'scores' and 'boxes' tensors from model.
        original_sizes: List of (width, height) for each image in batch.
        model_size: Model input size (width, height).

    Returns:
        List of dicts with 'boxes', 'scores', 'labels' for each image.
    """
    scores = predictions["scores"]  # [B, num_queries]
    boxes = predictions["boxes"]  # [B, num_queries, 4]
    batch_size = scores.shape[0]

    # Vectorized processing where possible
    processed = []

    # Convert original_sizes to tensor for vectorized operations when needed
    orig_sizes_tensor = torch.tensor(original_sizes, device=boxes.device, dtype=torch.float32)

    for i in range(batch_size):
        # Get predictions for this image (keep as tensor slicing for performance)
        image_scores = scores[i]  # (num_queries,)
        image_boxes = boxes[i]  # (num_queries, 4)

        # Optimized valid mask computation
        valid_mask = image_scores > 0  # Boolean mask more efficient than using scores directly

        if not valid_mask.any():
            # No valid detections - use pre-allocated empty tensors
            processed.append(
                {
                    "boxes": torch.empty((0, 4), dtype=boxes.dtype, device=boxes.device),
                    "scores": torch.empty((0,), dtype=scores.dtype, device=scores.device),
                    "labels": torch.empty((0,), dtype=torch.long, device=boxes.device),
                }
            )
            continue

        # Vectorized filtering
        valid_boxes = image_boxes[valid_mask]
        valid_scores = image_scores[valid_mask]

        # Pre-allocate labels tensor
        valid_labels = torch.zeros(valid_scores.shape[0], dtype=torch.long, device=boxes.device)

        # Optimized scaling using tensor operations
        orig_w, orig_h = orig_sizes_tensor[i]
        scale_factors = torch.tensor([orig_w, orig_h, orig_w, orig_h], device=boxes.device)
        scaled_boxes = valid_boxes * scale_factors

        processed.append({"boxes": scaled_boxes, "scores": valid_scores, "labels": valid_labels})

    return processed


def visualize_predictions(
    image_path: str,
    predictions: dict[str, torch.Tensor],
    original_size: tuple[int, int],
    model_size: tuple[int, int] = (384, 384),
    save_path: str | None = None,
) -> None:
    """Visualize predictions on original image.

    Args:
        image_path: Path to original image.
        predictions: Model predictions dict with 'boxes' and 'scores'.
        original_size: (width, height) of original image.
        model_size: Model input size (width, height).
        save_path: Optional path to save visualization. If None, displays image.
    """
    # Load original image
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    # Optimized tensor extraction
    pred_boxes = predictions["boxes"][0]  # [1, 4] or [4]
    pred_scores = predictions["scores"][0]  # [1] or scalar

    # Ensure consistent shape with minimal operations
    if pred_boxes.dim() == 1:
        pred_boxes = pred_boxes.unsqueeze(0)  # [1, 4]
    if pred_scores.dim() == 0:
        pred_scores = pred_scores.unsqueeze(0)  # [1]

    # Optimized scaling
    scaled_boxes = scale_boxes_to_original(pred_boxes, original_size, model_size)

    # Cache font loading to avoid repeated I/O
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        font = ImageFont.load_default()

    # Optimized box drawing with vectorized coordinate conversion
    boxes_list = scaled_boxes.cpu().tolist()  # Single CPU transfer
    scores_list = pred_scores.cpu().tolist()  # Single CPU transfer

    for box, score in zip(boxes_list, scores_list):
        # Optimized coordinate conversion
        cx, cy, w, h = box
        half_w, half_h = w * 0.5, h * 0.5
        x1, y1 = cx - half_w, cy - half_h
        x2, y2 = cx + half_w, cy + half_h

        # Draw rectangle and text
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1 - 20), f"{score:.3f}", fill="red", font=font)

    # Optimized save/display
    if save_path:
        image.save(save_path)
        logging.info(f"Visualization saved to {save_path}")
    else:
        image.show()


def convert_to_nested_tensor(images: torch.Tensor) -> NestedTensor:
    """Convert regular tensor batch to NestedTensor format for LW-DETR.

    Args:
        images: Batch of images tensor of shape [B, C, H, W].

    Returns:
        NestedTensor with image data and attention masks.
    """
    if isinstance(images, NestedTensor):
        return images

    # Convert tensor batch to NestedTensor
    # For uniform batches (same size images), we can create simple masks
    batch_size, channels, height, width = images.shape

    # Create mask for valid pixels (all True for uniform batches)
    masks = torch.zeros((batch_size, height, width), dtype=torch.bool, device=images.device)

    # Create NestedTensor
    return NestedTensor(images, masks)


def is_nested_tensor_input(model: torch.nn.Module) -> bool:
    """Check if model requires NestedTensor input.

    Args:
        model: Model to check.

    Returns:
        True if model is LW-DETR variant requiring NestedTensor input.
    """
    # Simple heuristic: check if model has LW-DETR in its class name
    model_class_name = model.__class__.__name__.lower()
    return "lwdetr" in model_class_name or "nested" in model_class_name


def prepare_model_input(
    images: torch.Tensor | NestedTensor, model: torch.nn.Module | None = None
) -> torch.Tensor | NestedTensor:
    """Prepare model input by converting to appropriate format.

    For LW-DETR we get best performance by passing a NestedTensor directly.
    torch.compile can obscure class names, so instead of trying to detect the
    exact model type, we unconditionally create a NestedTensor here. This is
    safe for our object_detector pipeline, which only uses LW-DETR.

    Args:
        images: Input images as tensor or NestedTensor.
        model: Optional model (unused, kept for API compatibility).

    Returns:
        NestedTensor suitable for LW-DETR input.
    """
    if isinstance(images, torch.Tensor):
        return convert_to_nested_tensor(images)
    return images
