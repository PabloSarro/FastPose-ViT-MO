# This file contains the visualization script for RF-DETR predictions

import json
import argparse
import logging
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
from object_detector.utils import clamp_bbox_coordinates


def load_predictions_json(json_path: str) -> dict:
    """Load predictions JSON file.

    Args:
        json_path: Path to the predictions JSON file.

    Returns:
        Parsed JSON data as a dictionary.
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def denormalize_bbox(
    bbox_normalized: list[float], img_width: int, img_height: int, clamp: bool = True
) -> list[int]:
    """Convert normalized [cx, cy, w, h] to pixel [x1, y1, x2, y2].

    Args:
        bbox_normalized: Bounding box as [center_x, center_y, width, height] in [0,1] range.
        img_width: Image width in pixels.
        img_height: Image height in pixels.
        clamp: Whether to clamp coordinates to valid range before conversion.

    Returns:
        Bounding box as [x1, y1, x2, y2] in pixel coordinates.
    """
    if clamp:
        bbox_normalized = clamp_bbox_coordinates(bbox_normalized)

    cx, cy, w, h = bbox_normalized

    # Convert to pixel coordinates
    cx_px = cx * img_width
    cy_px = cy * img_height
    w_px = w * img_width
    h_px = h * img_height

    # Convert center+size to corner coordinates
    x1 = int(cx_px - w_px / 2)
    y1 = int(cy_px - h_px / 2)
    x2 = int(cx_px + w_px / 2)
    y2 = int(cy_px + h_px / 2)

    # Additional safety clamping for pixel coordinates
    x1 = max(0, min(img_width - 1, x1))
    y1 = max(0, min(img_height - 1, y1))
    x2 = max(0, min(img_width - 1, x2))
    y2 = max(0, min(img_height - 1, y2))

    # Ensure valid box (x2 > x1, y2 > y1)
    if x2 <= x1:
        x2 = x1 + 1
    if y2 <= y1:
        y2 = y1 + 1

    return [x1, y1, x2, y2]


def draw_bbox_opencv(
    image: np.ndarray,
    bbox_xyxy: list[int],
    color: tuple[int, int, int],
    thickness: int = 2,
    label: str | None = None,
) -> None:
    """Draw bounding box on image using OpenCV.

    Args:
        image: OpenCV image in BGR format (modified in-place).
        bbox_xyxy: Bounding box as [x1, y1, x2, y2] in pixel coordinates.
        color: BGR color tuple.
        thickness: Line thickness in pixels.
        label: Optional text label to draw above the box.
    """
    x1, y1, x2, y2 = bbox_xyxy

    # Draw rectangle
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    # Add label if provided
    if label:
        # Calculate text size and position
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        text_thickness = 1
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, text_thickness)

        # Draw background rectangle for text
        text_bg_x1 = x1
        text_bg_y1 = y1 - text_h - 10
        text_bg_x2 = x1 + text_w + 10
        text_bg_y2 = y1

        cv2.rectangle(image, (text_bg_x1, text_bg_y1), (text_bg_x2, text_bg_y2), color, -1)

        # Draw text
        text_color = (255, 255, 255) if sum(color) < 400 else (0, 0, 0)
        cv2.putText(image, label, (x1 + 5, y1 - 5), font, font_scale, text_color, text_thickness)


def create_visualization_opencv(
    image_path: str,
    gt_bbox_normalized: list[float],
    pred_bbox_normalized: list[float],
    iou_score: float,
    save_path: str | None = None,
) -> np.ndarray | None:
    """Create visualization using OpenCV.

    Args:
        image_path: Path to original image.
        gt_bbox_normalized: Ground truth bbox [cx, cy, w, h] normalized to [0,1].
        pred_bbox_normalized: Prediction bbox [cx, cy, w, h] normalized to [0,1].
        iou_score: IoU score between ground truth and prediction.
        save_path: Optional path to save the visualization.

    Returns:
        Visualization image as numpy array, or None if loading fails.
    """
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        logging.info(f"Warning: Could not load image {image_path}")
        return None

    img_height, img_width = image.shape[:2]

    # Convert normalized bboxes to pixel coordinates
    gt_bbox_xyxy = denormalize_bbox(gt_bbox_normalized, img_width, img_height)
    pred_bbox_xyxy = denormalize_bbox(pred_bbox_normalized, img_width, img_height)

    # Draw bounding boxes
    # Ground truth in green (BGR format: (0, 255, 0))
    draw_bbox_opencv(image, gt_bbox_xyxy, (0, 255, 0), thickness=3, label="GT")

    # Prediction in red (BGR format: (0, 0, 255))
    draw_bbox_opencv(
        image, pred_bbox_xyxy, (0, 0, 255), thickness=3, label=f"Pred (IoU: {iou_score:.3f})"
    )

    # Add title text
    title = f"Ground Truth (Green) vs Prediction (Red) | IoU: {iou_score:.3f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    text_thickness = 2
    text_color = (255, 255, 255)

    # Add text background
    (text_w, text_h), _ = cv2.getTextSize(title, font, font_scale, text_thickness)
    cv2.rectangle(image, (10, 10), (text_w + 20, text_h + 20), (0, 0, 0), -1)
    cv2.putText(image, title, (15, text_h + 15), font, font_scale, text_color, text_thickness)

    # Save if requested
    if save_path:
        cv2.imwrite(str(save_path), image)

    return image


def create_visualization_matplotlib(
    image_path: str,
    gt_bbox_normalized: list[float],
    pred_bbox_normalized: list[float],
    iou_score: float,
    save_path: str | None = None,
    show_plot: bool = False,
) -> None:
    """Create visualization using matplotlib.

    Args:
        image_path: Path to original image.
        gt_bbox_normalized: Ground truth bbox [cx, cy, w, h] normalized to [0,1].
        pred_bbox_normalized: Prediction bbox [cx, cy, w, h] normalized to [0,1].
        iou_score: IoU score between ground truth and prediction.
        save_path: Optional path to save the visualization.
        show_plot: Whether to display the plot interactively.
    """
    # Load image
    image = Image.open(image_path).convert("RGB")
    img_width, img_height = image.size

    # Convert normalized bboxes to pixel coordinates for matplotlib
    def norm_to_matplotlib_rect(bbox_norm):
        cx, cy, w, h = bbox_norm
        # Convert to matplotlib rectangle format (bottom-left corner + width, height)
        x = (cx - w / 2) * img_width
        y = (cy - h / 2) * img_height
        w_px = w * img_width
        h_px = h * img_height
        return x, y, w_px, h_px

    gt_rect = norm_to_matplotlib_rect(gt_bbox_normalized)
    pred_rect = norm_to_matplotlib_rect(pred_bbox_normalized)

    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)

    # Draw ground truth (green)
    gt_patch = patches.Rectangle(
        (gt_rect[0], gt_rect[1]),
        gt_rect[2],
        gt_rect[3],
        linewidth=3,
        edgecolor="green",
        facecolor="none",
        label="Ground Truth",
    )
    ax.add_patch(gt_patch)

    # Draw prediction (red)
    pred_patch = patches.Rectangle(
        (pred_rect[0], pred_rect[1]),
        pred_rect[2],
        pred_rect[3],
        linewidth=3,
        edgecolor="red",
        facecolor="none",
        label=f"Prediction (IoU: {iou_score:.3f})",
    )
    ax.add_patch(pred_patch)

    # Add legend and title
    ax.legend(loc="upper right", fontsize=12)
    ax.set_title(f"RF-DETR Prediction vs Ground Truth | IoU: {iou_score:.3f}", fontsize=14)
    ax.axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close()


def visualize_predictions(
    predictions_json_path: str,
    images_dir: str,
    output_dir: str,
    max_images: int | None = None,
    min_iou_threshold: float = 0.0,
    max_iou_threshold: float = 1.0,
    backend: str = "opencv",
    sort_by: str = "iou_desc",
) -> None:
    """Create visualizations for all predictions.

    Args:
        predictions_json_path: Path to predictions JSON file.
        images_dir: Directory containing original images.
        output_dir: Directory to save visualizations.
        max_images: Maximum number of images to visualize. None for all.
        min_iou_threshold: Minimum IoU threshold to include.
        max_iou_threshold: Maximum IoU threshold to include.
        backend: Visualization backend ("opencv" or "matplotlib").
        sort_by: Sort order ("iou_asc", "iou_desc", "filename", or "random").
    """
    # Load predictions
    logging.info(f"Loading predictions from {predictions_json_path}...")
    data = load_predictions_json(predictions_json_path)
    predictions = data["predictions"]

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Filter by IoU threshold
    filtered_predictions = [
        p
        for p in predictions
        if min_iou_threshold <= p["metrics"]["best_iou"] <= max_iou_threshold
    ]

    logging.info(
        f"Filtered to {len(filtered_predictions)} images with IoU in [{min_iou_threshold:.2f}, {max_iou_threshold:.2f}]"
    )

    # Sort predictions
    if sort_by == "iou_desc":
        filtered_predictions.sort(key=lambda x: x["metrics"]["best_iou"], reverse=True)
    elif sort_by == "iou_asc":
        filtered_predictions.sort(key=lambda x: x["metrics"]["best_iou"])
    elif sort_by == "filename":
        filtered_predictions.sort(key=lambda x: x["image_filename"])
    elif sort_by == "random":
        import random

        random.shuffle(filtered_predictions)

    # Limit number of images
    if max_images:
        filtered_predictions = filtered_predictions[:max_images]

    logging.info(
        f"Creating visualizations for {len(filtered_predictions)} images using {backend} backend..."
    )

    # Process images
    success_count = 0
    for i, pred_data in enumerate(tqdm(filtered_predictions, desc="Creating visualizations")):
        image_filename = pred_data["image_filename"]
        image_path = Path(images_dir) / image_filename

        if not image_path.exists():
            logging.info(f"Warning: Image not found: {image_path}")
            continue

        # Get data
        gt_bbox = pred_data["ground_truth"]["box_normalized"]
        pred_bbox = pred_data["predictions"]["best_prediction"]
        iou_score = pred_data["metrics"]["best_iou"]

        # Create output filename
        image_name_no_ext = Path(image_filename).stem
        output_filename = f"vis_{i:04d}_{image_name_no_ext}_iou{iou_score:.3f}.jpg"
        output_path_full = output_path / output_filename

        try:
            if backend == "opencv":
                result = create_visualization_opencv(
                    image_path, gt_bbox, pred_bbox, iou_score, output_path_full
                )
                if result is not None:
                    success_count += 1
            else:  # matplotlib
                create_visualization_matplotlib(
                    image_path, gt_bbox, pred_bbox, iou_score, output_path_full
                )
                success_count += 1

        except Exception as e:
            logging.info(f"Error processing {image_filename}: {e}")
            continue

    logging.info(f"\nSuccessfully created {success_count} visualizations!")
    logging.info(f"Saved to: {output_path}")

    # Create summary statistics
    iou_scores = [p["metrics"]["best_iou"] for p in filtered_predictions[:success_count]]
    if iou_scores:
        logging.info("\nIoU Statistics:")
        logging.info(f"   Mean IoU: {np.mean(iou_scores):.4f}")
        logging.info(f"   Median IoU: {np.median(iou_scores):.4f}")
        logging.info(f"   Min IoU: {np.min(iou_scores):.4f}")
        logging.info(f"   Max IoU: {np.max(iou_scores):.4f}")
        logging.info(
            f"   IoU > 0.9: {sum(1 for iou in iou_scores if iou > 0.9)}/{len(iou_scores)} images"
        )


def main() -> None:
    """Main function with argument parsing for visualization script."""
    parser = argparse.ArgumentParser(
        description="Visualize RF-DETR predictions vs ground truth",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--predictions_json",
        type=str,
        required=True,
        help="Path to predictions JSON file (from evaluate.py --save_predictions)",
    )
    parser.add_argument(
        "--images_dir", type=str, required=True, help="Directory containing original images"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./visualizations",
        help="Directory to save visualization images",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=50,
        help="Maximum number of images to visualize (None for all)",
    )
    parser.add_argument(
        "--min_iou", type=float, default=0.0, help="Minimum IoU threshold to include"
    )
    parser.add_argument(
        "--max_iou", type=float, default=1.0, help="Maximum IoU threshold to include"
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["opencv", "matplotlib"],
        default="opencv",
        help="Visualization backend to use",
    )
    parser.add_argument(
        "--sort_by",
        type=str,
        choices=["iou_desc", "iou_asc", "filename", "random"],
        default="iou_desc",
        help="How to sort images before visualization",
    )

    args = parser.parse_args()

    # Validate inputs
    if not Path(args.predictions_json).exists():
        logging.info(f"Error: Predictions JSON file not found: {args.predictions_json}")
        return

    if not Path(args.images_dir).exists():
        logging.info(f"Error: Images directory not found: {args.images_dir}")
        return

    # Run visualization
    visualize_predictions(
        predictions_json_path=args.predictions_json,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        max_images=args.max_images,
        min_iou_threshold=args.min_iou,
        max_iou_threshold=args.max_iou,
        backend=args.backend,
        sort_by=args.sort_by,
    )


if __name__ == "__main__":
    main()
