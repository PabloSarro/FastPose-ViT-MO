# This file contains the evaluation script for the RF-DETR model on the SPEED dataset

import argparse
import os
import torch
import json
from pathlib import Path

from object_detector.LWDETR.lwdetr_models import create_lwdetr_model, get_lwdetr_loss
from object_detector.constants import LWDETR_IMG_SIZES
from object_detector.datasets import verify_dataset, SPEEDDataset
from object_detector.utils import setup_logging, visualize_predictions, box_iou, box_cxcywh_to_xyxy
from torch.utils.data import DataLoader

# Use LW-DETR criterion for evaluation
from object_detector.engine import val_epoch
from PIL import Image
import logging


def evaluate_model_on_dataset(
    model_weights: str,
    model_variant: str,
    dataset_root_dir: str,
    bbox_json_path: str | None = None,
    batch_size: int = 8,
    num_workers: int = 4,
    save_predictions: bool = False,
    output_dir: str | None = None,
    save_visualizations: bool = False,
    logger: logging.Logger | None = None,
) -> dict[str, float] | None:
    """Evaluate a trained RF-DETR model on the SPEED dataset.

    Args:
        model_weights: Path to trained model weights.
        model_variant: Model variant used ("tiny", "small", "medium").
        dataset_root_dir: Root directory of SPEED dataset.
        bbox_json_path: Path to bounding box JSON file. If None, uses
            dataset_root_dir/speed_bbox_annotations.json.
        batch_size: Batch size for evaluation.
        num_workers: Number of data loading workers.
        save_predictions: Whether to save predictions to file.
        output_dir: Directory to save predictions.
        save_visualizations: Whether to save visualization images.
        logger: Logger instance to use.

    Returns:
        Dictionary with test_loss key, or None if evaluation fails.
    """

    # Setup logging - use passed logger if provided
    if logger is None:
        logger, _ = setup_logging()  # Fallback to simple logging

    logger.info("Starting RF-DETR evaluation...")

    # Setup device - always use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Setup dataset paths - follow src/train.py pattern
    images_dir = os.path.join(dataset_root_dir, "images")
    if bbox_json_path is None:
        bbox_json_path = os.path.join(dataset_root_dir, "speed_bbox_annotations.json")

    logger.info(f"Model path: {model_weights}")
    logger.info(f"Model variant: {model_variant}")
    logger.info(f"Images directory: {images_dir}")
    logger.info(f"Bounding box JSON file: {bbox_json_path}")

    # Verify dataset
    logger.info("Verifying dataset...")
    if not verify_dataset(images_dir, bbox_json_path, split="test"):
        logger.error("Dataset verification failed!")
        return
    logger.info("Dataset verification passed!")

    # Get model configuration
    if model_variant not in LWDETR_IMG_SIZES:
        raise ValueError(f"Unknown model variant: {model_variant}")

    img_size_px = LWDETR_IMG_SIZES[model_variant]
    img_size = (img_size_px, img_size_px)

    # Create model
    logger.info("Loading model...")
    model = create_lwdetr_model(variant=model_variant, num_classes=1)

    # Load trained weights
    if not os.path.exists(model_weights):
        raise FileNotFoundError(f"Model file not found: {model_weights}")

    model.load_from_pretrained(model_weights)
    model = model.to(device)
    model.eval()

    logger.info(f"Model loaded from: {model_weights}")

    # Create evaluation dataset (using validation split)
    logger.info("Creating evaluation dataset...")
    eval_dataset = SPEEDDataset(
        images_dir=images_dir,
        annotations_file=bbox_json_path,
        img_size=img_size,
        split="test",
        no_pixel_augmentation=True,  # Always disable augmentation for evaluation
        no_spatial_augmentation=True,
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=3 if num_workers > 0 else None,
        multiprocessing_context="fork" if num_workers > 0 else None,
    )

    logger.info(f"Evaluation samples: {len(eval_dataset)}")

    # Use LW-DETR loss for evaluation
    try:
        criterion = get_lwdetr_loss()
        logging.info("Using LW-DETR loss for evaluation")
    except Exception as e:
        raise ImportError(f"Could not create LW-DETR loss for evaluation. Details: {e}")

    # Run evaluation using val_epoch (same as training)
    logger.info("Starting evaluation...")

    test_loss, test_components = val_epoch(
        model=model,
        val_loader=eval_loader,
        criterion=criterion,
        device=device,
        epoch=0,  # Dummy epoch for evaluation
        compute_ap_metrics=True,  # Enable AP metrics computation
    )

    # Print simple results like training script
    logger.info("=" * 50)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 50)
    logger.info(f"Model: {model_variant}")
    logger.info(f"Dataset Size: {len(eval_dataset)} images")
    logger.info(f"Test Loss (IoU): {test_loss:.4f}")

    # Print AP metrics if available
    if test_components:
        logger.info("\nDetection Metrics:")
        if "AP@0.50" in test_components:
            logger.info(f"AP@0.50: {test_components['AP@0.50']:.4f}")
        if "AP@0.75" in test_components:
            logger.info(f"AP@0.75: {test_components['AP@0.75']:.4f}")
        if "AP@[0.50:0.95]" in test_components:
            logger.info(f"AP@[0.50:0.95]: {test_components['AP@[0.50:0.95]']:.4f}")
        if "num_predictions" in test_components and "num_targets" in test_components:
            logger.info(f"Total Predictions: {test_components['num_predictions']}")
            logger.info(f"Total Targets: {test_components['num_targets']}")

    logger.info("=" * 50)

    # Single inference pass to collect all prediction data
    logger.info("Collecting predictions...")
    all_predictions = []
    model.eval()
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(eval_loader):
            images = images.to(device)
            outputs = model(images)

            # Batch CPU transfers
            pred_boxes = outputs["boxes"].cpu()  # [batch_size, 1, 4]
            pred_scores = outputs["scores"].cpu()  # [batch_size, 1]
            batch_targets = targets.cpu()  # [batch_size, 4]

            # Batched IoU computation
            pred_squeezed = pred_boxes.squeeze(1)  # [batch_size, 4]
            pred_xyxy = box_cxcywh_to_xyxy(pred_squeezed)
            target_xyxy = box_cxcywh_to_xyxy(batch_targets)
            iou_matrix = box_iou(pred_xyxy, target_xyxy)
            ious = torch.diag(iou_matrix)

            for i in range(images.shape[0]):
                image_idx = batch_idx * eval_loader.batch_size + i
                all_predictions.append(
                    {
                        "image_idx": image_idx,
                        "pred_boxes": pred_boxes[i],
                        "pred_scores": pred_scores[i],
                        "target_bbox": batch_targets[i],
                        "iou": ious[i].item(),
                    }
                )

    # Save predictions if requested
    if save_predictions and output_dir:
        logger.info("Saving predictions...")
        save_model_predictions(
            predictions=all_predictions,
            output_dir=output_dir,
            model_variant=model_variant,
        )

    # Get worst predictions for analysis
    logger.info("Analyzing worst predictions...")
    worst_predictions = get_worst_predictions(
        predictions=all_predictions,
        dataset=eval_loader.dataset,
        num_worst=10,
    )

    # Print worst predictions
    logger.info("\n" + "=" * 60)
    logger.info("TOP 10 WORST PREDICTIONS (Lowest IoU)")
    logger.info("=" * 60)
    for i, pred_info in enumerate(worst_predictions, 1):
        logger.info(f"{i:2d}. {pred_info['image_name']:<20} IoU: {pred_info['iou']:.4f}")
    logger.info("=" * 60)

    # Save visualizations if requested
    if save_visualizations and output_dir:
        logger.info("Saving visualizations...")
        save_evaluation_visualizations(
            model=model,
            eval_loader=eval_loader,
            device=device,
            output_dir=output_dir,
            images_dir=images_dir,
        )

    return {"test_loss": test_loss}


def save_model_predictions(
    predictions: list[dict],
    output_dir: str,
    model_variant: str,
) -> None:
    """Save pre-collected model predictions to JSON file for analysis.

    Args:
        predictions: List of prediction dicts from the collection loop.
        output_dir: Directory to save the predictions JSON file.
        model_variant: Model variant name for the output filename.
    """
    json_predictions = []

    for pred in predictions:
        pred_boxes = pred["pred_boxes"]  # [1, 4] or [4]
        pred_scores = pred["pred_scores"]  # [1] or scalar

        # Ensure consistent shape
        if pred_boxes.dim() == 1:
            pred_boxes = pred_boxes.unsqueeze(0)
        if pred_scores.dim() == 0:
            pred_scores = pred_scores.unsqueeze(0)

        target_bbox = pred["target_bbox"].tolist()

        prediction_data = {
            "image_id": pred["image_idx"],
            "predictions": {
                "boxes": pred_boxes.tolist(),
                "scores": pred_scores.tolist(),
                "num_detections": pred_boxes.shape[0],
            },
            "targets": {
                "boxes": [target_bbox],
                "labels": [1],
                "num_targets": 1,
            },
        }

        json_predictions.append(prediction_data)

    # Save to JSON file
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    predictions_file = output_path / f"predictions_{model_variant}.json"
    with open(predictions_file, "w") as f:
        json.dump(json_predictions, f, indent=2)

    logging.info(f"Predictions saved to: {predictions_file}")


def get_worst_predictions(
    predictions: list[dict],
    dataset: torch.utils.data.Dataset,
    num_worst: int = 10,
) -> list[dict[str, str | float | int]]:
    """Get the worst predictions based on pre-computed IoU scores.

    Args:
        predictions: List of prediction dicts from the collection loop (with iou field).
        dataset: The evaluation dataset (for image name lookup).
        num_worst: Number of worst predictions to return.

    Returns:
        List of dicts with image_name, iou, and image_idx for worst predictions,
        sorted by IoU in ascending order.
    """
    prediction_scores = []

    for pred in predictions:
        image_idx = pred["image_idx"]

        # Get image name from dataset
        if hasattr(dataset, "image_files"):
            image_file = dataset.image_files[image_idx]
            if isinstance(image_file, str):
                image_name = Path(image_file).name
            else:
                image_name = image_file.name
        else:
            image_name = f"image_{image_idx:06d}"

        prediction_scores.append(
            {"image_name": image_name, "iou": pred["iou"], "image_idx": image_idx}
        )

    # Sort by IoU (ascending) and return worst N
    prediction_scores.sort(key=lambda x: x["iou"])
    return prediction_scores[:num_worst]


def save_evaluation_visualizations(
    model: torch.nn.Module,
    eval_loader: DataLoader,
    device: torch.device,
    output_dir: str,
    images_dir: str,
) -> None:
    """Save visualizations for evaluation dataset with predictions and ground truth.

    Args:
        model: Trained model to generate predictions.
        eval_loader: DataLoader for the evaluation dataset.
        device: Device to run inference on.
        output_dir: Directory to save visualization images.
        images_dir: Directory containing original images.
    """
    # Create visualizations directory
    vis_output_dir = Path(output_dir) / "visualizations"
    vis_output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(eval_loader):
            images = images.to(device)

            # Forward pass - direct single prediction (no inference method needed)
            outputs = model(images)

            batch_size = images.shape[0]
            for i in range(batch_size):
                # Get image name from dataset
                image_idx = batch_idx * eval_loader.batch_size + i
                if hasattr(eval_loader.dataset, "image_files"):
                    image_file = eval_loader.dataset.image_files[image_idx]
                    image_name = image_file.stem
                    # Construct full image path
                    image_path = str(image_file)
                else:
                    # Fallback - try to construct path
                    image_name = f"image_{image_idx:06d}"
                    # This is a fallback that might not work in all cases
                    image_path = f"{images_dir}/test/{image_name}.jpg"
                    if not Path(image_path).exists():
                        logging.info(f"Warning: Could not find image at {image_path}")
                        continue

                # Skip if no detections to visualize
                if "scores" in outputs:
                    if outputs["scores"][i].numel() == 0:
                        continue
                else:
                    if outputs["boxes"][i].numel() == 0:
                        continue

                # Create visualization
                vis_path = vis_output_dir / f"eval_{image_name}.jpg"

                try:
                    with Image.open(image_path) as img:
                        original_size = img.size  # (width, height)

                    # Get model input size
                    variant = (
                        eval_loader.dataset.model_variant
                        if hasattr(eval_loader.dataset, "model_variant")
                        else "medium"
                    )
                    img_size_px = LWDETR_IMG_SIZES[variant]
                    model_size = (img_size_px, img_size_px)

                    # Create single-image output dict for visualization
                    image_outputs = {
                        "scores": outputs["scores"][i : i + 1],  # Keep batch dimension
                        "boxes": outputs["boxes"][i : i + 1],
                    }

                    visualize_predictions(
                        image_path=image_path,
                        predictions=image_outputs,
                        original_size=original_size,
                        model_size=model_size,
                        save_path=str(vis_path),
                    )

                except Exception as e:
                    logging.info(f"Failed to create visualization for {image_name}: {e}")
                    continue

    logging.info(f"Visualizations saved to: {vis_output_dir}")


def main() -> None:
    """Main function with argument parsing for evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate RF-DETR on SPEED dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--model_weights", type=str, required=True, help="Path to trained model weights"
    )
    parser.add_argument(
        "--model_variant",
        type=str,
        choices=list(LWDETR_IMG_SIZES.keys()),
        required=True,
        help="Model variant used for training",
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["SPEED", "SPEED_PLUS_SYNTHETIC", "SPEED_PLUS"],
        default="SPEED",
        help="Dataset to use for evaluation",
    )
    parser.add_argument(
        "--dataset_root_dir",
        type=str,
        default=None,
        help="Root directory of dataset (auto-determined from dataset choice if not specified)",
    )
    parser.add_argument(
        "--bbox_json_path",
        type=str,
        default=None,
        help="Path to the JSON file containing bounding box information. If this path is provided, the model will use bounding boxes, otherwise it will use the full image",
    )

    # Evaluation arguments
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")

    # Output arguments
    parser.add_argument(
        "--output_dir", type=str, default="./eval_results", help="Output directory for results"
    )

    parser.add_argument(
        "--save_visualizations",
        action="store_true",
        help="Save visualization images with bounding boxes",
    )
    parser.add_argument(
        "--log_dir", type=str, default="./runs", help="Directory to store TensorBoard logs"
    )

    args = parser.parse_args()

    # Set dataset_root_dir based on dataset choice if not specified
    if args.dataset_root_dir is None:
        if args.dataset == "SPEED":
            args.dataset_root_dir = "./SPEED_FIXED"
        elif args.dataset == "SPEED_PLUS_SYNTHETIC":
            args.dataset_root_dir = "./SPEED_PLUS_SYNTHETIC_FIXED"
        elif args.dataset == "SPEED_PLUS":
            args.dataset_root_dir = "./SPEED_PLUS_FIXED"

    # Set up logging with args
    logger, log_file = setup_logging(args, base_filename="eval")

    # Log the arguments provided in a clean format
    logger.info("=========================================")
    logger.info("Arguments provided:")
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")
    logger.info("=========================================")
    logger.info("")

    # Run evaluation
    evaluate_model_on_dataset(
        model_weights=args.model_weights,
        model_variant=args.model_variant,
        dataset_root_dir=args.dataset_root_dir,
        bbox_json_path=args.bbox_json_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        save_predictions=True,
        output_dir=args.output_dir,
        save_visualizations=args.save_visualizations,
        logger=logger,
    )


if __name__ == "__main__":
    main()
