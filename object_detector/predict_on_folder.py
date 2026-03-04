# This file contains the script to run inference on a folder of images using the RF-DETR model

import argparse
import json
import logging
import os
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from object_detector.utils import box_cxcywh_to_xyxy

from object_detector.LWDETR.lwdetr_models import create_lwdetr_model
from object_detector.constants import LWDETR_IMG_SIZES
from object_detector.utils import (
    setup_logging,
    scale_boxes_to_original,
    visualize_predictions,
    create_output_dir,
)
from object_detector.utils import prepare_model_input


def custom_collate_fn(
    batch: list[tuple[torch.Tensor, tuple[int, int], str, str]],
) -> tuple[torch.Tensor, list[tuple[int, int]], list[str], list[str]]:
    """Custom collate function to handle mixed data types in batch.

    Args:
        batch: List of tuples containing (image_tensor, original_size, path, name).

    Returns:
        Tuple of (stacked_images, original_sizes, paths, names).
    """
    # Separate the different elements
    images = []
    original_sizes = []
    paths = []
    names = []

    for image_tensor, original_size, path, name in batch:
        images.append(image_tensor)
        original_sizes.append(original_size)
        paths.append(path)
        names.append(name)

    # Stack images into a tensor
    batch_images = torch.stack(images, dim=0)

    # Keep original_sizes, paths, names as lists (don't stack)
    return batch_images, original_sizes, paths, names


class ImageFolderDataset(Dataset):
    """Dataset for loading images from a folder for inference.

    Args:
        image_files: List of paths to image files.
        transform: Albumentations transform pipeline.
        model_img_size: Target model input size (height, width).
        convert_to_grayscale: Whether to convert images to grayscale.
    """

    def __init__(
        self,
        image_files: list[Path],
        transform,
        model_img_size: tuple[int, int],
        convert_to_grayscale: bool = False,
    ) -> None:
        self.image_files = image_files
        self.transform = transform
        self.model_img_size = model_img_size
        self.convert_to_grayscale = convert_to_grayscale

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, tuple[int, int], str, str]:
        image_file = self.image_files[idx]
        try:
            # Load and preprocess image
            image = Image.open(image_file)
            if self.convert_to_grayscale:
                grayscale_image = image.convert("L")
                image = Image.merge("RGB", (grayscale_image, grayscale_image, grayscale_image))
            else:
                image = image.convert("RGB")

            original_size = image.size  # (width, height)

            # Convert PIL to numpy for Albumentations
            import numpy as np

            image_np = np.array(image)

            # Convert to float32 and normalize to [0,1] to match training preprocessing
            # Training uses max_pixel_value=1.0, so input should be in [0,1] range
            image_np = image_np.astype(np.float32) / 255.0

            # Transform image for model input using Albumentations
            transformed = self.transform(image=image_np)
            image_tensor = transformed["image"]

            return image_tensor, original_size, str(image_file), image_file.name
        except Exception as e:
            # Return placeholder for failed images with correct size
            print(f"Failed to load {image_file}: {e}")  # Debug
            dummy_tensor = torch.zeros(3, self.model_img_size[0], self.model_img_size[1])
            return dummy_tensor, self.model_img_size, str(image_file), image_file.name


def predict_on_folder(
    model_weights: str,
    model_variant: str,
    images_dir: str,
    output_dir: str,
    save_visualizations: bool = False,
    batch_size: int = 1,
    num_workers: int = 4,
    image_extensions: list[str] | None = None,
    convert_to_grayscale: bool = False,
    logger: logging.Logger | None = None,
) -> dict:
    """Run inference on all images in a folder.

    Args:
        model_weights: Path to trained model weights.
        model_variant: Model variant ("tiny", "small", "medium").
        images_dir: Directory containing images.
        output_dir: Directory to save predictions and visualizations.
        save_visualizations: Whether to save visualization images.
        batch_size: Batch size for processing.
        num_workers: Number of data loading workers.
        image_extensions: List of image file extensions to process.
        convert_to_grayscale: Convert images to grayscale (replicated to 3-channel).
        logger: Logger instance to use. If None, creates a simple logger.

    Returns:
        Dictionary with metadata, statistics, and predictions.
    """
    # Setup logging - use passed logger if provided
    if logger is None:
        logger, _ = setup_logging()  # Fallback to simple logging
    logger.info("Starting object detection inference on folder...")

    # Default image extensions
    if image_extensions is None:
        image_extensions = ["jpg", "jpeg", "png", "bmp", "tiff"]

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Get model configuration
    if model_variant not in LWDETR_IMG_SIZES:
        raise ValueError(
            f"Model variant '{model_variant}' not found. Available: {list(LWDETR_IMG_SIZES.keys())}"
        )

    img_size_px = LWDETR_IMG_SIZES[model_variant]
    model_img_size = (img_size_px, img_size_px)
    logger.info(f"Model variant: {model_variant}, Input size: {model_img_size}")
    logger.info(f"Convert to grayscale: {convert_to_grayscale}")

    # Create and load model
    logger.info("Loading model...")
    model = create_lwdetr_model(variant=model_variant, num_classes=1)

    if not os.path.exists(model_weights):
        raise FileNotFoundError(f"Model weights not found: {model_weights}")

    model.load_from_pretrained(model_weights)
    model = model.to(device)
    model.eval()

    # CRITICAL FIX: Match EXACT model configuration sequence from training
    # Training does: False -> True, so we must replicate this exact sequence
    if hasattr(model, "return_all_queries"):
        # Step 1: Set to False (matches training line 238)
        model.return_all_queries = False
        logger.info("Step 1: Set model.return_all_queries = False (matching training)")

        # Step 2: Set to True (matches training line 244)
        model.return_all_queries = True
        logger.info("Step 2: Set model.return_all_queries = True (matching training validation)")
    else:
        logger.warning(
            "⚠️  Model does not have 'return_all_queries' attribute - using default behavior"
        )

    # Log model configuration for debugging
    logger.info("Model configuration:")
    logger.info(f"  - Variant: {model_variant}")
    logger.info(f"  - Training mode: {model.training}")
    logger.info(f"  - Return all queries: {getattr(model, 'return_all_queries', 'N/A')}")
    logger.info(f"  - Model type: {type(model).__name__}")

    logger.info(f"Model loaded from: {model_weights}")

    # Create output directory
    output_path = create_output_dir(output_dir)

    # Setup visualization directory
    vis_output_dir = None
    if save_visualizations:
        vis_output_dir = output_path / "visualizations"
        vis_output_dir.mkdir(exist_ok=True)
        logger.info(f"Visualizations will be saved to: {vis_output_dir}")

    # Find all image files
    image_files = []
    images_path = Path(images_dir)

    if not images_path.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    for ext in image_extensions:
        image_files.extend(images_path.glob(f"*.{ext}"))
        image_files.extend(images_path.glob(f"*.{ext.upper()}"))

    image_files = sorted(image_files)
    logger.info(f"Found {len(image_files)} images in {images_dir}")

    if len(image_files) == 0:
        logger.warning("No images found! Check the directory path and image extensions.")
        return {"predictions": [], "statistics": {"total_images": 0, "images_with_detections": 0}}

    # CRITICAL FIX: Use the exact same preprocessing as training
    # Training uses Albumentations, not torchvision transforms
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    from object_detector.constants import IMAGENET_MEAN, IMAGENET_STD

    logger.info(
        f"Using training-compatible preprocessing: IMAGENET_MEAN={IMAGENET_MEAN}, IMAGENET_STD={IMAGENET_STD}"
    )

    # Setup image preprocessing to match training exactly
    # CRITICAL: Training uses max_pixel_value=1.0, not 255.0!
    transform = A.Compose(
        [
            A.Resize(height=model_img_size[0], width=model_img_size[1]),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=1.0),
            ToTensorV2(),
        ]
    )

    # Create dataset and dataloader for efficient batch processing
    logger.info("Creating dataset and dataloader...")
    dataset = ImageFolderDataset(
        image_files,
        transform,
        model_img_size,
        convert_to_grayscale=convert_to_grayscale,
    )

    # Optimize dataloader for prefetching and performance (match src/ settings)
    use_cuda = device.type == "cuda"
    prefetch_factor = 3 if num_workers > 0 else None

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,  # Enable pinned memory for faster GPU transfer
        drop_last=False,
        collate_fn=custom_collate_fn,
        prefetch_factor=prefetch_factor,  # Match src/ prefetch factor
        persistent_workers=False,  # Disable for consistency with other optimizations
        multiprocessing_context="fork" if num_workers > 0 else None,
    )

    logger.info(
        f"DataLoader created with {num_workers} workers, pin_memory={'enabled' if use_cuda else 'disabled'}"
    )
    if prefetch_factor:
        logger.info(
            f"Prefetch factor: {prefetch_factor}, persistent workers: disabled, context: fork"
        )

    # Run inference
    results = []
    images_with_detections = 0
    total_detections = 0

    logger.info("Running inference...")

    # Create iterator for better memory management with prefetching
    dataloader_iter = iter(dataloader)

    # Create CUDA stream for async operations if using GPU
    if use_cuda:
        stream = torch.cuda.Stream()
    else:
        stream = None

    with torch.no_grad():
        # Use enumerate with tqdm for better performance
        dataloader_len = len(dataloader)  # Cache length
        for batch_idx in tqdm(range(dataloader_len), desc="Processing batches"):
            try:
                batch_data = next(dataloader_iter)
            except StopIteration:
                break
            try:
                # Unpack batch data
                batch_images, batch_original_sizes, batch_paths, batch_names = batch_data

                # Use CUDA streams for overlapping data transfer and computation
                if use_cuda and stream is not None:
                    with torch.cuda.stream(stream):
                        # Use non_blocking transfer for better GPU pipeline utilization
                        batch_images = batch_images.to(device, non_blocking=True)
                        # Prepare input format (NestedTensor for LW-DETR)
                        model_inputs = prepare_model_input(batch_images, model)
                        # Run inference on batch - direct single prediction
                        outputs = model(model_inputs)
                    # Synchronize stream to ensure computation is complete
                    stream.synchronize()
                else:
                    # Use non_blocking transfer for better GPU pipeline utilization
                    batch_images = batch_images.to(device, non_blocking=use_cuda)
                    # Prepare input format (NestedTensor for LW-DETR)
                    model_inputs = prepare_model_input(batch_images, model)
                    # Run inference on batch - direct single prediction
                    outputs = model(model_inputs)

                # Debug: Log output format on first batch
                if batch_idx == 0:
                    logger.info(f"Model output keys: {list(outputs.keys())}")
                    for key, value in outputs.items():
                        if isinstance(value, torch.Tensor):
                            logger.info(f"  {key}: shape {value.shape}, dtype {value.dtype}")
                        else:
                            logger.info(f"  {key}: type {type(value)}")

                # Handle multi-query outputs (when return_all_queries=True)
                # Model now returns all queries: pred_boxes [B, num_queries, 4], pred_logits [B, num_queries, num_classes]
                if "pred_boxes" in outputs and "pred_logits" in outputs:
                    # Multi-query outputs - select best query per image
                    pred_boxes = outputs["pred_boxes"]  # [B, num_queries, 4]
                    pred_logits = outputs["pred_logits"]  # [B, num_queries, num_classes]

                    # Calculate confidence scores to select best query (match validation exactly)
                    if pred_logits.shape[-1] > 1:
                        # Multi-class: use satellite class confidence (class 1)
                        probs = torch.sigmoid(pred_logits[..., 1])  # [B, num_queries]
                        pred_logits[..., 1]
                    else:
                        # Single class: use the logit directly
                        probs = torch.sigmoid(pred_logits[..., 0])  # [B, num_queries]
                        pred_logits[..., 0]

                    # Select best query per image (match validation exactly)
                    best_query_idx = torch.argmax(probs, dim=1)  # [B]
                    batch_indices = torch.arange(pred_boxes.shape[0], device=pred_boxes.device)

                    # Extract best predictions using validation logic
                    batch_pred_boxes = pred_boxes[batch_indices, best_query_idx]  # [B, 4]
                    batch_prob_scores = probs[batch_indices, best_query_idx]  # [B]

                    # Debug: Log confidence scores for first batch
                    if batch_idx == 0:
                        logger.info(f"Debug - Best query confidence: {batch_prob_scores[0]:.6f}")
                        logger.info(
                            f"Debug - All confidences: min={probs[0].min():.6f}, max={probs[0].max():.6f}, mean={probs[0].mean():.6f}"
                        )

                else:
                    # Fallback: assume single prediction format (legacy path)
                    batch_pred_boxes = outputs.get(
                        "boxes", outputs.get("pred_boxes")
                    )  # [batch_size, 4]
                    batch_pred_scores = outputs.get(
                        "scores", outputs.get("pred_logits")
                    )  # [batch_size] or [batch_size, 1]

                    # Ensure consistent shapes for legacy path
                    if batch_pred_boxes.dim() == 3 and batch_pred_boxes.shape[1] == 1:
                        batch_pred_boxes = batch_pred_boxes.squeeze(1)  # [batch_size, 4]

                    if batch_pred_scores.dim() == 2 and batch_pred_scores.shape[1] == 1:
                        batch_pred_scores = batch_pred_scores.squeeze(1)  # [batch_size]
                    elif batch_pred_scores.dim() == 0:
                        batch_pred_scores = batch_pred_scores.unsqueeze(0)  # [1]

                    batch_prob_scores = batch_pred_scores

                # Process each image in the batch (still need per-image processing for different original sizes)
                batch_len = len(batch_images)  # Cache length
                for i in range(batch_len):
                    image_path = batch_paths[i]
                    image_name = batch_names[i]
                    original_size = batch_original_sizes[i]

                    # Get this image's predictions
                    pred_boxes = batch_pred_boxes[i : i + 1]  # Keep batch dim [1, 4]
                    prob_scores = batch_prob_scores[i : i + 1]  # Keep batch dim [1]

                    # Scale boxes to original image coordinates
                    scaled_boxes = scale_boxes_to_original(
                        pred_boxes, original_size, model_img_size
                    )

                    # Convert from cxcywh to x1,y1,x2,y2 format - vectorized
                    xyxy_boxes = box_cxcywh_to_xyxy(scaled_boxes)  # [1, 4]

                    # Convert to dictionary format - vectorized
                    xyxy_np = xyxy_boxes.cpu().numpy().astype(int)  # [1, 4]
                    boxes_dict_list = [
                        {
                            "x1": int(xyxy_np[0, 0]),
                            "y1": int(xyxy_np[0, 1]),
                            "x2": int(xyxy_np[0, 2]),
                            "y2": int(xyxy_np[0, 3]),
                        }
                    ]

                    scores_list = prob_scores.cpu().tolist()

                    images_with_detections += 1
                    total_detections += len(boxes_dict_list)

                    # Create result entry
                    result = {
                        "image_name": image_name,
                        "image_path": image_path,
                        "original_size": {"width": original_size[0], "height": original_size[1]},
                        "model_size": {"width": model_img_size[0], "height": model_img_size[1]},
                        "detections": {
                            "boxes_xyxy": boxes_dict_list,  # [{"x1": x1, "y1": y1, "x2": x2, "y2": y2}] in original image coordinates
                            "scores": scores_list,
                            "num_detections": len(boxes_dict_list),
                        },
                        "model_config": {
                            "variant": model_variant,
                        },
                    }

                    results.append(result)

                    # Create visualization if requested and detections exist
                    if save_visualizations and len(boxes_dict_list) > 0:
                        image_file = Path(image_path)
                        vis_path = vis_output_dir / f"vis_{image_file.stem}.jpg"
                        try:
                            # Create single-image output dict for visualization
                            # Use the selected best predictions for this image
                            image_outputs = {
                                "scores": batch_prob_scores[i : i + 1],  # Use confidence scores
                                "boxes": batch_pred_boxes[i : i + 1],
                            }
                            visualize_predictions(
                                image_path=image_path,
                                predictions=image_outputs,
                                original_size=original_size,
                                model_size=model_img_size,
                                save_path=str(vis_path),
                            )
                        except Exception as e:
                            logger.warning(f"Failed to create visualization for {image_name}: {e}")

            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")
                # Add error entries for this batch (use batch_size as fallback)
                for i in range(batch_size):  # batch_size already cached
                    results.append(
                        {
                            "image_name": f"batch_{batch_idx}_item_{i}",
                            "image_path": "unknown",
                            "error": str(e),
                            "detections": {"boxes_xyxy": [], "scores": [], "num_detections": 0},
                        }
                    )

    # Create summary statistics with cached lengths
    total_images = len(image_files)
    successfully_processed = len([r for r in results if "error" not in r])

    statistics = {
        "total_images": total_images,
        "successfully_processed": successfully_processed,
        "images_with_detections": images_with_detections,
        "total_detections": total_detections,
        "avg_detections_per_image": total_detections / total_images if total_images > 0 else 0,
        "detection_rate": images_with_detections / total_images if total_images > 0 else 0,
        "model_variant": model_variant,
    }

    # Save results to JSON
    results_file = output_path / f"predictions_{model_variant}.json"
    output_data = {
        "metadata": {
            "model_weights": model_weights,
            "model_variant": model_variant,
            "images_directory": str(images_dir),
            "output_directory": str(output_dir),
            "convert_to_grayscale": convert_to_grayscale,
        },
        "statistics": statistics,
        "predictions": results,
    }

    with open(results_file, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Results saved to: {results_file}")
    logger.info("=" * 50)
    logger.info("INFERENCE SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Total images processed: {statistics['total_images']}")
    logger.info(f"Successfully processed: {statistics['successfully_processed']}")
    logger.info(f"Images with detections: {statistics['images_with_detections']}")
    logger.info(f"Total detections: {statistics['total_detections']}")
    logger.info(f"Average detections per image: {statistics['avg_detections_per_image']:.2f}")
    logger.info(f"Detection rate: {statistics['detection_rate']:.1%}")
    logger.info("=" * 50)

    if save_visualizations:
        logger.info(f"Visualizations saved to: {vis_output_dir}")

    return output_data


def main() -> None:
    """Main function with argument parsing for inference script."""
    parser = argparse.ArgumentParser(
        description="Run object detection inference on all images in a folder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--model_weights",
        type=str,
        required=True,
        help="Path to trained model weights (e.g., rf-detr-nano.pth)",
    )
    parser.add_argument(
        "--model_variant",
        type=str,
        choices=list(LWDETR_IMG_SIZES.keys()),
        required=True,
        help="Model variant used for training",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="Directory containing images to process",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save predictions and visualizations",
    )

    # Optional arguments
    parser.add_argument(
        "--save_visualizations",
        action="store_true",
        help="Save visualization images with bounding boxes drawn",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for processing images",
    )
    parser.add_argument(
        "--image_extensions",
        nargs="+",
        default=["jpg", "jpeg", "png", "bmp", "tiff"],
        help="Image file extensions to process",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--log_dir", type=str, default="./runs", help="Directory to store TensorBoard logs"
    )
    parser.add_argument(
        "--convert_to_grayscale",
        action="store_true",
        help="Convert images to grayscale before inference",
    )

    args = parser.parse_args()

    # Set up logging with args
    logger, log_file = setup_logging(args, base_filename="predict")

    # Log the arguments provided in a clean format
    logger.info("=========================================")
    logger.info("Arguments provided:")
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")
    logger.info("=========================================")
    logger.info("")

    # Run inference
    try:
        predict_on_folder(
            model_weights=args.model_weights,
            model_variant=args.model_variant,
            images_dir=args.images_dir,
            output_dir=args.output_dir,
            save_visualizations=args.save_visualizations,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_extensions=args.image_extensions,
            convert_to_grayscale=args.convert_to_grayscale,
            logger=logger,
        )
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise


if __name__ == "__main__":
    main()
