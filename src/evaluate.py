# This script evaluates the FastPoseViT model on the SPEED dataset

import argparse
import torch
from torch.utils.data import DataLoader
from src.datasets import SPEEDDataset, get_dataset_config
from src.models import FastPoseViT, SUPPORTED_VIT_MODELS, VIT_MODELS
from src.metrics import compute_metrics
from src.utils import (
    setup_logging,
    process_rotation,
    bbox_relative_translation_to_translation,
    rotation_matrix_to_quaternion,
    get_absolute_orientation,
)
from tqdm import tqdm
import json
import os


def evaluate(args: argparse.Namespace, bbox_json_path: str) -> None:
    """
    Evaluate the FastPoseViT model on the SPEED dataset

    Args:
        args (argparse.Namespace): Command-line arguments

    Returns:
        None
    """
    # Set up logging
    logger, _ = setup_logging(args, base_filename="eval")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # If log_dir does not exist, create it
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # If output_json is provided, create the directory if it does not exist
    if args.output_json:
        output_dir = os.path.dirname(args.output_json)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    # Get the image size for the ViT model
    image_size = VIT_MODELS[args.vit_model][3] if args.vit_model in VIT_MODELS else (224, 224)
    logger.debug(f"ViT model: {args.vit_model}, config: {VIT_MODELS[args.vit_model]}")

    # Load the dataset
    test_dataset = SPEEDDataset(
        dataset_root_dir=args.dataset_root_dir,
        split="test",
        rotation_format=args.rotation_format,
        img_size=image_size,
        bbox_json_path=bbox_json_path,
        args=args,
        dataset_name=args.dataset,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    # Initialize the model
    merge_outputs = args.merge_outputs
    model = FastPoseViT(
        vit_model=args.vit_model,
        num_hidden_layers=args.num_hidden_layers,
        hidden_layer_dim=args.hidden_layer_dim,
        out_dim_translation=3,
        out_dim_rotation=4 if args.rotation_format == "quaternion" else 6,
        vit_weights=None,
        merge_outputs=merge_outputs,
        nb_class_tokens=args.nb_class_tokens,
        use_layer_norm=args.use_layer_norm,
        dropout_rate=args.dropout_rate,
        use_residual=args.use_residual,
        no_mlp=args.no_mlp,
    ).to(device)

    # Load model weights
    checkpoint = torch.load(args.model_weights, map_location=device, weights_only=True)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    # Remap legacy "vit." prefix to "backbone."
    state_dict = {
        k.replace("vit.", "backbone.", 1) if k.startswith("vit.") else k: v
        for k, v in state_dict.items()
    }
    model.load_state_dict(state_dict)

    logger.info(f"Loaded model weights from {args.model_weights}")

    # Set model to evaluation mode
    model.eval()

    # Initialize metric accumulators and results dictionary
    total_metrics = {
        "translation_metric": [],
        "rotation_metric": [],
        "total_metric": [],
        "relative_translation_metric": [],
        "relative_total_metric": [],
    }

    # Initialize separate metrics for SPEED+ sunlamp and lightbox if applicable
    sunlamp_metrics = None
    lightbox_metrics = None
    if args.dataset == "SPEED_PLUS":
        sunlamp_metrics = {
            "translation_metric": [],
            "rotation_metric": [],
            "total_metric": [],
            "relative_translation_metric": [],
            "relative_total_metric": [],
        }
        lightbox_metrics = {
            "translation_metric": [],
            "rotation_metric": [],
            "total_metric": [],
            "relative_translation_metric": [],
            "relative_total_metric": [],
        }

    results = {}

    # Evaluation loop
    sample_idx = 0  # Counter for unique sample numbering
    with torch.no_grad():
        for images, translations, rotations, bbox in tqdm(test_loader, desc="Evaluating"):
            # Move data to device
            images = images.to(device)

            translations = translations.to(device)
            rotations = rotations.to(device)

            # Forward pass
            if merge_outputs:
                output = model(images)
                # Split output into translations and rotations
                pred_translations = output[:, :3]
                pred_rotations = output[:, 3:]
            else:
                pred_translations, pred_rotations = model(images)

            if args.rotation_format == "matrix":
                pred_rotations = process_rotation(pred_rotations, args.rotation_format)

                # Convert both pred and ground truth rotations to quaternions for evaluation
                pred_rotations = rotation_matrix_to_quaternion(pred_rotations)
                rotations = rotation_matrix_to_quaternion(rotations)
            elif args.rotation_format == "quaternion":
                # Normalize quaternions if enabled and using quaternion format
                if args.normalize_quaternions and args.rotation_format == "quaternion":
                    pred_rotations = pred_rotations / torch.norm(
                        pred_rotations, dim=1, keepdim=True
                    )
            else:
                raise ValueError(f"Invalid rotation format: {args.rotation_format}")

            # Convert relative translations to absolute translations if bbox is not None
            if bbox is not None and not torch.all(bbox.eq(0)):
                # Send bbox to device
                bbox = bbox.to(device)

                # Convert relative translations to absolute translations (skip if using absolute mode)
                if not args.use_absolute_translation:
                    pred_translations = bbox_relative_translation_to_translation(
                        pred_translations,
                        bbox,
                        test_dataset.camera,
                    )
                    translations = bbox_relative_translation_to_translation(
                        translations,
                        bbox,
                        test_dataset.camera,
                    )

            # Convert rotation to centered rotation
            pred_rotations = get_absolute_orientation(
                translation=pred_translations,
                rotation_quat=pred_rotations,
                no_rotation_compensation=args.no_rotation_compensation,
            )
            rotations = get_absolute_orientation(
                translation=translations,
                rotation_quat=rotations,
                no_rotation_compensation=args.no_rotation_compensation,
            )

            # Compute metrics
            metrics = compute_metrics(pred_translations, translations, pred_rotations, rotations)

            # Accumulate metrics
            total_metrics["translation_metric"].append(metrics["translation_metric_mean"])
            total_metrics["rotation_metric"].append(metrics["rotation_metric_mean"])
            total_metrics["total_metric"].append(metrics["total_metric"])
            total_metrics["relative_translation_metric"].append(
                metrics["relative_translation_metric_mean"]
            )
            total_metrics["relative_total_metric"].append(metrics["relative_total_metric"])

            # Accumulate separate metrics for SPEED+ sunlamp and lightbox
            if args.dataset == "SPEED_PLUS":
                # Compute individual sample metrics (compute_metrics only returns batch means)
                from src.metrics import (
                    translation_metric,
                    rotation_metric,
                    relative_translation_metric,
                )

                individual_trans_metrics = translation_metric(pred_translations, translations)
                individual_rot_metrics = rotation_metric(pred_rotations, rotations)
                individual_relative_trans_metrics = relative_translation_metric(
                    pred_translations, translations
                )

                # Get filenames for current batch to determine dataset type
                batch_size = len(translations)
                for i in range(batch_size):
                    current_sample_idx = sample_idx + i
                    filename = test_dataset.data[current_sample_idx]["filename"]

                    # Determine if this is sunlamp or lightbox based on filename
                    if "_sunlamp.jpg" in filename:
                        current_metrics = sunlamp_metrics
                    elif "_lightbox.jpg" in filename:
                        current_metrics = lightbox_metrics
                    else:
                        # Skip if filename doesn't match expected pattern
                        continue

                    # Add individual sample metrics to appropriate category
                    trans_val = individual_trans_metrics[i].item()
                    rot_val = individual_rot_metrics[i].item()
                    rel_trans_val = individual_relative_trans_metrics[i].item()

                    current_metrics["translation_metric"].append(trans_val)
                    current_metrics["rotation_metric"].append(rot_val)
                    current_metrics["total_metric"].append(trans_val + rot_val)
                    current_metrics["relative_translation_metric"].append(rel_trans_val)
                    current_metrics["relative_total_metric"].append(rel_trans_val + rot_val)

            # Store predictions and ground truth using vectorized operations
            # Batch transfer to CPU once per batch instead of per sample
            pred_trans_cpu = pred_translations.cpu().numpy()
            pred_rot_cpu = pred_rotations.cpu().numpy()
            gt_trans_cpu = translations.cpu().numpy()
            gt_rot_cpu = rotations.cpu().numpy()

            # Vectorized results creation
            batch_size = len(translations)
            for i in range(batch_size):
                results[str(sample_idx + i)] = {
                    "prediction": {
                        "translation": pred_trans_cpu[i].tolist(),
                        "rotation": pred_rot_cpu[i].tolist(),
                    },
                    "ground_truth": {
                        "translation": gt_trans_cpu[i].tolist(),
                        "rotation": gt_rot_cpu[i].tolist(),
                    },
                }
            sample_idx += batch_size

    # Compute average metrics using vectorized operations
    avg_metrics = {}
    for k, v in total_metrics.items():
        if v:  # Only process non-empty lists
            # Convert to tensor and compute mean in one operation
            metrics_tensor = torch.tensor(v)
            avg_metrics[k] = metrics_tensor.mean().item()
        else:
            avg_metrics[k] = 0.0

    # Compute separate metrics for SPEED+ if applicable
    sunlamp_avg_metrics = {}
    lightbox_avg_metrics = {}
    if (
        args.dataset == "SPEED_PLUS"
        and sunlamp_metrics is not None
        and lightbox_metrics is not None
    ):
        # Compute sunlamp averages
        for k, v in sunlamp_metrics.items():
            if v:  # Only process non-empty lists
                metrics_tensor = torch.tensor(v)
                sunlamp_avg_metrics[k] = metrics_tensor.mean().item()
            else:
                sunlamp_avg_metrics[k] = 0.0

        # Compute lightbox averages
        for k, v in lightbox_metrics.items():
            if v:  # Only process non-empty lists
                metrics_tensor = torch.tensor(v)
                lightbox_avg_metrics[k] = metrics_tensor.mean().item()
            else:
                lightbox_avg_metrics[k] = 0.0

    # Add metrics to results as a special key
    results["metrics"] = {
        "translation_metric": float(avg_metrics["translation_metric"]),
        "rotation_metric": float(avg_metrics["rotation_metric"]),
        "total_metric": float(avg_metrics["total_metric"]),
        "relative_translation_metric": float(avg_metrics["relative_translation_metric"]),
        "relative_total_metric": float(avg_metrics["relative_total_metric"]),
    }

    # Add separate metrics for SPEED+ if applicable
    if args.dataset == "SPEED_PLUS" and sunlamp_avg_metrics and lightbox_avg_metrics:
        results["metrics"]["sunlamp"] = {
            "translation_metric": float(sunlamp_avg_metrics["translation_metric"]),
            "rotation_metric": float(sunlamp_avg_metrics["rotation_metric"]),
            "total_metric": float(sunlamp_avg_metrics["total_metric"]),
            "relative_translation_metric": float(
                sunlamp_avg_metrics["relative_translation_metric"]
            ),
            "relative_total_metric": float(sunlamp_avg_metrics["relative_total_metric"]),
        }
        results["metrics"]["lightbox"] = {
            "translation_metric": float(lightbox_avg_metrics["translation_metric"]),
            "rotation_metric": float(lightbox_avg_metrics["rotation_metric"]),
            "total_metric": float(lightbox_avg_metrics["total_metric"]),
            "relative_translation_metric": float(
                lightbox_avg_metrics["relative_translation_metric"]
            ),
            "relative_total_metric": float(lightbox_avg_metrics["relative_total_metric"]),
        }

    # Print results
    logger.info("\n\nEvaluation Results:")
    logger.info(f"Average Translation Metric: {avg_metrics['translation_metric']:.4f}")
    logger.info(f"Average Rotation Metric: {avg_metrics['rotation_metric']:.4f}")
    logger.info(f"Average Total Metric: {avg_metrics['total_metric']:.4f}")
    logger.info("\n")
    logger.info(
        f"Average Relative Translation Metric: {avg_metrics['relative_translation_metric']:.4f}"
    )
    logger.info(f"Average Rotation Metric: {avg_metrics['rotation_metric']:.4f}")
    logger.info(
        f"Average Relative Total Metric (SPEED score): {avg_metrics['relative_total_metric']:.4f}"
    )

    # Print separate metrics for SPEED+ if applicable
    if args.dataset == "SPEED_PLUS" and sunlamp_avg_metrics and lightbox_avg_metrics:
        logger.info("\n\n=== SPEED+ Sunlamp Results ===")
        logger.info(f"Sunlamp Translation Metric: {sunlamp_avg_metrics['translation_metric']:.4f}")
        logger.info(f"Sunlamp Rotation Metric: {sunlamp_avg_metrics['rotation_metric']:.4f}")
        logger.info(f"Sunlamp Total Metric: {sunlamp_avg_metrics['total_metric']:.4f}")
        logger.info(
            f"Sunlamp Relative Translation Metric: {sunlamp_avg_metrics['relative_translation_metric']:.4f}"
        )
        logger.info(
            f"Sunlamp Relative Total Metric (SPEED score): {sunlamp_avg_metrics['relative_total_metric']:.4f}"
        )

        logger.info("\n=== SPEED+ Lightbox Results ===")
        logger.info(
            f"Lightbox Translation Metric: {lightbox_avg_metrics['translation_metric']:.4f}"
        )
        logger.info(f"Lightbox Rotation Metric: {lightbox_avg_metrics['rotation_metric']:.4f}")
        logger.info(f"Lightbox Total Metric: {lightbox_avg_metrics['total_metric']:.4f}")
        logger.info(
            f"Lightbox Relative Translation Metric: {lightbox_avg_metrics['relative_translation_metric']:.4f}"
        )
        logger.info(
            f"Lightbox Relative Total Metric (SPEED score): {lightbox_avg_metrics['relative_total_metric']:.4f}"
        )

        # Log sample counts for verification
        logger.info(
            f"\nSample counts - Sunlamp: {len(sunlamp_metrics['total_metric'])}, Lightbox: {len(lightbox_metrics['total_metric'])}"
        )
        logger.info(
            f"Total samples: {len(sunlamp_metrics['total_metric']) + len(lightbox_metrics['total_metric'])}"
        )

    # Save results to JSON if output path is provided
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=4)
        logger.info(f"Saved predictions and ground-truth to {args.output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate FastPoseViT on SPEED dataset")

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
        help="Root directory of the dataset (auto-determined from dataset choice if not specified)",
    )
    parser.add_argument("--model_weights", type=str, help="Path to the trained model weights")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for data loading"
    )
    parser.add_argument(
        "--num_hidden_layers",
        type=int,
        default=5,
        help="Number of hidden layers in MLPWithProjection",
    )
    parser.add_argument(
        "--hidden_layer_dim",
        type=int,
        default=512,
        help="Dimension of hidden layers in MLPWithProjection",
    )
    parser.add_argument(
        "--rotation_format",
        type=str,
        choices=["quaternion", "matrix"],
        default="quaternion",
        help="Format for representing rotations",
    )
    parser.add_argument("--log_dir", type=str, default="./runs", help="Directory to store logs")
    parser.add_argument(
        "--vit_model",
        type=str,
        choices=SUPPORTED_VIT_MODELS,
        default="vit_b_16",
        help="ViT model to use",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        help="Path to save the predictions and ground-truth in JSON format",
    )
    parser.add_argument(
        "--nb_class_tokens",
        type=int,
        default=1,
        help="Number of class tokens to use in the model",
    )
    parser.add_argument(
        "--use_layer_norm",
        action="store_true",
        help="Use layer normalization in MLP layers",
    )
    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.0,
        help="Dropout rate for MLP layers (0.0 = no dropout)",
    )
    parser.add_argument(
        "--use_residual",
        action="store_true",
        help="Use residual connections in MLP layers",
    )
    parser.add_argument(
        "--normalize_quaternions",
        action="store_true",
        help="Normalize quaternion outputs to unit length",
    )

    # Experiments arguments
    # Don't use pixel data augmentation
    parser.add_argument(
        "--no_pixel_augmentation",
        action="store_true",
        help="Don't use pixel data augmentation",
    )
    # Don't use spatial data augmentation
    parser.add_argument(
        "--no_spatial_augmentation",
        action="store_true",
        help="Don't use spatial data augmentation",
    )
    # Use absolute translations instead of relative translations
    parser.add_argument(
        "--use_absolute_translation",
        action="store_true",
        help="Use absolute translations instead of relative translations for ablation study",
    )
    # Use full image instead of cropped image
    parser.add_argument(
        "--use_full_image",
        action="store_true",
        help="Use full image instead of cropped image for ablation study",
    )
    # Don't use rotation-compensating correction factor in loss
    parser.add_argument(
        "--no_rotation_compensation",
        action="store_true",
        help="Don't use rotation-compensating correction factor in loss",
    )
    # Use domain gap pixel augmentation
    parser.add_argument(
        "--domain_gap_pixel_augmentation",
        action="store_true",
        help="Use domain gap pixel augmentation with sun flare, blur, noise, compression, and dropout",
    )
    # Use style augmentation
    parser.add_argument(
        "--do_style_aug",
        action="store_true",
        help="Enable neural style augmentation (requires --domain_gap_pixel_augmentation)",
    )
    # Merge heads
    parser.add_argument(
        "--merge_outputs",
        action="store_true",
        help="Merge translation and rotation outputs into a single MLP",
    )
    # Skip MLP layers
    parser.add_argument(
        "--no_mlp",
        action="store_true",
        help="Skip MLP layers and use direct projection only",
    )
    # Don't pad cropped images to make them square
    parser.add_argument(
        "--no_crop_padding",
        action="store_true",
        help="Don't pad cropped images to make them square",
    )
    # Skip tensorboard logging
    parser.add_argument(
        "--skip_tensorboard",
        action="store_true",
        help="Skip tensorboard logging to save time and reduce I/O",
    )
    # Convert images to grayscale
    parser.add_argument(
        "--convert_to_grayscale",
        action="store_true",
        help="Convert images to grayscale (replicated as 3-channel for model compatibility)",
    )

    args = parser.parse_args()

    # Set dataset_root_dir based on dataset choice if not specified
    if args.dataset_root_dir is None:
        dataset_config = get_dataset_config(args.dataset)
        args.dataset_root_dir = f"./{dataset_config['folder']}"

    # Automatically determine bbox_json_path based on dataset
    dataset_config = get_dataset_config(args.dataset)
    bbox_json_path = os.path.join(args.dataset_root_dir, dataset_config["bbox_annotations"])

    evaluate(args, bbox_json_path)
