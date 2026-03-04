# This script generates predictions on the SPEED competition dataset (test + real_test)

import argparse
import torch
from torch.utils.data import DataLoader
from src.datasets import SPEEDDataset, get_dataset_config
from src.models import FastPoseViT, SUPPORTED_VIT_MODELS, VIT_MODELS
from src.utils import (
    setup_logging,
    process_rotation,
    bbox_relative_translation_to_translation,
    rotation_matrix_to_quaternion,
    get_absolute_orientation,
)
from tqdm import tqdm
import os
import csv
from datetime import datetime


class SubmissionWriter:
    """Collects pose estimation results and exports them for SPEED competition submission.

    From: https://gitlab.com/EuropeanSpaceAgency/speed-utils/-/blob/master/submission.py?ref_type=heads
    """

    def __init__(self) -> None:
        self.test_results: list[dict] = []
        self.real_test_results: list[dict] = []

    def _append(self, filename: str, q: list[float], r: list[float], real: bool) -> None:
        """Append pose estimation result to internal storage.

        Args:
            filename: Image filename.
            q: Quaternion as [q0, q1, q2, q3].
            r: Translation vector as [r0, r1, r2].
            real: Whether this is a real test image.
        """
        if real:
            self.real_test_results.append({"filename": filename, "q": list(q), "r": list(r)})
        else:
            self.test_results.append({"filename": filename, "q": list(q), "r": list(r)})

    def append_test(self, filename: str, q: list[float], r: list[float]) -> None:
        """Append pose estimation for test image to submission.

        Args:
            filename: Image filename.
            q: Quaternion as [q0, q1, q2, q3].
            r: Translation vector as [r0, r1, r2].
        """
        self._append(filename, q, r, real=False)

    def append_real_test(self, filename: str, q: list[float], r: list[float]) -> None:
        """Append pose estimation for real image to submission.

        Args:
            filename: Image filename.
            q: Quaternion as [q0, q1, q2, q3].
            r: Translation vector as [r0, r1, r2].
        """
        self._append(filename, q, r, real=True)

    def export(self, out_dir: str = "", suffix: str | None = None) -> str:
        """Export submission CSV file containing collected pose estimates.

        Args:
            out_dir: Output directory for the submission file.
            suffix: Suffix for the output filename. Defaults to timestamp.

        Returns:
            Path to the exported submission file.
        """
        sorted_test = sorted(self.test_results, key=lambda k: k["filename"])
        sorted_real_test = sorted(self.real_test_results, key=lambda k: k["filename"])
        timestamp = datetime.now().strftime("%Y%m%d-%H%M")
        if suffix is None:
            suffix = timestamp
        submission_path = os.path.join(out_dir, "submission_{}.csv".format(suffix))
        with open(submission_path, "w") as f:
            csv_writer = csv.writer(f, lineterminator="\n")
            for result in sorted_test + sorted_real_test:
                csv_writer.writerow([result["filename"], *(result["q"] + result["r"])])

        print("Submission saved to {}.".format(submission_path))
        return submission_path


def predict(args: argparse.Namespace) -> None:
    """
    Generate predictions on the SPEED competition dataset (test + real_test merged)

    Args:
        args (argparse.Namespace): Command-line arguments

    Returns:
        None
    """
    # Set up logging
    logger, _ = setup_logging(args, base_filename="predict")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create output directory if it doesn't exist
    if args.output_csv:
        output_dir = os.path.dirname(args.output_csv)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

    # Get the image size for the ViT model
    image_size = VIT_MODELS[args.vit_model][3] if args.vit_model in VIT_MODELS else (224, 224)
    logger.info(f"Using ViT model: {args.vit_model} with image size: {image_size}")

    # Determine which split to use
    split = "test" if args.use_test_split else "to_predict"

    # Load the dataset
    prediction_dataset = SPEEDDataset(
        dataset_root_dir=args.dataset_root_dir,
        split=split,  # Use test split or merged prediction dataset
        rotation_format=args.rotation_format,
        img_size=image_size,
        bbox_json_path=args.bbox_json_path,
        args=args,
        dataset_name=args.dataset,
    )
    prediction_loader = DataLoader(
        prediction_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    logger.info(f"Loaded {split} dataset with {len(prediction_dataset)} images")
    logger.info(f"Using bbox annotations from: {args.bbox_json_path}")

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

    # Initialize submission writer
    submission = SubmissionWriter()

    # Get original filenames for proper mapping
    original_data = prediction_dataset.data

    sample_idx = 0  # Counter for tracking samples
    with torch.no_grad():
        for images, translations, rotations, bbox in tqdm(
            prediction_loader, desc="Generating predictions"
        ):
            # Move data to device
            images = images.to(device)
            bbox = bbox.to(device) if bbox is not None else None

            if bbox is None:
                logger.error(
                    "CRITICAL: Bounding box data is missing! This will cause incorrect predictions."
                )
                logger.error(
                    "Please ensure bbox_json_path is correct and contains bounding box data for all images."
                )
                raise ValueError("Missing bbox data - predictions would be inaccurate")

            # Forward pass
            if merge_outputs:
                output = model(images)
                # Split output into translations and rotations
                pred_translations = output[:, :3]
                pred_rotations = output[:, 3:]
            else:
                pred_translations, pred_rotations = model(images)

            # Process rotation format
            if args.rotation_format == "matrix":
                pred_rotations = process_rotation(pred_rotations, args.rotation_format)
                # Convert rotation matrices to quaternions for submission
                pred_rotations = rotation_matrix_to_quaternion(pred_rotations)
            elif args.rotation_format == "quaternion":
                # Normalize quaternions if enabled
                if args.normalize_quaternions:
                    pred_rotations = pred_rotations / torch.norm(
                        pred_rotations, dim=1, keepdim=True
                    )
            else:
                raise ValueError(f"Invalid rotation format: {args.rotation_format}")

            # Convert relative translations to absolute translations if bbox is available
            if bbox is not None and not torch.all(bbox.eq(0)):
                pred_translations = bbox_relative_translation_to_translation(
                    pred_translations,
                    bbox,
                    prediction_dataset.camera,
                )

            # Convert absolute quaternion to apparent (relative to the camera)
            pred_rotations = get_absolute_orientation(
                pred_translations, pred_rotations, args.no_rotation_compensation
            )

            # Move predictions to CPU for CSV serialization
            pred_trans_cpu = pred_translations.cpu()
            pred_rot_cpu = pred_rotations.cpu()

            # Store predictions with original filenames using SubmissionWriter
            batch_size = len(pred_translations)
            for i in range(batch_size):
                filename = original_data[sample_idx + i]["filename"]
                q = pred_rot_cpu[i].tolist()
                r = pred_trans_cpu[i].tolist()

                # Determine if this is a real test image (only relevant for to_predict split)
                is_real = "real" in filename
                if is_real:
                    submission.append_real_test(filename, q, r)
                else:
                    submission.append_test(filename, q, r)
            sample_idx += batch_size

    logger.info(
        f"Generated predictions for {len(submission.test_results + submission.real_test_results)} images"
    )
    logger.info(
        f"Test images: {len(submission.test_results)}, Real test images: {len(submission.real_test_results)}"
    )

    # Export submission CSV
    if args.output_csv:
        output_dir = os.path.dirname(args.output_csv) if os.path.dirname(args.output_csv) else "."
        suffix = (
            os.path.splitext(os.path.basename(args.output_csv))[0] if args.output_csv else None
        )
        submission_path = submission.export(out_dir=output_dir, suffix=suffix)
        logger.info(f"Saved submission to {submission_path}")

    logger.info("Prediction generation completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate predictions on SPEED competition dataset"
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset",
        type=str,
        default="SPEED",
        choices=["SPEED", "SPEED_PLUS", "SPEED_PLUS_SYNTHETIC"],
        help="Dataset to use",
    )
    parser.add_argument(
        "--dataset_root_dir",
        type=str,
        default="./SPEED_FIXED",
        help="Root directory of the dataset",
    )
    parser.add_argument(
        "--bbox_json_path",
        type=str,
        default=None,
        help="Path to the JSON file containing bounding box information",
    )

    # Model arguments
    parser.add_argument(
        "--model_weights", type=str, required=True, help="Path to the model weights file"
    )
    parser.add_argument(
        "--vit_model",
        type=str,
        default="base",
        choices=SUPPORTED_VIT_MODELS,
        help="ViT model variant to use",
    )
    parser.add_argument(
        "--rotation_format",
        type=str,
        default="quaternion",
        choices=["quaternion", "matrix"],
        help="Rotation representation format",
    )
    parser.add_argument(
        "--merge_outputs",
        action="store_true",
        help="Merge translation and rotation outputs into a single tensor",
    )
    parser.add_argument(
        "--normalize_quaternions",
        action="store_true",
        help="Normalize quaternions during inference",
    )

    # Model architecture arguments
    parser.add_argument("--num_hidden_layers", type=int, default=2, help="Number of hidden layers")
    parser.add_argument("--hidden_layer_dim", type=int, default=512, help="Hidden layer dimension")
    parser.add_argument("--nb_class_tokens", type=int, default=1, help="Number of class tokens")
    parser.add_argument("--use_layer_norm", action="store_true", help="Use layer normalization")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--use_residual", action="store_true", help="Use residual connections")
    parser.add_argument("--no_mlp", action="store_true", help="Do not use MLP in the head")

    # Data processing arguments
    parser.add_argument(
        "--no_pixel_augmentation",
        action="store_true",
        help="Disable pixel augmentations during inference",
    )
    parser.add_argument(
        "--no_spatial_augmentation",
        action="store_true",
        help="Disable spatial augmentations during inference",
    )
    parser.add_argument(
        "--domain_gap_pixel_augmentation",
        action="store_true",
        help="Use domain gap pixel augmentations",
    )
    parser.add_argument(
        "--no_rotation_compensation",
        action="store_true",
        help="Disable rotation compensation",
    )
    parser.add_argument(
        "--convert_to_grayscale", action="store_true", help="Convert images to grayscale"
    )
    parser.add_argument(
        "--use_test_split",
        action="store_true",
        help="Use the regular test split instead of to_predict (merged test + real_test)",
    )

    # Inference arguments
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")

    # Output arguments
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path to save the predictions in CSV format for competition submission",
    )
    parser.add_argument(
        "--log_dir", type=str, default="./logs", help="Directory to save log files"
    )

    args = parser.parse_args()

    # Validate required arguments
    if not os.path.exists(args.model_weights):
        print("Error: --model_weights must be provided and exist")
        exit(1)

    if not os.path.exists(args.dataset_root_dir):
        print(f"Error: Dataset root directory does not exist: {args.dataset_root_dir}")
        exit(1)

    # Automatically determine bbox_json_path based on dataset (same as evaluate.py)
    if args.bbox_json_path is None:
        dataset_config = get_dataset_config(args.dataset)
        args.bbox_json_path = os.path.join(
            args.dataset_root_dir, dataset_config["bbox_annotations"]
        )
        print(f"Auto-determined bbox_json_path: {args.bbox_json_path}")

    # Validate that bbox annotations file exists
    if not os.path.exists(args.bbox_json_path):
        print(f"Error: Bbox annotations file does not exist: {args.bbox_json_path}")
        print(
            "This file is required for accurate predictions. Please ensure bbox annotations are available."
        )
        exit(1)

    # Check if required dataset files exist
    if args.use_test_split:
        # Check if test.json exists for regular test split
        test_json = os.path.join(args.dataset_root_dir, "test.json")
        if not os.path.exists(test_json):
            print(f"Error: test.json does not exist at: {test_json}")
            exit(1)
    else:
        # Check if to_predict.json exists for merged prediction dataset
        to_predict_json = os.path.join(args.dataset_root_dir, "to_predict.json")
        if not os.path.exists(to_predict_json):
            print(f"Error: to_predict.json does not exist at: {to_predict_json}")
            print(
                "Please run the speed_prediction_merger.py script first to create the prediction dataset"
            )
            exit(1)

    predict(args)
