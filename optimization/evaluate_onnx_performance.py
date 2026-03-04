"""ONNX model performance evaluation on SPEED dataset."""

import argparse
import json
import os
import time
from types import TracebackType
from typing import Optional

import numpy as np
import onnxruntime as ort
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets import SPEEDDataset, get_dataset_config
from src.metrics import compute_metrics
from src.utils import (
    bbox_relative_translation_to_translation,
    get_absolute_orientation,
    process_rotation,
    rotation_matrix_to_quaternion,
    setup_logging,
)


class ONNXInference:
    """ONNX Runtime inference wrapper for pose estimation models.

    Provides a context manager interface for running inference with ONNX Runtime,
    supporting both CPU and GPU execution providers.
    """

    def __init__(
        self,
        onnx_path: str,
        providers: Optional[list[str | tuple[str, dict]]] = None,
    ) -> None:
        """Initialize the ONNX inference session.

        Args:
            onnx_path: Path to the ONNX model file.
            providers: List of ONNX Runtime execution providers. Defaults to
                CUDA with CPU fallback.
        """
        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.onnx_path = onnx_path
        self.providers = providers
        self.session: Optional[ort.InferenceSession] = None

    def load_model(self) -> "ONNXInference":
        """Load the ONNX model and create an inference session.

        Returns:
            Self for method chaining.
        """
        print("Loading ONNX model...")

        # Create inference session with specified providers
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.log_severity_level = 3  # Reduce logging to suppress warnings

        self.session = ort.InferenceSession(
            self.onnx_path, sess_options=sess_options, providers=self.providers
        )

        # Log model info
        print("ONNX model loaded successfully")
        print(f"Execution providers: {self.session.get_providers()}")

        # Log input/output info
        input_details = self.session.get_inputs()
        output_details = self.session.get_outputs()

        print("Model inputs:")
        for i, input_detail in enumerate(input_details):
            print(
                f"  {i}: {input_detail.name}, shape: {input_detail.shape}, dtype: {input_detail.type}"
            )

        print("Model outputs:")
        for i, output_detail in enumerate(output_details):
            print(
                f"  {i}: {output_detail.name}, shape: {output_detail.shape}, dtype: {output_detail.type}"
            )

        return self

    def inference(
        self, input_data: np.ndarray, get_timing: bool = False
    ) -> dict | tuple[dict, float]:
        """Run inference on the input data.

        Args:
            input_data: Input tensor as a numpy array.
            get_timing: Whether to measure and return inference time.

        Returns:
            If get_timing is False, returns a dictionary mapping output names
            to numpy arrays. If get_timing is True, returns a tuple of
            (output_dict, inference_time_seconds).
        """
        # Get input and output names
        input_names = [inp.name for inp in self.session.get_inputs()]
        output_names = [out.name for out in self.session.get_outputs()]

        # Prepare input dictionary
        input_dict = {input_names[0]: input_data}

        if get_timing:
            start = time.perf_counter()

        # Run inference
        outputs = self.session.run(output_names, input_dict)

        if get_timing:
            infer_time = time.perf_counter() - start

        # Convert to dictionary
        output_dict = {}
        for name, output in zip(output_names, outputs):
            output_dict[name] = output

        if get_timing:
            return output_dict, infer_time
        else:
            return output_dict

    def cleanup(self) -> None:
        """Clean up ONNX Runtime resources."""
        if self.session:
            self.session = None

    def __enter__(self) -> "ONNXInference":
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> bool:
        self.cleanup()
        return False


def parse_tuple(s: str) -> tuple[int, int]:
    """Parse an image size tuple from a comma-separated string.

    Args:
        s: String in format 'height,width' (e.g., '224,224').

    Returns:
        Tuple of (height, width) as integers.

    Raises:
        argparse.ArgumentTypeError: If the string format is invalid.
    """
    try:
        return tuple(map(int, s.split(",")))
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Tuple must be in format 'height,width' (e.g., '224,224')"
        )


def evaluate(args: argparse.Namespace) -> None:
    """Evaluate an ONNX model on the SPEED dataset.

    Runs inference on the test split and computes pose estimation metrics
    including translation error, rotation error, and SPEED score.

    Args:
        args: Command line arguments containing model path, dataset configuration,
            and evaluation parameters.
    """
    logger, _ = setup_logging(args=args, base_filename="evaluate_onnx_latency")
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

    # Get the image size for the ViT model (match PyTorch evaluation)
    from src.models import VIT_MODELS

    image_size = (
        VIT_MODELS[args.vit_model][3]
        if hasattr(args, "vit_model") and args.vit_model in VIT_MODELS
        else args.image_size
    )

    # Automatically determine bbox_json_path based on dataset (match PyTorch evaluation)
    dataset_config = get_dataset_config(args.dataset)
    bbox_json_path = os.path.join(args.dataset_root_dir, dataset_config["bbox_annotations"])

    # Load dataset with same parameters as PyTorch evaluation
    test_dataset = SPEEDDataset(
        dataset_root_dir=args.dataset_root_dir,
        split="test",
        rotation_format=args.rotation_format,  # Use command-line argument like PyTorch
        img_size=image_size,  # Use VIT model size like PyTorch
        bbox_json_path=bbox_json_path,  # Use auto-determined path like PyTorch
        args=args,
        dataset_name=args.dataset,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    try:
        # Initialize ONNX model with optimal GPU acceleration
        providers = []
        if torch.cuda.is_available():
            # Try TensorRT execution provider first (best performance)
            trt_provider_options = {
                "device_id": 0,
                "trt_max_workspace_size": 2 * 1024 * 1024 * 1024,  # 2GB
                "trt_fp16_enable": False,
                "trt_int8_enable": False,
                "trt_engine_cache_enable": True,
                "trt_timing_cache_enable": True,
            }
            providers.append(("TensorrtExecutionProvider", trt_provider_options))  # Fixed case

            # Fallback to CUDA execution provider
            cuda_provider_options = {
                "device_id": 0,
                "arena_extend_strategy": "kSameAsRequested",
                "gpu_mem_limit": 4 * 1024 * 1024 * 1024,  # 4GB
                "cudnn_conv_algo_search": "DEFAULT",
                "do_copy_in_default_stream": False,
                "cudnn_conv_use_max_workspace": True,
            }
            providers.append(("CUDAExecutionProvider", cuda_provider_options))

        providers.append("CPUExecutionProvider")

        with ONNXInference(args.onnx_model_path, providers=providers) as onnx_model:
            onnx_model.load_model()
            logger.info(f"Loaded ONNX model from {args.onnx_model_path}")

            # Initialize metric accumulators and results dictionary
            total_metrics = {
                "translation_metric": [],
                "rotation_metric": [],
                "total_metric": [],
                "relative_translation_metric": [],
                "relative_total_metric": [],
            }
            results = {}
            inference_times = []

            # Evaluation loop
            sample_idx = 0  # Counter for unique sample numbering
            with torch.no_grad():
                for images, translations, rotations, bbox in tqdm(test_loader, desc="Evaluating"):
                    try:
                        # Convert images to numpy and prepare for ONNX
                        images_np = images.numpy()

                        # Run inference with timing
                        outputs, batch_time = onnx_model.inference(images_np, get_timing=True)
                        inference_times.append(batch_time)

                        # Move ground truth to GPU
                        translations = translations.cuda()
                        rotations = rotations.cuda()

                        # Process ONNX outputs
                        if args.merge_outputs:
                            output = torch.from_numpy(outputs["output"]).cuda()
                            pred_translations = output[:, :3]
                            pred_rotations = output[:, 3:]
                        else:
                            pred_translations = torch.from_numpy(outputs["translation"]).cuda()
                            pred_rotations = torch.from_numpy(outputs["rotation"]).cuda()

                        # Process rotations based on format (match PyTorch evaluation exactly)
                        if args.rotation_format == "matrix" or pred_rotations.shape[1] == 6:
                            # Convert 6D representation to rotation matrix
                            pred_rotations = process_rotation(pred_rotations, "matrix")

                            # Convert both pred and ground truth rotations to quaternions for evaluation
                            pred_rotations = rotation_matrix_to_quaternion(pred_rotations)
                            rotations = rotation_matrix_to_quaternion(rotations)

                        elif args.rotation_format == "quaternion":
                            if args.normalize_quaternions:
                                pred_rotations = pred_rotations / torch.norm(
                                    pred_rotations, dim=1, keepdim=True
                                )

                        # Convert relative translations to absolute translations if bbox is not None
                        # Check that bbox is not made only of zeros and 1s
                        if bbox is not None and not torch.all(bbox.eq(0)):
                            bbox = bbox.cuda()
                            pred_translations = bbox_relative_translation_to_translation(
                                pred_translations,
                                bbox,
                                test_dataset.camera,
                                args.no_translation_compensation,
                                args.new_Z,
                            )
                            translations = bbox_relative_translation_to_translation(
                                translations,
                                bbox,
                                test_dataset.camera,
                                args.no_translation_compensation,
                                args.new_Z,
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
                        metrics = compute_metrics(
                            pred_translations, translations, pred_rotations, rotations
                        )

                        # Accumulate metrics
                        total_metrics["translation_metric"].append(
                            metrics["translation_metric_mean"]
                        )
                        total_metrics["rotation_metric"].append(metrics["rotation_metric_mean"])
                        total_metrics["total_metric"].append(metrics["total_metric"])
                        total_metrics["relative_translation_metric"].append(
                            metrics["relative_translation_metric_mean"]
                        )
                        total_metrics["relative_total_metric"].append(
                            metrics["relative_total_metric"]
                        )

                        # Store predictions and ground truth
                        for i in range(len(translations)):
                            results[str(sample_idx)] = {
                                "prediction": {
                                    "translation": pred_translations[i].cpu().tolist(),
                                    "rotation": pred_rotations[i].cpu().tolist(),
                                },
                                "ground_truth": {
                                    "translation": translations[i].cpu().tolist(),
                                    "rotation": rotations[i].cpu().tolist(),
                                },
                            }
                            sample_idx += 1

                    except Exception as e:
                        logger.error(f"Error processing batch {sample_idx}: {e}")
                        # Continue to next batch instead of failing completely
                        for i in range(len(translations)):
                            sample_idx += 1
                        continue

            # Check if we have any valid results
            if not total_metrics["total_metric"]:
                logger.error("No valid results obtained during evaluation")
                return

            # Compute final metrics
            avg_metrics = {k: sum(v) / len(v) for k, v in total_metrics.items()}

            # Log results
            logger.info("\nEvaluation Results:")
            logger.info(f"Average Translation Metric: {avg_metrics['translation_metric']:.4f}")
            logger.info(f"Average Rotation Metric: {avg_metrics['rotation_metric']:.4f}")
            logger.info(f"Average Total Metric: {avg_metrics['total_metric']:.4f}")
            logger.info("")

            logger.info(
                f"Average Relative Translation Metric: {avg_metrics['relative_translation_metric']:.4f}"
            )
            logger.info(f"Average Rotation Metric: {avg_metrics['rotation_metric']:.4f}")
            logger.info(
                f"Average Relative Total Metric (SPEED score): {avg_metrics['relative_total_metric']:.4f}"
            )

            # Log inference performance
            avg_inference_time = sum(inference_times) / len(inference_times)
            std_inference_time = torch.std(torch.tensor(inference_times)).item()
            throughput = 1.0 / avg_inference_time * args.batch_size

            logger.info("\nInference Performance:")
            logger.info(f"Average inference time: {avg_inference_time * 1000:.2f} ms")
            logger.info(f"Inference time std: {std_inference_time * 1000:.2f} ms")
            logger.info(f"Throughput: {throughput:.2f} FPS")

            # Save results to JSON if requested
            if args.output_json:
                output_data = {
                    "metrics": avg_metrics,
                    "inference_performance": {
                        "avg_inference_time_ms": avg_inference_time * 1000,
                        "std_inference_time_ms": std_inference_time * 1000,
                        "throughput_fps": throughput,
                    },
                    "predictions": results,
                }
                with open(args.output_json, "w") as f:
                    json.dump(output_data, f, indent=2)
                logger.info(f"Saved results to {args.output_json}")

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ONNX model on SPEED dataset")

    # Modified image_size argument to use the parse_tuple function
    parser.add_argument(
        "--image_size",
        type=parse_tuple,
        default=(384, 384),
        help="Input image size as height,width (e.g., '384,384')",
    )

    # ONNX model path
    parser.add_argument(
        "--onnx_model_path",
        type=str,
        required=True,
        help="Path to the ONNX model file",
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["SPEED", "SPEED_PLUS_SYNTHETIC"],
        default="SPEED",
        help="Dataset to use for evaluation",
    )
    parser.add_argument(
        "--dataset_root_dir",
        type=str,
        default=None,
        help="Root directory of the dataset (auto-determined from dataset choice if not specified)",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for data loading"
    )
    parser.add_argument(
        "--log_dir", type=str, default="./runs", help="Directory to store TensorBoard logs"
    )
    parser.add_argument(
        "--output_json",
        type=str,
        help="Path to save the predictions and ground-truth in JSON format",
    )
    parser.add_argument(
        "--bbox_json_path",
        type=str,
        default=None,
        help="Path to the JSON file containing bounding box information",
    )
    parser.add_argument(
        "--rotation_format",
        type=str,
        choices=["quaternion", "matrix"],
        default="quaternion",
        help="Format for representing rotations",
    )
    parser.add_argument(
        "--vit_model",
        type=str,
        default="vit_b_16_384",
        help="ViT model to determine correct image size",
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
    # Don't use translation-compensating correction factor in loss
    parser.add_argument(
        "--no_translation_compensation",
        action="store_true",
        help="Don't use translation-compensating correction factor in loss",
    )
    # Don't use rotation-compensating correction factor in loss
    parser.add_argument(
        "--no_rotation_compensation",
        action="store_true",
        help="Don't use rotation-compensating correction factor in loss",
    )
    # Merge heads
    parser.add_argument(
        "--merge_outputs",
        action="store_true",
        help="Merge translation and rotation outputs into a single MLP",
    )

    # Model architecture parameters
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
        "--no_mlp",
        action="store_true",
        help="Skip MLP layers and use direct projection only",
    )

    # Data processing parameters
    parser.add_argument(
        "--normalize_quaternions",
        action="store_true",
        help="Normalize quaternion outputs to unit length",
    )
    parser.add_argument(
        "--new_Z",
        action="store_true",
        help="Use least squares solution for avg_ratio computation instead of geometric mean",
    )
    parser.add_argument(
        "--no_crop_padding",
        action="store_true",
        help="Don't pad cropped images to make them square",
    )

    # Augmentation parameters
    parser.add_argument(
        "--domain_gap_pixel_augmentation",
        action="store_true",
        help="Use domain gap pixel augmentation with sun flare, blur, noise, compression, and dropout",
    )
    parser.add_argument(
        "--do_style_aug",
        action="store_true",
        help="Enable neural style augmentation (requires --domain_gap_pixel_augmentation)",
    )

    # Utility parameters
    parser.add_argument(
        "--skip_tensorboard",
        action="store_true",
        help="Skip tensorboard logging to save time and reduce I/O",
    )

    args = parser.parse_args()

    # Set dataset_root_dir based on dataset choice if not specified
    if args.dataset_root_dir is None:
        dataset_config = get_dataset_config(args.dataset)
        args.dataset_root_dir = f"./{dataset_config['folder']}"

    evaluate(args)
