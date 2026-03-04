"""PyTorch model latency evaluation on SPEED dataset."""

import argparse
import os
import time

import numpy as np
import torch

# Fix torch.compile Docker UID issues by setting comprehensive cache directories
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/tmp/torchinductor_cache")
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_cache")
os.environ.setdefault("PYTORCH_TRITON_CACHE_DIR", "/tmp/triton_cache")
# Set user environment variables to avoid getpwuid lookups
os.environ.setdefault("USER", "pytorch")
os.environ.setdefault("LOGNAME", "pytorch")
os.environ.setdefault("HOME", "/tmp")
import torch._dynamo
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets import SPEEDDataset, get_dataset_config
from src.models import FastPoseViT, SUPPORTED_VIT_MODELS, VIT_MODELS
from src.utils import setup_logging


def measure_latency(args: argparse.Namespace, bbox_json_path: str) -> None:
    """Measure inference latency of a PyTorch pose estimation model.

    Runs the model on the SPEED dataset test split and collects timing statistics
    including mean, median, and percentile latencies.

    Args:
        args: Command line arguments containing model configuration, paths,
            and performance optimization settings.
        bbox_json_path: Path to the JSON file containing bounding box annotations.
    """
    # Set up logging
    logger, _ = setup_logging(args=args, base_filename="evaluate_torch_latency")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # If log_dir does not exist, create it
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # Get the image size for the ViT model
    image_size = VIT_MODELS[args.vit_model][3] if args.vit_model in VIT_MODELS else (224, 224)

    # Load the dataset
    val_dataset = SPEEDDataset(
        dataset_root_dir=args.dataset_root_dir,
        split="test",
        rotation_format=args.rotation_format,
        img_size=image_size,
        bbox_json_path=bbox_json_path,
        args=args,
        dataset_name=args.dataset,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    # Initialize the model
    model = FastPoseViT(
        vit_model=args.vit_model,
        num_hidden_layers=args.num_hidden_layers,
        hidden_layer_dim=args.hidden_layer_dim,
        out_dim_translation=3,
        out_dim_rotation=4 if args.rotation_format == "quaternion" else 6,
        vit_weights=None,
        merge_outputs=args.merge_outputs,
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

    # Apply torch.compile optimization if enabled
    if args.enable_torch_compile:
        logger.info("Applying torch.compile optimization...")
        logger.info("Note: First few steps will be slower due to compilation warmup")

        # Simple warmup for torch.compile
        if device.type == "cuda":
            logger.info("Performing model warmup for torch.compile...")
            with torch.no_grad():
                image_size = (
                    VIT_MODELS[args.vit_model][3] if args.vit_model in VIT_MODELS else (224, 224)
                )
                dummy_input = torch.randn(args.batch_size, 3, *image_size, device=device)
                try:
                    _ = model(dummy_input)
                    logger.info("Model warmup successful")
                except Exception as e:
                    logger.warning(f"Model warmup failed: {e}")

        try:
            # Enable error suppression for Docker environments
            torch._dynamo.config.suppress_errors = True

            model = torch.compile(model, mode="max-autotune")
            logger.info("Model compiled with torch.compile (max-autotune mode)")
        except (KeyError, RuntimeError, OSError) as e:
            if "getpwuid" in str(e) or "uid not found" in str(e):
                logger.warning(
                    "torch.compile failed due to Docker UID issue (getpwuid error). "
                    "Falling back to uncompiled model. Consider running Docker with "
                    "-e USER=pytorch -e LOGNAME=pytorch for optimal performance."
                )
            elif "Triton" in str(e):
                logger.warning(
                    f"torch.compile failed due to Triton compilation issue: {e}. "
                    "Falling back to uncompiled model."
                )
            else:
                logger.warning(
                    f"torch.compile failed with error: {e}. Falling back to uncompiled model."
                )
            # Continue with uncompiled model

    # Warmup phase
    if args.warmup_iterations > 0:
        logger.info(f"Performing {args.warmup_iterations} warmup iterations...")
        with torch.inference_mode():
            warmup_count = 0
            for images, _, _, _ in val_loader:
                if warmup_count >= args.warmup_iterations:
                    break
                images = images.to(device)
                if args.mixed_precision and device.type == "cuda":
                    with torch.autocast("cuda", dtype=torch.float16):
                        _ = model(images)
                else:
                    _ = model(images)
                warmup_count += 1
                if device.type == "cuda":
                    torch.cuda.synchronize()
        logger.info("Warmup completed")

    # Clear cache for accurate memory measurements
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Measure latency
    logger.info("Measuring latency...")
    latencies = []

    # Initialize CUDA events for GPU timing if enabled
    if args.use_cuda_events and device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

    with torch.inference_mode():
        for images, _, _, _ in tqdm(val_loader, desc="Computing latency"):
            images = images.to(device)
            batch_latencies = []

            # Multiple timing runs per sample for statistical accuracy
            for _ in range(args.timing_runs_per_sample):
                if args.use_cuda_events and device.type == "cuda":
                    # GPU timing with CUDA events (more accurate)
                    torch.cuda.synchronize()
                    start_event.record()

                    if args.mixed_precision:
                        with torch.autocast("cuda", dtype=torch.float16):
                            _ = model(images)
                    else:
                        _ = model(images)

                    end_event.record()
                    torch.cuda.synchronize()
                    latency = start_event.elapsed_time(end_event)  # Already in milliseconds
                else:
                    # CPU timing fallback
                    if device.type == "cuda":
                        torch.cuda.synchronize()

                    start_time = time.perf_counter()

                    if args.mixed_precision and device.type == "cuda":
                        with torch.autocast("cuda", dtype=torch.float16):
                            _ = model(images)
                    else:
                        _ = model(images)

                    if device.type == "cuda":
                        torch.cuda.synchronize()

                    end_time = time.perf_counter()
                    latency = (end_time - start_time) * 1000  # Convert to milliseconds

                batch_latencies.append(latency)

            # Use median of multiple runs for robustness
            median_latency = np.median(batch_latencies)
            latencies.append(median_latency)

    # Compute statistics
    latencies = np.array(latencies)
    mean_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    median_latency = np.median(latencies)
    percentile_95 = np.percentile(latencies, 95)
    percentile_99 = np.percentile(latencies, 99)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)

    # Compute throughput statistics
    total_samples = len(val_dataset)
    total_batches = len(latencies)
    samples_per_batch = args.batch_size
    mean_fps = 1000 / mean_latency  # FPS based on mean latency
    max_throughput = samples_per_batch * mean_fps  # Samples per second

    # Performance configuration summary
    timing_method = (
        "CUDA Events" if (args.use_cuda_events and device.type == "cuda") else "CPU Timing"
    )
    precision = "Mixed (FP16)" if args.mixed_precision else "Full (FP32)"
    compilation = "Enabled" if args.enable_torch_compile else "Disabled"

    # Print comprehensive results
    logger.info("")
    logger.info("=" * 60)
    logger.info("PERFORMANCE EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Configuration:")
    logger.info(f"  Model: {args.vit_model}")
    logger.info(f"  Batch Size: {args.batch_size}")
    logger.info(f"  Device: {device}")
    logger.info(f"  Precision: {precision}")
    logger.info(f"  Torch Compile: {compilation}")
    logger.info(f"  Timing Method: {timing_method}")
    logger.info(f"  Warmup Iterations: {args.warmup_iterations}")
    logger.info(f"  Timing Runs per Sample: {args.timing_runs_per_sample}")

    logger.info("")
    logger.info("Dataset:")
    logger.info(f"  Total Samples: {total_samples}")
    logger.info(f"  Total Batches: {total_batches}")

    logger.info("")
    logger.info("Latency Statistics (per batch, milliseconds):")
    logger.info(f"  Mean: {mean_latency:.2f} ± {std_latency:.2f} ms")
    logger.info(f"  Median: {median_latency:.2f} ms")
    logger.info(f"  Min: {min_latency:.2f} ms")
    logger.info(f"  Max: {max_latency:.2f} ms")
    logger.info(f"  95th Percentile: {percentile_95:.2f} ms")
    logger.info(f"  99th Percentile: {percentile_99:.2f} ms")

    logger.info("")
    logger.info("Throughput Statistics:")
    logger.info(f"  FPS (per batch): {mean_fps:.2f}")
    logger.info(f"  Samples/second: {max_throughput:.2f}")
    logger.info(f"  Batches/second: {1000 / mean_latency:.2f}")

    logger.info("")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure FastPoseViT latency on SPEED dataset")

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
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation")
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
    parser.add_argument(
        "--log_dir", type=str, default="./runs", help="Directory to store TensorBoard logs"
    )
    parser.add_argument(
        "--vit_model",
        type=str,
        choices=SUPPORTED_VIT_MODELS,
        default="vit_b_16",
        help="ViT model to use",
    )
    parser.add_argument(
        "--nb_class_tokens",
        type=int,
        default=1,
        help="Number of class tokens to use in the model",
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
    # Use new Z computation (least squares solution)
    parser.add_argument(
        "--new_Z",
        action="store_true",
        help="Use least squares solution for avg_ratio computation instead of geometric mean",
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

    # Performance optimization arguments
    parser.add_argument(
        "--enable_torch_compile",
        action="store_true",
        help="Enable torch.compile optimization for faster inference",
    )
    parser.add_argument(
        "--warmup_iterations",
        type=int,
        default=10,
        help="Number of warmup iterations before timing measurements",
    )
    parser.add_argument(
        "--timing_runs_per_sample",
        type=int,
        default=3,
        help="Number of timing runs per sample for statistical accuracy",
    )
    parser.add_argument(
        "--use_cuda_events",
        action="store_true",
        help="Use CUDA events for GPU timing (more accurate than CPU timing)",
    )
    parser.add_argument(
        "--mixed_precision",
        action="store_true",
        help="Use mixed precision (FP16) for faster inference",
    )

    args = parser.parse_args()

    # Set dataset_root_dir based on dataset choice if not specified
    if args.dataset_root_dir is None:
        dataset_config = get_dataset_config(args.dataset)
        args.dataset_root_dir = f"./{dataset_config['folder']}"

    # Automatically determine bbox_json_path based on dataset
    dataset_config = get_dataset_config(args.dataset)
    bbox_json_path = os.path.join(args.dataset_root_dir, dataset_config["bbox_annotations"])

    measure_latency(args, bbox_json_path)
