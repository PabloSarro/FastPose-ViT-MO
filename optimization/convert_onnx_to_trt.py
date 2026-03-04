#!/usr/bin/env python3
"""TensorRT ONNX conversion script using TensorRT Python API.

This approach ensures version compatibility with the host system.
"""

import argparse
import logging
import os
import sys
from typing import Optional

import glob
import numpy as np
import pycuda.driver as cuda
from PIL import Image
import tensorrt as trt
import torchvision.transforms as transforms

from object_detector.constants import LWDETR_IMG_SIZES
from src.models import SUPPORTED_VIT_MODELS, VIT_MODELS
from src.utils import setup_logging


class ImageNetCalibrator(trt.IInt8EntropyCalibrator2):
    """INT8 calibrator for Vision Transformer models using ImageNet-like data.

    Provides calibration data for TensorRT INT8 quantization by loading and
    preprocessing images from a directory or generating synthetic data.
    """

    def __init__(
        self,
        calibration_data_path: str,
        cache_file: str,
        batch_size: int,
        input_shape: tuple[int, int, int, int],
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initialize the calibrator.

        Args:
            calibration_data_path: Path to directory containing calibration images.
            cache_file: Path to save/load calibration cache file.
            batch_size: Number of images per calibration batch.
            input_shape: Input tensor shape as (batch, channels, height, width).
            logger: Optional logger for status messages.
        """
        trt.IInt8EntropyCalibrator2.__init__(self)

        self.calibration_data_path = calibration_data_path
        self.cache_file = cache_file
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.logger = logger

        # Calculate input size in bytes
        _, channels, height, width = input_shape
        self.input_size = batch_size * channels * height * width * np.float32().itemsize

        # Allocate device memory for calibration
        self.device_input = cuda.mem_alloc(self.input_size)

        # Load and prepare calibration data
        self.calibration_data = self._load_calibration_data()
        self.current_index = 0

        if self.logger:
            self.logger.info(f"Calibrator initialized with {len(self.calibration_data)} images")
            self.logger.info(f"Input shape: {input_shape}, Batch size: {batch_size}")

    def _load_calibration_data(self) -> list[np.ndarray]:
        """Load and preprocess calibration images from disk.

        Returns:
            List of numpy arrays, each containing a batch of preprocessed images.
        """
        calibration_data = []

        if not os.path.exists(self.calibration_data_path):
            if self.logger:
                self.logger.warning(
                    f"Calibration data path not found: {self.calibration_data_path}"
                )
                self.logger.info("Generating synthetic calibration data")
            return self._generate_synthetic_data()

        # Define image preprocessing transforms
        _, channels, height, width = self.input_shape
        transform = transforms.Compose(
            [
                transforms.Resize((height, width)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # Load images from directory
        image_files = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            image_files.extend(glob.glob(os.path.join(self.calibration_data_path, ext)))
            image_files.extend(glob.glob(os.path.join(self.calibration_data_path, ext.upper())))

        if not image_files:
            if self.logger:
                self.logger.warning("No image files found in calibration directory")
                self.logger.info("Generating synthetic calibration data")
            return self._generate_synthetic_data()

        # Process images in batches
        for i in range(0, min(len(image_files), 1000), self.batch_size):  # Limit to 1000 images
            batch_images = []

            for j in range(min(self.batch_size, len(image_files) - i)):
                try:
                    img_path = image_files[i + j]
                    img = Image.open(img_path).convert("RGB")
                    img_tensor = transform(img)
                    batch_images.append(img_tensor.numpy())
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"Failed to load image {image_files[i + j]}: {e}")
                    # Fill with zeros if image loading fails
                    batch_images.append(np.zeros((channels, height, width), dtype=np.float32))

            # Pad batch if needed
            while len(batch_images) < self.batch_size:
                batch_images.append(np.zeros((channels, height, width), dtype=np.float32))

            batch_data = np.array(batch_images, dtype=np.float32)
            calibration_data.append(batch_data)

        return calibration_data

    def _generate_synthetic_data(self) -> list[np.ndarray]:
        """Generate synthetic calibration data when real images are unavailable.

        Creates random data with ImageNet-like distribution for calibration.

        Returns:
            List of numpy arrays containing synthetic calibration batches.
        """
        calibration_data = []
        _, channels, height, width = self.input_shape

        # Generate 10 batches of synthetic data
        for _ in range(10):
            # Generate random data with ImageNet-like distribution
            batch_data = np.random.normal(
                0.0, 1.0, (self.batch_size, channels, height, width)
            ).astype(np.float32)
            # Apply ImageNet normalization in reverse to get realistic input ranges
            mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
            std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
            batch_data = (batch_data * std) + mean
            batch_data = np.clip(batch_data, 0.0, 1.0)
            calibration_data.append(batch_data)

        if self.logger:
            self.logger.info(f"Generated {len(calibration_data)} synthetic calibration batches")

        return calibration_data

    def get_batch_size(self) -> int:
        """Return the batch size for calibration.

        Returns:
            The calibration batch size.
        """
        return self.batch_size

    def get_batch(self, names: list[str]) -> Optional[list[int]]:
        """Get the next batch of calibration data.

        Args:
            names: List of input tensor names (unused but required by interface).

        Returns:
            List of device pointers for input tensors, or None if no more data.
        """
        if self.current_index >= len(self.calibration_data):
            return None

        try:
            # Get current batch
            batch_data = self.calibration_data[self.current_index]
            self.current_index += 1

            # Ensure data is contiguous and in the right format
            batch_data = np.ascontiguousarray(batch_data, dtype=np.float32)

            # Verify data size matches expected input size
            expected_size = (
                self.batch_size * self.input_shape[1] * self.input_shape[2] * self.input_shape[3]
            )
            if batch_data.size != expected_size:
                if self.logger:
                    self.logger.warning(
                        f"Batch data size mismatch: {batch_data.size} vs expected {expected_size}"
                    )
                # Reshape or pad as needed
                batch_data = batch_data.flatten()[:expected_size]
                batch_data = batch_data.reshape(self.input_shape)
                batch_data = np.ascontiguousarray(batch_data, dtype=np.float32)

            # Copy batch to device
            cuda.memcpy_htod(self.device_input, batch_data)

            return [int(self.device_input)]

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error in get_batch: {e}")
            return None

    def read_calibration_cache(self) -> Optional[bytes]:
        """Read calibration cache from file.

        Returns:
            Cache data as bytes if file exists, None otherwise.
        """
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                if self.logger:
                    self.logger.info(f"Reading calibration cache from {self.cache_file}")
                return f.read()
        return None

    def write_calibration_cache(self, cache: bytes) -> None:
        """Write calibration cache to file.

        Args:
            cache: Calibration cache data to save.
        """
        with open(self.cache_file, "wb") as f:
            f.write(cache)
            if self.logger:
                self.logger.info(f"Calibration cache saved to {self.cache_file}")


def convert_onnx_to_tensorrt(
    onnx_path: str,
    engine_path: str,
    logger: logging.Logger,
    fp16: bool = False,
    int8: bool = False,
    workspace_size: int = 1024,
    input_shape: Optional[tuple[int, int, int, int]] = None,
    max_batch_size: Optional[int] = None,
    calibration_data_path: Optional[str] = None,
    calibration_cache_path: Optional[str] = None,
) -> bool:
    """Convert an ONNX model to a TensorRT engine.

    Uses TensorRT Python API to parse the ONNX model and build an optimized
    inference engine with optional FP16 or INT8 precision modes.

    Args:
        onnx_path: Path to the input ONNX model file.
        engine_path: Path to save the output TensorRT engine.
        logger: Logger instance for status messages.
        fp16: Enable FP16 (half-precision) mode for faster inference.
        int8: Enable INT8 quantization for maximum performance.
        workspace_size: TensorRT workspace memory limit in megabytes.
        input_shape: Input tensor shape as (batch, channels, height, width).
        max_batch_size: Maximum batch size for dynamic batch optimization.
        calibration_data_path: Path to calibration images for INT8 mode.
        calibration_cache_path: Path to save/load INT8 calibration cache.

    Returns:
        True if conversion succeeded, False otherwise.
    """

    # Create TensorRT logger
    trt_logger = trt.Logger(trt.Logger.INFO)

    # Create builder and network
    builder = trt.Builder(trt_logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, trt_logger)

    logger.info("Parsing ONNX model...")

    # Parse ONNX file
    # If the ONNX uses external data (e.g., model.onnx.data), TensorRT expects
    # to resolve it relative to the current working directory. Temporarily
    # switch to the model directory to allow the parser to find external data.
    model_dir = os.path.dirname(os.path.abspath(onnx_path)) or "."
    model_name = os.path.basename(onnx_path)
    prev_cwd = os.getcwd()
    try:
        os.chdir(model_dir)
        with open(model_name, "rb") as model:
            if not parser.parse(model.read()):
                logger.error("Failed to parse ONNX model")
                for i in range(parser.num_errors):
                    logger.error(f"Parser error {i}: {parser.get_error(i)}")
                return False
    finally:
        os.chdir(prev_cwd)

    logger.info("ONNX model parsed successfully")

    # Configure builder
    config = builder.create_builder_config()

    # Set workspace size (convert MB to bytes)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size * 1024 * 1024)
    logger.info(f"Workspace size set to: {workspace_size} MB")

    # Set precision options
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        logger.info("FP16 precision enabled")

    if int8:
        config.set_flag(trt.BuilderFlag.INT8)
        logger.info("INT8 precision enabled")

        # Set up INT8 calibrator
        if calibration_cache_path is None:
            calibration_cache_path = f"{os.path.splitext(engine_path)[0]}_calibration.cache"

        calibrator = ImageNetCalibrator(
            calibration_data_path=calibration_data_path or "./calibration_data",
            cache_file=calibration_cache_path,
            batch_size=input_shape[0] if input_shape else 1,
            input_shape=input_shape or (1, 3, 224, 224),
            logger=logger,
        )
        config.int8_calibrator = calibrator
        logger.info(f"INT8 calibrator configured with cache: {calibration_cache_path}")

    # Handle dynamic batch size optimization
    if max_batch_size and max_batch_size > 1:
        logger.info("Setting up dynamic batch size optimization...")

        # Create optimization profile
        profile = builder.create_optimization_profile()

        # Get input tensor info
        input_tensor = network.get_input(0)
        input_name = input_tensor.name

        if input_shape:
            _, channels, height, width = input_shape
        else:
            # Fallback to network input shape
            shape = input_tensor.shape
            channels, height, width = shape[1], shape[2], shape[3]

        min_batch = 1
        opt_batch = max_batch_size // 2 if max_batch_size > 2 else 1
        max_batch = max_batch_size

        # Set shape ranges for the input
        min_shape = (min_batch, channels, height, width)
        opt_shape = (opt_batch, channels, height, width)
        max_shape = (max_batch, channels, height, width)

        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)

        logger.info(f"Dynamic batch sizes - Min: {min_batch}, Opt: {opt_batch}, Max: {max_batch}")
        logger.info(f"Shape ranges - Min: {min_shape}, Opt: {opt_shape}, Max: {max_shape}")

    logger.info("Building TensorRT engine...")
    logger.info("This may take several minutes...")

    try:
        # Build engine
        serialized_engine = builder.build_serialized_network(network, config)

        if serialized_engine is None:
            logger.error("Failed to build TensorRT engine")
            return False

        logger.info("TensorRT engine built successfully")

        # Save engine to file
        with open(engine_path, "wb") as f:
            f.write(serialized_engine)

        logger.info("TensorRT conversion completed successfully")

        # Report engine file size
        try:
            engine_size = os.path.getsize(engine_path) / (1024 * 1024)
            logger.info(f"Engine saved - Size: {engine_size:.2f} MB")
        except Exception as e:
            logger.warning(f"Could not get engine size: {e}")

        return True

    except Exception as e:
        logger.error(f"Error building TensorRT engine: {str(e)}")
        return False


def check_tensorrt_available() -> bool:
    """Check if TensorRT Python library is available and functional.

    Returns:
        True if TensorRT can be imported and a builder created, False otherwise.
    """
    try:
        import tensorrt as trt

        logger = trt.Logger(trt.Logger.ERROR)
        builder = trt.Builder(logger)
        return builder is not None
    except ImportError:
        return False
    except Exception:
        return False


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for ONNX to TensorRT conversion.

    Returns:
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Convert ONNX to TensorRT using trtexec command-line tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument("--onnx_path", type=str, required=True, help="Path to ONNX model")
    parser.add_argument(
        "--engine_path", type=str, required=True, help="Path to save TensorRT engine"
    )

    # Model configuration
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["pose", "detector"],
        default="pose",
        help="Model family: 'pose' (ViT pose) or 'detector' (LW-DETR)",
    )
    parser.add_argument(
        "--detector_variant",
        type=str,
        choices=list(LWDETR_IMG_SIZES.keys()),
        default="tiny",
        help="LW-DETR variant input size when --model_type=detector",
    )
    parser.add_argument(
        "--vit_model",
        type=str,
        choices=SUPPORTED_VIT_MODELS,
        help="ViT model to use (auto-determines input shape)",
    )
    parser.add_argument(
        "--input_shape",
        type=str,
        default=None,
        help="Input shape (HxW) - overrides ViT model default",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=None,
        help="Maximum batch size for dynamic optimization",
    )

    # Precision options
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 precision")
    parser.add_argument("--int8", action="store_true", help="Enable INT8 precision")

    # INT8 calibration options
    parser.add_argument(
        "--calibration_data",
        type=str,
        default=None,
        help="Path to calibration images directory for INT8 quantization (optional - uses synthetic data if not provided)",
    )
    parser.add_argument(
        "--calibration_cache",
        type=str,
        default=None,
        help="Path to calibration cache file (auto-generated if not specified)",
    )

    # Engine configuration
    parser.add_argument(
        "--workspace_size", type=int, default=1024, help="TensorRT workspace size in MB"
    )

    # Logging
    parser.add_argument("--log_dir", type=str, default="./runs", help="Directory to store logs")

    return parser.parse_args()


def main() -> bool:
    """Run the ONNX to TensorRT conversion pipeline.

    Returns:
        True if conversion succeeded, False otherwise.
    """
    args = parse_args()

    # Set up logging
    logger, _ = setup_logging(args=args, base_filename="convert_onnx_to_trt")

    # Check if TensorRT Python library is available
    if not check_tensorrt_available():
        logger.error("TensorRT Python library not found or not working")
        logger.error(
            "Please ensure TensorRT Python library is properly installed with: pip install tensorrt"
        )
        return False

    # Create log directory if needed
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
        logger.info(f"Created log directory: {args.log_dir}")

    # Validate ONNX file exists
    if not os.path.exists(args.onnx_path):
        logger.error(f"ONNX file not found: {args.onnx_path}")
        return False

    # Create output directory if needed
    output_dir = os.path.dirname(args.engine_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")

    try:
        # Determine input shape
        if args.input_shape is not None:
            # Manual override
            h, w = map(int, args.input_shape.split(","))
            logger.info(f"Using manual input shape: {h}x{w}")
        elif args.model_type == "pose" and args.vit_model is not None:
            # Auto-determine from ViT model
            if args.vit_model in VIT_MODELS:
                h, w = VIT_MODELS[args.vit_model][3]
                logger.info(f"Using ViT model {args.vit_model} input shape: {h}x{w}")
            else:
                logger.warning(f"Unknown ViT model {args.vit_model}, using default 224x224")
                h, w = 224, 224
        elif args.model_type == "detector":
            if args.detector_variant in LWDETR_IMG_SIZES:
                h = w = LWDETR_IMG_SIZES[args.detector_variant]
                logger.info(f"Using LW-DETR {args.detector_variant} input shape: {h}x{w}")
                logger.warning(
                    "LW-DETR ONNX often contains GridSample; TensorRT requires a plugin. "
                    "Conversion may fail without custom plugins."
                )
            else:
                logger.warning(
                    f"Unknown LW-DETR variant {args.detector_variant}, using default 640x640"
                )
                h = w = 640
        else:
            # Default fallback
            h, w = 224, 224
            logger.info("Using default input shape: 224x224")

        input_shape = (args.batch_size, 3, h, w)

        # Print configuration summary
        logger.info("")
        logger.info("=" * 60)
        logger.info("TENSORRT CONVERSION CONFIGURATION")
        logger.info("=" * 60)
        logger.info(f"Input ONNX: {args.onnx_path}")
        logger.info(f"Output Engine: {args.engine_path}")
        logger.info(f"Input Shape: {input_shape}")
        logger.info(f"Model Type: {args.model_type}")
        if args.model_type == "detector":
            logger.info(f"Detector Variant: {args.detector_variant}")
        logger.info(f"FP16: {args.fp16}")
        logger.info(f"INT8: {args.int8}")
        logger.info(f"Workspace: {args.workspace_size} MB")
        if args.vit_model:
            logger.info(f"ViT Model: {args.vit_model}")
        if args.max_batch_size:
            logger.info(f"Max Batch Size: {args.max_batch_size}")
        logger.info("=" * 60)
        logger.info("")

        # Convert using TensorRT Python API
        success = convert_onnx_to_tensorrt(
            onnx_path=args.onnx_path,
            engine_path=args.engine_path,
            logger=logger,
            fp16=args.fp16,
            int8=args.int8,
            workspace_size=args.workspace_size,
            input_shape=input_shape,
            max_batch_size=args.max_batch_size,
            calibration_data_path=args.calibration_data,
            calibration_cache_path=args.calibration_cache,
        )

        if not success:
            logger.error("TensorRT conversion failed")
            return False

        # Success summary
        logger.info("")
        logger.info("=" * 60)
        logger.info("TENSORRT CONVERSION COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Engine saved to: {args.engine_path}")
        logger.info("=" * 60)

        return True

    except Exception as e:
        logger.error(f"Error during TensorRT conversion: {str(e)}")
        logger.error("Common solutions:")
        logger.error("  - Ensure TensorRT Python library is properly installed")
        logger.error("  - Try with --fp16 for better compatibility")
        logger.error("  - Increase --workspace_size if memory issues")
        logger.error("  - Check ONNX model compatibility with TensorRT")
        logger.error("  - Verify CUDA and cuDNN are properly installed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
