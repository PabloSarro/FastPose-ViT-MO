"""TensorRT model performance evaluation on SPEED dataset."""

import argparse
import ctypes
import json
import os
import time
from typing import Optional

import cv2
import numpy as np
import tensorrt as trt
import torch
from cuda import cudart
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from object_detector.constants import IMAGENET_MEAN, IMAGENET_STD
from object_detector.metrics import evaluate_predictions_single_class
from object_detector.utils import box_cxcywh_to_xyxy, box_iou
from src.datasets import SPEEDDataset, get_dataset_config
from src.metrics import compute_metrics
from src.utils import (
    bbox_relative_translation_to_translation,
    get_absolute_orientation,
    process_rotation,
    rotation_matrix_to_quaternion,
    setup_logging,
)


class TensorRTInference:
    """TensorRT inference engine wrapper with CUDA memory management.

    Provides a context manager interface for running inference with TensorRT,
    handling buffer allocation, memory transfers, and cleanup.
    """

    def __init__(self, engine_path: str) -> None:
        """Initialize TensorRT inference engine.

        Args:
            engine_path: Path to the serialized TensorRT engine file.

        Raises:
            RuntimeError: If CUDA device initialization fails.
        """
        self.logger = trt.Logger(trt.Logger.WARNING)  # Reduce verbosity
        self.engine_path = engine_path
        self.engine: Optional[trt.ICudaEngine] = None
        self.context: Optional[trt.IExecutionContext] = None
        self.bindings: dict[str, dict] = {}
        self.stream = None
        self.device_id = 0
        # Cache frequently accessed bindings for faster lookup
        self.input_binding: Optional[dict] = None
        self.output_bindings: dict[str, dict] = {}

        # Set CUDA device
        err = cudart.cudaSetDevice(self.device_id)[0]
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaSetDevice failed: {err}")

    def load_engine(self) -> "TensorRTInference":
        """Load and initialize the TensorRT engine.

        Deserializes the engine, creates execution context, allocates buffers,
        and performs warmup inference runs.

        Returns:
            Self for method chaining.

        Raises:
            RuntimeError: If engine loading, context creation, or buffer allocation fails.
        """
        print("Loading engine...")
        with open(self.engine_path, "rb") as f:
            engine_bytes = f.read()

        print("Deserializing engine...")
        runtime = trt.Runtime(self.logger)
        self.engine = runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(
                "Failed to deserialize TensorRT engine. Check TensorRT version compatibility."
            )

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create execution context")

        # Create CUDA stream
        err, stream = cudart.cudaStreamCreate()
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaStreamCreate failed: {err}")
        self.stream = stream

        # Create CUDA events for precise timing
        err, self.start_event = cudart.cudaEventCreate()
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaEventCreate failed for start event: {err}")

        err, self.end_event = cudart.cudaEventCreate()
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaEventCreate failed for end event: {err}")

        # Allocate buffers
        print("Allocating buffers...")
        self._allocate_buffers()

        # Warm up the inference engine for consistent timing
        print("Warming up inference engine...")
        self._warm_up()
        return self

    def _allocate_buffers(self) -> None:
        """Allocate host and device memory buffers for all engine I/O tensors."""
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            mode = self.engine.get_tensor_mode(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            size = int(trt.volume(shape))
            nbytes = size * np.dtype(dtype).itemsize

            # Allocate pinned host memory
            err, host_ptr = cudart.cudaHostAlloc(nbytes, cudart.cudaHostAllocDefault)
            if err != cudart.cudaError_t.cudaSuccess:
                raise RuntimeError(f"cudaHostAlloc failed for {name}: {err}")

            # Create numpy array view of pinned memory
            buf_type = ctypes.c_char * nbytes
            host_buf = (buf_type).from_address(host_ptr)
            host_array = np.frombuffer(host_buf, dtype=dtype, count=size).reshape(shape)

            # Allocate device memory
            err, dev_ptr = cudart.cudaMalloc(nbytes)
            if err != cudart.cudaError_t.cudaSuccess:
                raise RuntimeError(f"cudaMalloc failed for {name}: {err}")

            self.bindings[name] = {
                "host_ptr": host_ptr,
                "host": host_array,
                "device": dev_ptr,
                "shape": shape,
                "dtype": dtype,
                "mode": mode,
                "nbytes": nbytes,
            }

            # Cache binding references for faster lookup
            if mode == trt.TensorIOMode.INPUT:
                self.input_binding = self.bindings[name]
            else:  # OUTPUT
                self.output_bindings[name] = self.bindings[name]

    def _warm_up(self, num_warmup_runs: int = 5) -> None:
        """Warm up the inference engine to eliminate cold-start overhead.

        Args:
            num_warmup_runs: Number of warmup inference iterations.
        """
        if self.input_binding is None:
            return

        # Create dummy input data with the right shape
        dummy_input = np.random.randn(*self.input_binding["shape"]).astype(
            self.input_binding["dtype"]
        )

        for _ in range(num_warmup_runs):
            # Run inference without timing to warm up GPU kernels and caches
            self.inference(dummy_input, get_timing=False)

        # Synchronize to ensure all warmup operations are complete
        err = cudart.cudaStreamSynchronize(self.stream)[0]
        if err != cudart.cudaError_t.cudaSuccess:
            print(f"Warning: cudaStreamSynchronize failed during warmup: {err}")

    def inference(
        self, input_data: np.ndarray, get_timing: bool = False
    ) -> dict[str, np.ndarray] | tuple[dict[str, np.ndarray], float]:
        """Run inference on the input data.

        Args:
            input_data: Input tensor as a numpy array matching the engine's input shape.
            get_timing: Whether to measure and return inference time.

        Returns:
            If get_timing is False, returns a dictionary mapping output names
            to numpy arrays. If get_timing is True, returns a tuple of
            (output_dict, inference_time_seconds).

        Raises:
            RuntimeError: If no input binding is found or input size mismatches.
        """
        # Use cached input binding for faster access
        if self.input_binding is None:
            raise RuntimeError("No input binding found")

        # Ensure input data matches expected shape
        input_data_flat = input_data.ravel()
        expected_size = self.input_binding["host"].size
        if len(input_data_flat) != expected_size:
            raise RuntimeError(
                f"Input size mismatch: got {len(input_data_flat)}, expected {expected_size}"
            )

        # Copy input to host buffer (in-place operation, reusing buffer)
        self.input_binding["host"].flat[:] = input_data_flat

        if get_timing:
            # Record start event on the stream for precise GPU timing
            err = cudart.cudaEventRecord(self.start_event, self.stream)[0]
            if err != cudart.cudaError_t.cudaSuccess:
                raise RuntimeError(f"cudaEventRecord failed for start event: {err}")

        # Execute inference
        self._execute()

        if get_timing:
            # Record end event and calculate elapsed time
            err = cudart.cudaEventRecord(self.end_event, self.stream)[0]
            if err != cudart.cudaError_t.cudaSuccess:
                raise RuntimeError(f"cudaEventRecord failed for end event: {err}")

            # Synchronize to ensure timing is accurate
            err = cudart.cudaEventSynchronize(self.end_event)[0]
            if err != cudart.cudaError_t.cudaSuccess:
                raise RuntimeError(f"cudaEventSynchronize failed: {err}")

            # Get elapsed time in milliseconds and convert to seconds
            err, elapsed_ms = cudart.cudaEventElapsedTime(self.start_event, self.end_event)
            if err != cudart.cudaError_t.cudaSuccess:
                raise RuntimeError(f"cudaEventElapsedTime failed: {err}")
            infer_time = elapsed_ms / 1000.0  # Convert to seconds

        # Get outputs using cached output bindings for faster access
        outputs = {}
        for name, binding in self.output_bindings.items():
            # Return a copy only when necessary - the caller may need to store this
            outputs[name] = binding["host"].copy()

        return (outputs, infer_time) if get_timing else outputs

    def _execute(self) -> None:
        """Execute inference: copy inputs to GPU, run engine, copy outputs back."""
        # H2D: Copy input from host to device (use cached input binding)
        if self.input_binding:
            err = cudart.cudaMemcpyAsync(
                self.input_binding["device"],
                self.input_binding["host"].ctypes.data,
                self.input_binding["nbytes"],
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                self.stream,
            )[0]
            if err != cudart.cudaError_t.cudaSuccess:
                raise RuntimeError(f"cudaMemcpyAsync H2D failed: {err}")

        # Set tensor addresses (use all bindings)
        for name, binding in self.bindings.items():
            self.context.set_tensor_address(name, binding["device"])

        # Execute inference
        if not self.context.execute_async_v3(self.stream):
            raise RuntimeError("TensorRT inference execution failed")

        # D2H: Copy outputs from device to host (use cached output bindings)
        for name, binding in self.output_bindings.items():
            err = cudart.cudaMemcpyAsync(
                binding["host"].ctypes.data,
                binding["device"],
                binding["nbytes"],
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                self.stream,
            )[0]
            if err != cudart.cudaError_t.cudaSuccess:
                raise RuntimeError(f"cudaMemcpyAsync D2H failed for {name}: {err}")

        # Synchronize stream
        err = cudart.cudaStreamSynchronize(self.stream)[0]
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaStreamSynchronize failed: {err}")

    def cleanup(self) -> None:
        """Clean up CUDA resources including streams, events, and memory."""
        try:
            # Synchronize and destroy stream
            if self.stream is not None:
                cudart.cudaStreamSynchronize(self.stream)
                cudart.cudaStreamDestroy(self.stream)
                self.stream = None

            # Destroy CUDA events
            if hasattr(self, "start_event") and self.start_event is not None:
                cudart.cudaEventDestroy(self.start_event)
                self.start_event = None
            if hasattr(self, "end_event") and self.end_event is not None:
                cudart.cudaEventDestroy(self.end_event)
                self.end_event = None

            # Free device and host memory
            for binding in self.bindings.values():
                if binding.get("device"):
                    cudart.cudaFree(binding["device"])
                if binding.get("host_ptr"):
                    cudart.cudaFreeHost(binding["host_ptr"])

            self.bindings.clear()

            # Clear cached binding references
            self.input_binding = None
            self.output_bindings.clear()

            # Clean up TensorRT objects
            if self.context:
                del self.context
                self.context = None
            if self.engine:
                del self.engine
                self.engine = None

        except Exception as e:
            print(f"Warning: Error during cleanup: {e}")

    def __enter__(self) -> "TensorRTInference":
        return self

    def __exit__(self, *exc: object) -> None:
        self.cleanup()

    def __del__(self) -> None:
        self.cleanup()


def _prepare_image_for_trt(
    image: np.ndarray,
    output_hw: tuple[int, int],
    mean: np.ndarray,
    std: np.ndarray,
    inv_std: np.ndarray,
    *,
    resized_buffer: np.ndarray | None = None,
    chw_buffer: np.ndarray | None = None,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Resize and normalize an image for TensorRT inference.

    Args:
        image: Input image as HWC float32 array with values in [0, 1].
        output_hw: Target (height, width) for resizing.
        mean: Per-channel mean for normalization (shape 1x1x3).
        std: Per-channel standard deviation (shape 1x1x3).
        inv_std: Precomputed 1/std for faster normalization.
        resized_buffer: Optional pre-allocated buffer for resized image.
        chw_buffer: Optional pre-allocated buffer for CHW output.
        interpolation: OpenCV interpolation method.

    Returns:
        Normalized CHW float32 array ready for TensorRT inference.
    """

    height, width = output_hw
    needs_resize = image.shape[0] != height or image.shape[1] != width

    if needs_resize:
        if resized_buffer is not None and resized_buffer.shape == (height, width, 3):
            resized = cv2.resize(
                image,
                (width, height),
                dst=resized_buffer,
                interpolation=interpolation,
            )
        else:
            resized = cv2.resize(image, (width, height), interpolation=interpolation)
    else:
        if resized_buffer is not None and resized_buffer.shape == (height, width, 3):
            np.copyto(resized_buffer, image)
            resized = resized_buffer
        else:
            resized = image.copy()

    resized -= mean
    resized *= inv_std

    chw_view = np.transpose(resized, (2, 0, 1))
    if chw_buffer is not None and chw_buffer.shape == (3, height, width):
        np.copyto(chw_buffer, chw_view)
        return chw_buffer
    return np.ascontiguousarray(chw_view)


def _select_output(
    outputs: dict[str, np.ndarray], name_options: list[str]
) -> Optional[np.ndarray]:
    """Select the first available output from a list of possible names.

    Args:
        outputs: Dictionary of output name to numpy array.
        name_options: List of possible output names to search for.

    Returns:
        The first matching output array, or None if no match found.
    """
    for name in name_options:
        if name in outputs:
            return outputs[name]
    return None


def _clamp_bbox_xyxy(bbox: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Clamp a bounding box to image bounds while ensuring positive area.

    Args:
        bbox: Bounding box tensor in xyxy format (x1, y1, x2, y2).
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        Clamped bounding box tensor with guaranteed positive area.
    """
    clamped = bbox.clone()
    clamped[0] = clamped[0].clamp(0.0, float(max(width - 1, 0)))
    clamped[1] = clamped[1].clamp(0.0, float(max(height - 1, 0)))
    clamped[2] = clamped[2].clamp(clamped[0] + 1.0, float(max(width, 1)))
    clamped[3] = clamped[3].clamp(clamped[1] + 1.0, float(max(height, 1)))
    return clamped


def _letterbox_resize(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Resize an image with letterbox padding to preserve aspect ratio.

    Args:
        src: Source image as HWC numpy array.
        dst: Pre-allocated destination buffer to fill.

    Returns:
        The destination buffer with the resized and padded image.
    """
    dst.fill(0.0)
    h, w = src.shape[:2]
    target_h, target_w = dst.shape[:2]
    if h == 0 or w == 0:
        return dst

    scale = min(target_w / float(w), target_h / float(h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_LINEAR if scale >= 1.0 else cv2.INTER_AREA
    resized = cv2.resize(src, (new_w, new_h), interpolation=interp)

    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    dst[y_offset : y_offset + new_h, x_offset : x_offset + new_w, :] = resized
    return dst


def evaluate_pose(args: argparse.Namespace) -> dict:
    """Evaluate a TensorRT pose estimation model on the SPEED dataset.

    Runs inference on the test split and computes pose estimation metrics
    including translation error, rotation error, and SPEED score.

    Args:
        args: Command line arguments containing engine path, dataset configuration,
            and evaluation parameters.

    Returns:
        Dictionary containing metrics, predictions, and timing statistics.
    """
    logger, _ = setup_logging(args=args, base_filename="evaluate_trt_latency")
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

    results_summary = {}
    try:
        # Initialize TensorRT model
        engine_path = getattr(args, "trt_engine_pose_path", None) or args.trt_engine_path
        trt_model = TensorRTInference(engine_path)
        trt_model.load_engine()
        logger.info(f"Loaded TensorRT engine from {engine_path}")

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
        postproc_times = [] if getattr(args, "measure_postproc", False) else None

        # Cache frequently used dataset properties for optimization
        camera_params = test_dataset.camera if hasattr(test_dataset, "camera") else None

        # Pre-allocate GPU tensors to avoid repeated .cuda() calls
        device = torch.device("cuda")
        # These will be resized as needed during the loop
        pred_translations_gpu = torch.empty(
            (args.batch_size, 3), device=device, dtype=torch.float32
        )
        # For predictions, we handle 6D representation initially, then convert to quaternions
        pred_rotations_gpu = torch.empty(
            (args.batch_size, 6), device=device, dtype=torch.float32
        )  # 6D representation
        translations_gpu = torch.empty((args.batch_size, 3), device=device, dtype=torch.float32)
        # For rotations, we need to handle both quaternion (4,) and matrix (3, 3) formats
        if args.rotation_format == "matrix":
            rotations_gpu = torch.empty(
                (args.batch_size, 3, 3), device=device, dtype=torch.float32
            )
        else:
            rotations_gpu = torch.empty((args.batch_size, 4), device=device, dtype=torch.float32)
        bbox_gpu = torch.empty((args.batch_size, 4), device=device, dtype=torch.float32)

        # Evaluation loop
        sample_idx = 0  # Counter for unique sample numbering
        with torch.no_grad():
            for images, translations, rotations, bbox in tqdm(test_loader, desc="Evaluating"):
                try:
                    # Convert images to numpy and prepare for TensorRT
                    images_np = images.numpy()

                    # Run inference with timing
                    outputs, batch_time = trt_model.inference(images_np, get_timing=True)
                    inference_times.append(batch_time)

                    # Move ground truth to GPU using pre-allocated tensors
                    batch_size = translations.shape[0]
                    if translations_gpu.shape[0] != batch_size:
                        # Resize tensors if batch size changed
                        translations_gpu = (
                            translations_gpu[:batch_size]
                            if batch_size <= translations_gpu.shape[0]
                            else torch.empty((batch_size, 3), device=device, dtype=torch.float32)
                        )
                        if args.rotation_format == "matrix":
                            rotations_gpu = (
                                rotations_gpu[:batch_size]
                                if batch_size <= rotations_gpu.shape[0]
                                else torch.empty(
                                    (batch_size, 3, 3), device=device, dtype=torch.float32
                                )
                            )
                        else:
                            rotations_gpu = (
                                rotations_gpu[:batch_size]
                                if batch_size <= rotations_gpu.shape[0]
                                else torch.empty(
                                    (batch_size, rotations.shape[1]),
                                    device=device,
                                    dtype=torch.float32,
                                )
                            )

                    translations_gpu[:batch_size].copy_(translations)
                    if args.rotation_format == "matrix":
                        rotations_gpu[:batch_size].copy_(rotations)
                    else:
                        rotations_gpu[:batch_size, : rotations.shape[1]].copy_(rotations)

                    # Process TensorRT outputs using pre-allocated tensors
                    if args.merge_outputs:
                        output = torch.from_numpy(outputs["output"])
                        # Resize pred tensors if needed
                        if pred_translations_gpu.shape[0] != batch_size:
                            pred_translations_gpu = (
                                pred_translations_gpu[:batch_size]
                                if batch_size <= pred_translations_gpu.shape[0]
                                else torch.empty(
                                    (batch_size, 3), device=device, dtype=torch.float32
                                )
                            )
                            pred_rotations_gpu = (
                                pred_rotations_gpu[:batch_size]
                                if batch_size <= pred_rotations_gpu.shape[0]
                                else torch.empty(
                                    (batch_size, output.shape[1] - 3),
                                    device=device,
                                    dtype=torch.float32,
                                )
                            )

                        pred_translations_gpu[:batch_size].copy_(output[:, :3])
                        pred_rotations_gpu[:batch_size, : output.shape[1] - 3].copy_(output[:, 3:])
                        pred_translations = pred_translations_gpu[:batch_size]
                        pred_rotations = pred_rotations_gpu[:batch_size, : output.shape[1] - 3]
                    else:
                        # Resize pred tensors if needed
                        if pred_translations_gpu.shape[0] != batch_size:
                            pred_translations_gpu = (
                                pred_translations_gpu[:batch_size]
                                if batch_size <= pred_translations_gpu.shape[0]
                                else torch.empty(
                                    (batch_size, 3), device=device, dtype=torch.float32
                                )
                            )
                            pred_rotations_gpu = (
                                pred_rotations_gpu[:batch_size]
                                if batch_size <= pred_rotations_gpu.shape[0]
                                else torch.empty(
                                    (batch_size, outputs["rotation"].shape[1]),
                                    device=device,
                                    dtype=torch.float32,
                                )
                            )

                        pred_translations_gpu[:batch_size].copy_(
                            torch.from_numpy(outputs["translation"])
                        )
                        pred_rotations_gpu[:batch_size, : outputs["rotation"].shape[1]].copy_(
                            torch.from_numpy(outputs["rotation"])
                        )
                        pred_translations = pred_translations_gpu[:batch_size]
                        pred_rotations = pred_rotations_gpu[
                            :batch_size, : outputs["rotation"].shape[1]
                        ]

                    # Use references to avoid confusion
                    translations = translations_gpu[:batch_size]
                    if args.rotation_format == "matrix":
                        rotations = rotations_gpu[:batch_size]
                    else:
                        rotations = rotations_gpu[:batch_size, : rotations.shape[1]]

                    # Process rotations based on format (match PyTorch evaluation exactly)
                    post_start = time.perf_counter() if postproc_times is not None else None
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
                        # Use pre-allocated bbox tensor
                        if bbox_gpu.shape[0] != batch_size:
                            bbox_gpu = (
                                bbox_gpu[:batch_size]
                                if batch_size <= bbox_gpu.shape[0]
                                else torch.empty(
                                    (batch_size, bbox.shape[1]), device=device, dtype=torch.float32
                                )
                            )
                        bbox_gpu[:batch_size, : bbox.shape[1]].copy_(bbox)
                        bbox = bbox_gpu[:batch_size, : bbox.shape[1]]
                        pred_translations = bbox_relative_translation_to_translation(
                            pred_translations,
                            bbox,
                            camera_params,
                        )
                        translations = bbox_relative_translation_to_translation(
                            translations,
                            bbox,
                            camera_params,
                        )

                    # Convert rotation to centered rotation (both should be quaternions now)
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
                    if postproc_times is not None and post_start is not None:
                        postproc_times.append(time.perf_counter() - post_start)

                    # Accumulate metrics (batch append for better performance)
                    metric_values = [
                        metrics["translation_metric_mean"],
                        metrics["rotation_metric_mean"],
                        metrics["total_metric"],
                        metrics["relative_translation_metric_mean"],
                        metrics["relative_total_metric"],
                    ]
                    metric_keys = [
                        "translation_metric",
                        "rotation_metric",
                        "total_metric",
                        "relative_translation_metric",
                        "relative_total_metric",
                    ]
                    for key, value in zip(metric_keys, metric_values):
                        total_metrics[key].append(value)

                    # Store predictions and ground truth (optimized batch processing)
                    pred_trans_cpu = pred_translations.cpu()
                    pred_rot_cpu = pred_rotations.cpu()
                    trans_cpu = translations.cpu()
                    rot_cpu = rotations.cpu()

                    for i in range(len(translations)):
                        results[str(sample_idx)] = {
                            "prediction": {
                                "translation": pred_trans_cpu[i].tolist(),
                                "rotation": pred_rot_cpu[i].tolist(),
                            },
                            "ground_truth": {
                                "translation": trans_cpu[i].tolist(),
                                "rotation": rot_cpu[i].tolist(),
                            },
                        }
                        sample_idx += 1

                except Exception as e:
                    logger.error(f"Error processing batch : {e}")
                    continue

            # Compute average metrics
            avg_metrics = {k: sum(v) / len(v) for k, v in total_metrics.items()}

            # Calculate timing statistics
            avg_inference_time = np.mean(inference_times) * 1000  # Convert to ms
            std_inference_time = np.std(inference_times) * 1000

            # Add metrics and timing to results
            results["metrics"] = {
                "translation_metric": float(avg_metrics["translation_metric"]),
                "rotation_metric": float(avg_metrics["rotation_metric"]),
                "total_metric": float(avg_metrics["total_metric"]),
                "relative_translation_metric": float(avg_metrics["relative_translation_metric"]),
                "relative_total_metric": float(avg_metrics["relative_total_metric"]),
                "average_inference_time_ms": float(avg_inference_time),
                "std_inference_time_ms": float(std_inference_time),
                "fps": float(1000 / avg_inference_time),
            }
            if postproc_times is not None and len(postproc_times) > 0:
                avg_post_ms = float(np.mean(postproc_times) * 1000)
                std_post_ms = float(np.std(postproc_times) * 1000)
                results["metrics"]["average_postprocessing_time_ms"] = avg_post_ms
                results["metrics"]["std_postprocessing_time_ms"] = std_post_ms

            # Print results
            logger.info("\nEvaluation Results:")
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
            logger.info("\nInference Performance:")
            logger.info(f"Average inference time: {avg_inference_time:.2f} ms")
            logger.info(f"Inference time std: {std_inference_time:.2f} ms")
            logger.info(f"Throughput: {1000 / avg_inference_time:.2f} FPS")
            if postproc_times is not None and len(postproc_times) > 0:
                logger.info("Postprocessing:")
                logger.info(f"Average postprocessing time: {avg_post_ms:.2f} ms")
                logger.info(f"Postprocessing time std: {std_post_ms:.2f} ms")

            results_summary = results

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise
    finally:
        # Clean up TensorRT resources
        if "trt_model" in locals():
            trt_model.cleanup()

    return results_summary


def evaluate_pipeline(args: argparse.Namespace) -> dict:
    """Evaluate end-to-end pipeline with detector and pose TensorRT engines.

    Measures latency across the full pipeline: preprocessing, detection,
    crop extraction, pose estimation, and postprocessing.

    Args:
        args: Command line arguments containing detector and pose engine paths,
            dataset configuration, and evaluation parameters.

    Returns:
        Dictionary containing per-stage timing statistics, metrics, and predictions.

    Raises:
        ValueError: If required engine paths are not provided.
    """
    logger, _ = setup_logging(args=args, base_filename="evaluate_trt_latency_pipeline")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    if not args.trt_engine_detector_path:
        raise ValueError("Detector engine path (--trt_engine_detector_path) is required")
    if not args.trt_engine_pose_path and not args.trt_engine_path:
        raise ValueError(
            "Pose engine path (--trt_engine_pose_path or --trt_engine_path) is required"
        )

    det_engine_path = args.trt_engine_detector_path
    pose_engine_path = getattr(args, "trt_engine_pose_path", None) or args.trt_engine_path

    logger.info("Intermediate processing on: cpu")

    # Load engines
    det_trt = TensorRTInference(det_engine_path)
    pose_trt = TensorRTInference(pose_engine_path)
    det_trt.load_engine()
    pose_trt.load_engine()

    # Determine input resolutions
    det_shape = det_trt.input_binding["shape"]
    pose_shape = pose_trt.input_binding["shape"]
    if len(det_shape) != 4 or len(pose_shape) != 4:
        raise RuntimeError("Expected NCHW tensors for both detector and pose engines")

    det_height, det_width = det_shape[2], det_shape[3]
    pose_height, pose_width = pose_shape[2], pose_shape[3]
    logger.info(
        "Detector input shape: %s | Pose input shape: %s",
        det_shape,
        pose_shape,
    )

    # Precompute normalization tensors
    imagenet_mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
    imagenet_std = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)
    imagenet_mean_cv = imagenet_mean
    imagenet_std_cv = imagenet_std
    imagenet_inv_std = 1.0 / imagenet_std_cv
    imagenet_inv_std = 1.0 / imagenet_std

    det_resized_buffer = np.empty((det_height, det_width, 3), dtype=np.float32)
    det_chw_buffer = np.empty((3, det_height, det_width), dtype=np.float32)
    pose_resized_buffer = np.empty((pose_height, pose_width, 3), dtype=np.float32)
    pose_chw_buffer = np.empty((3, pose_height, pose_width), dtype=np.float32)

    # Dataset configuration & loading (reuse pose evaluation settings)
    dataset_config = get_dataset_config(args.dataset)
    bbox_json_path = os.path.join(args.dataset_root_dir, dataset_config["bbox_annotations"])

    # Determine pose image size from model registry or CLI override
    from src.models import VIT_MODELS

    pose_image_size = (
        VIT_MODELS[args.vit_model][3]
        if hasattr(args, "vit_model") and args.vit_model in VIT_MODELS
        else args.image_size
    )

    dataset_args = argparse.Namespace(**vars(args))
    dataset_args.trt_engine_path = None  # avoid accidental reuse

    pose_dataset = SPEEDDataset(
        dataset_root_dir=args.dataset_root_dir,
        split="test",
        rotation_format=args.rotation_format,
        img_size=pose_image_size,
        bbox_json_path=bbox_json_path,
        args=dataset_args,
        dataset_name=args.dataset,
    )

    camera = pose_dataset.camera
    max_samples = len(pose_dataset)
    logger.info("Evaluating pipeline on %d samples", max_samples)

    preprocess_times: list[float] = []
    detector_times: list[float] = []
    bridge_times: list[float] = []
    pose_times: list[float] = []
    post_times: list[float] = []

    metrics_accumulator = {
        "translation_metric": [],
        "rotation_metric": [],
        "total_metric": [],
        "relative_translation_metric": [],
        "relative_total_metric": [],
    }

    predictions: dict[str, dict] = {}
    fallback_bbox = 0

    try:
        for idx in tqdm(range(max_samples), desc="Pipeline", leave=False):
            filename = pose_dataset.data[idx]["filename"]
            image_path = pose_dataset._image_paths[filename]

            # Load ground truth for metrics (dataset already handles quaternion fixes)
            _, gt_translation_rel, gt_rotation_raw, gt_bbox = pose_dataset[idx]
            gt_translation_rel = gt_translation_rel.cpu()
            gt_rotation_raw = gt_rotation_raw.cpu()
            if gt_bbox is not None:
                gt_bbox = gt_bbox.cpu()

            # Load original image (float32 in [0, 1]) for detector preprocessing
            orig_image = pose_dataset._load_image_optimized(image_path)
            if orig_image.ndim != 3 or orig_image.shape[2] != 3:
                logger.warning(
                    "Image %s has unexpected shape %s; skipping", filename, orig_image.shape
                )
                continue
            pre_start = time.perf_counter()
            orig_h, orig_w = orig_image.shape[:2]
            det_input = _prepare_image_for_trt(
                orig_image,
                (det_height, det_width),
                imagenet_mean_cv,
                imagenet_std_cv,
                imagenet_inv_std,
                resized_buffer=det_resized_buffer,
                chw_buffer=det_chw_buffer,
            )
            det_batch = det_input[np.newaxis, ...]
            preprocess_times.append(time.perf_counter() - pre_start)

            # Detector inference
            det_outputs, det_time = det_trt.inference(det_batch, get_timing=True)
            detector_times.append(det_time)
            bridge_start = time.perf_counter()

            pred_boxes = _select_output(
                det_outputs,
                ["pred_boxes_all_layers", "pred_boxes", "boxes"],
            )
            pred_logits = _select_output(
                det_outputs,
                ["pred_logits_all_layers", "pred_logits", "scores"],
            )

            pred_bbox_xyxy: Optional[torch.Tensor] = None

            if pred_boxes is None or pred_logits is None:
                logger.warning(
                    "Detector outputs missing boxes/logits for %s; using GT bbox", filename
                )
                if gt_bbox is not None:
                    pred_bbox_xyxy = gt_bbox.clone()
                    fallback_bbox += 1
                else:
                    logger.warning("No GT bbox available for %s; skipping sample", filename)
                    continue
            else:
                if pred_boxes.ndim == 4:
                    pred_boxes = pred_boxes[-1]
                if pred_logits.ndim == 4:
                    pred_logits = pred_logits[-1]

                probs = (
                    1.0 / (1.0 + np.exp(-pred_logits[..., 1]))
                    if pred_logits.shape[-1] > 1
                    else 1.0 / (1.0 + np.exp(-pred_logits[..., 0]))
                )
                best_idx = int(np.argmax(probs, axis=1)[0])
                best_box = torch.from_numpy(pred_boxes[0, best_idx]).float()
                bbox_norm = box_cxcywh_to_xyxy(best_box.unsqueeze(0))[0]

                pred_bbox_xyxy = torch.tensor(
                    [
                        float(bbox_norm[0] * orig_w),
                        float(bbox_norm[1] * orig_h),
                        float(bbox_norm[2] * orig_w),
                        float(bbox_norm[3] * orig_h),
                    ],
                    dtype=torch.float32,
                )
                pred_bbox_xyxy = _clamp_bbox_xyxy(pred_bbox_xyxy, orig_w, orig_h)

                if (pred_bbox_xyxy[2] - pred_bbox_xyxy[0] < 1) or (
                    pred_bbox_xyxy[3] - pred_bbox_xyxy[1] < 1
                ):
                    logger.debug("Predicted bbox degenerate for %s; using GT bbox", filename)
                    if gt_bbox is not None:
                        pred_bbox_xyxy = gt_bbox.clone()
                        fallback_bbox += 1
                    else:
                        logger.warning(
                            "Degenerate bbox and no GT available for %s; skipping", filename
                        )
                        continue

            if pred_bbox_xyxy is None or torch.all(pred_bbox_xyxy.eq(0)):
                logger.warning("No valid bbox for %s; skipping sample", filename)
                continue

            bbox_np = pred_bbox_xyxy.cpu().numpy()
            x1 = max(0, int(np.floor(bbox_np[0])))
            y1 = max(0, int(np.floor(bbox_np[1])))
            x2 = min(orig_w, int(np.ceil(bbox_np[2])))
            y2 = min(orig_h, int(np.ceil(bbox_np[3])))
            if x2 <= x1 or y2 <= y1:
                logger.warning("Invalid crop for %s; skipping sample", filename)
                continue

            pose_crop = orig_image[y1:y2, x1:x2, :]
            if pose_crop.size == 0:
                logger.warning("Empty crop for %s; skipping sample", filename)
                continue

            pose_crop = pose_dataset.crop_image(orig_image, pred_bbox_xyxy.tolist())
            if pose_crop.size == 0:
                logger.warning("Empty crop for %s; skipping sample", filename)
                continue
            pose_input = _prepare_image_for_trt(
                pose_crop,
                (pose_height, pose_width),
                imagenet_mean_cv,
                imagenet_std_cv,
                imagenet_inv_std,
                resized_buffer=pose_resized_buffer,
                chw_buffer=pose_chw_buffer,
            )

            pose_batch = pose_input[np.newaxis, ...]
            bridge_times.append(time.perf_counter() - bridge_start)

            # Pose inference
            pose_outputs, pose_time = pose_trt.inference(pose_batch, get_timing=True)
            pose_times.append(pose_time)

            # Pose postprocessing
            post_start = time.perf_counter()
            if args.merge_outputs:
                pose_output = torch.from_numpy(pose_outputs["output"]).float()
                pred_translation_rel = pose_output[:, :3]
                pred_rotation_raw = pose_output[:, 3:]
            else:
                pred_translation_rel = torch.from_numpy(pose_outputs["translation"]).float()
                pred_rotation_raw = torch.from_numpy(pose_outputs["rotation"]).float()

            # Ensure batch dimension
            if pred_translation_rel.ndim == 1:
                pred_translation_rel = pred_translation_rel.unsqueeze(0)
            if pred_rotation_raw.ndim == 1:
                pred_rotation_raw = pred_rotation_raw.unsqueeze(0)

            if args.rotation_format == "matrix" or pred_rotation_raw.shape[1] == 6:
                pred_rotation_quat = rotation_matrix_to_quaternion(
                    process_rotation(pred_rotation_raw, "matrix")
                )
                if gt_rotation_raw.ndim == 2 and gt_rotation_raw.shape == (3, 3):
                    gt_rotation_quat = rotation_matrix_to_quaternion(gt_rotation_raw.unsqueeze(0))
                elif gt_rotation_raw.ndim == 1 and gt_rotation_raw.numel() == 6:
                    gt_rotation_quat = rotation_matrix_to_quaternion(
                        process_rotation(gt_rotation_raw.unsqueeze(0), "matrix")
                    )
                else:
                    gt_rotation_quat = gt_rotation_raw.unsqueeze(0)
            else:
                if args.normalize_quaternions:
                    pred_rotation_raw = pred_rotation_raw / torch.norm(
                        pred_rotation_raw, dim=1, keepdim=True
                    )
                pred_rotation_quat = pred_rotation_raw
                gt_rotation_quat = gt_rotation_raw.unsqueeze(0)

            # Convert translations to absolute coordinates
            pred_bbox_batch = pred_bbox_xyxy.unsqueeze(0)
            pred_translation_abs = bbox_relative_translation_to_translation(
                pred_translation_rel,
                pred_bbox_batch,
                camera,
            )

            if gt_bbox is not None and gt_bbox.numel() == 4:
                gt_translation_abs = bbox_relative_translation_to_translation(
                    gt_translation_rel.unsqueeze(0),
                    gt_bbox.unsqueeze(0),
                    camera,
                )
            else:
                gt_translation_abs = gt_translation_rel.unsqueeze(0)

            pred_rotation_abs = get_absolute_orientation(
                translation=pred_translation_abs,
                rotation_quat=pred_rotation_quat,
                no_rotation_compensation=args.no_rotation_compensation,
            )
            gt_rotation_abs = get_absolute_orientation(
                translation=gt_translation_abs,
                rotation_quat=gt_rotation_quat,
                no_rotation_compensation=args.no_rotation_compensation,
            )

            metrics = compute_metrics(
                pred_translation_abs,
                gt_translation_abs,
                pred_rotation_abs,
                gt_rotation_abs,
            )

            metric_key_map = {
                "translation_metric": "translation_metric_mean",
                "rotation_metric": "rotation_metric_mean",
                "total_metric": "total_metric",
                "relative_translation_metric": "relative_translation_metric_mean",
                "relative_total_metric": "relative_total_metric",
            }
            for key, metric_name in metric_key_map.items():
                if metric_name in metrics:
                    metrics_accumulator[key].append(metrics[metric_name])

            post_times.append(time.perf_counter() - post_start)

            pred_trans_cpu = pred_translation_abs.to("cpu")
            pred_rot_cpu = pred_rotation_abs.to("cpu")
            gt_trans_cpu = gt_translation_abs.to("cpu")
            gt_rot_cpu = gt_rotation_abs.to("cpu")

            for sample_idx in range(pred_trans_cpu.shape[0]):
                key = f"{filename}_{sample_idx}"
                predictions[key] = {
                    "prediction": {
                        "translation": pred_trans_cpu[sample_idx].tolist(),
                        "rotation": pred_rot_cpu[sample_idx].tolist(),
                    },
                    "ground_truth": {
                        "translation": gt_trans_cpu[sample_idx].tolist(),
                        "rotation": gt_rot_cpu[sample_idx].tolist(),
                    },
                }

    finally:
        det_trt.cleanup()
        pose_trt.cleanup()

    if not preprocess_times:
        return {}

    def _avg_std(values: list[float]) -> tuple[float, float]:
        arr = np.asarray(values, dtype=np.float64)
        return float(arr.mean() * 1000.0), float(arr.std() * 1000.0)

    avg_pre, std_pre = _avg_std(preprocess_times)
    avg_det, std_det = _avg_std(detector_times)
    avg_bridge, std_bridge = _avg_std(bridge_times)
    avg_pose, std_pose = _avg_std(pose_times)
    avg_post, std_post = _avg_std(post_times)

    avg_metrics = {k: float(sum(v) / len(v)) if v else 0.0 for k, v in metrics_accumulator.items()}

    results = {
        "mode": "pipeline",
        "num_samples": len(preprocess_times),
        "fallback_to_gt_bbox": fallback_bbox,
        "timings_ms": {
            "preprocess_avg": avg_pre,
            "preprocess_std": std_pre,
            "detector_infer_avg": avg_det,
            "detector_infer_std": std_det,
            "bridge_avg": avg_bridge,
            "bridge_std": std_bridge,
            "pose_infer_avg": avg_pose,
            "pose_infer_std": std_pose,
            "postprocess_avg": avg_post,
            "postprocess_std": std_post,
        },
        "metrics": avg_metrics,
        "predictions": predictions,
    }

    logger.info("Pose metrics:")
    logger.info("  Translation metric: %.4f", avg_metrics["translation_metric"])
    logger.info("  Rotation metric: %.4f", avg_metrics["rotation_metric"])
    logger.info("  Total metric: %.4f", avg_metrics["total_metric"])
    logger.info(
        "  Relative translation metric: %.4f",
        avg_metrics["relative_translation_metric"],
    )
    logger.info(
        "  SPEED score (relative total metric): %.4f",
        avg_metrics["relative_total_metric"],
    )

    logger.info("Pipeline timings (ms):")
    logger.info(
        "  Preprocess %.2f ± %.2f | Detector %.2f ± %.2f | Bridge %.2f ± %.2f | Pose %.2f ± %.2f | Post %.2f ± %.2f",
        avg_pre,
        std_pre,
        avg_det,
        std_det,
        avg_bridge,
        std_bridge,
        avg_pose,
        std_pose,
        avg_post,
        std_post,
    )
    logger.info("Fallback to GT bbox for %d samples", fallback_bbox)

    return results


def evaluate_detector(args: argparse.Namespace) -> dict:
    """Evaluate a TensorRT object detector on the SPEED dataset.

    Runs inference and computes detection metrics including IoU and mAP.

    Args:
        args: Command line arguments containing engine path, dataset configuration,
            and evaluation parameters.

    Returns:
        Dictionary containing timing statistics, IoU metrics, and mAP results.
    """
    logger, _ = setup_logging(args=args, base_filename="evaluate_trt_latency_detector")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Prepare engine
    engine_path = getattr(args, "trt_engine_detector_path", None) or args.trt_engine_path
    results = {"engine_path": engine_path, "mode": "detector"}
    try:
        trt_model = TensorRTInference(engine_path)
        trt_model.load_engine()
        logger.info(f"Loaded TensorRT engine from {engine_path}")

        # Determine input shape from engine binding (N, C, H, W)
        input_shape = trt_model.input_binding["shape"]
        _, c, h, w = input_shape
        logger.info(f"Engine input shape: {input_shape}")

        # Build SPEED test dataset with matching preprocessing
        dataset_config = get_dataset_config(args.dataset)
        bbox_json_path = os.path.join(args.dataset_root_dir, dataset_config["bbox_annotations"])

        # Force full-image path and disable augmentation via args
        args_det = argparse.Namespace(**vars(args))
        setattr(args_det, "use_full_image", True)
        setattr(args_det, "no_pixel_augmentation", True)
        setattr(args_det, "no_spatial_augmentation", True)
        setattr(args_det, "no_crop_padding", True)

        test_dataset = SPEEDDataset(
            dataset_root_dir=args.dataset_root_dir,
            split="test",
            rotation_format=args.rotation_format,
            img_size=(h, w),
            bbox_json_path=bbox_json_path,
            args=args_det,
            dataset_name=args.dataset,
        )
        test_loader = DataLoader(
            test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers
        )

        # Run timed inferences with IoU evaluation against GT bboxes
        max_images = getattr(args, "detector_num_samples", 200)
        inference_times = []
        ious = []
        processed = 0
        # Collect predictions/targets for mAP
        pred_list = []
        tgt_list = []

        def xyxy_to_cxcywh_norm(b: torch.Tensor, W: int, H: int) -> torch.Tensor:
            # b: [N,4] in pixels
            x1, y1, x2, y2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
            w = (x2 - x1).clamp(min=0)
            h = (y2 - y1).clamp(min=0)
            cx = x1 + w * 0.5
            cy = y1 + h * 0.5
            out = torch.stack([cx / W, cy / H, w / W, h / H], dim=-1)
            return out

        for idx, batch in enumerate(tqdm(test_loader, desc="Detector TRT eval")):
            images, translations, rotations, bbox = batch
            if processed >= max_images:
                break
            try:
                images_np = images.numpy()  # [1,3,H,W]
                outputs, tsec = trt_model.inference(images_np, get_timing=True)
                inference_times.append(tsec)

                # Extract outputs robustly
                def pick(name_options):
                    for k in name_options:
                        if k in outputs:
                            return outputs[k]
                    return None

                pred_boxes = pick(["pred_boxes_all_layers", "pred_boxes", "boxes"])
                pred_logits = pick(["pred_logits_all_layers", "pred_logits", "scores"])
                if pred_boxes is None or pred_logits is None:
                    logger.warning("Missing expected detector outputs; skipping sample")
                    continue

                # Handle optional layer dimension
                if pred_boxes.ndim == 4:  # [L, B, Q, 4]
                    pred_boxes = pred_boxes[-1]
                if pred_logits.ndim == 4:  # [L, B, Q, C]
                    pred_logits = pred_logits[-1]

                # Shapes: [B=1, Q, 4], [B=1, Q, C]
                probs = (
                    1.0 / (1.0 + np.exp(-pred_logits[..., 1]))
                    if pred_logits.shape[-1] > 1
                    else 1.0 / (1.0 + np.exp(-pred_logits[..., 0]))
                )
                best_idx = int(np.argmax(probs, axis=1)[0])
                box_cxcywh = torch.from_numpy(pred_boxes[0, best_idx : best_idx + 1, :]).float()

                # Scale to pixels and convert to xyxy
                scale = torch.tensor([w, h, w, h], dtype=torch.float32)
                box_px = box_cxcywh * scale
                box_xyxy = box_cxcywh_to_xyxy(box_px)  # [1,4]

                # Compute IoU with GT bbox (xyxy in pixels) scaled to resized image size
                # Derive original image size from dataset paths
                try:
                    filename = test_dataset.data[idx]["filename"]
                    path = test_dataset._image_paths.get(filename, None)
                    if path and os.path.exists(path):
                        with Image.open(path) as _img:
                            orig_w, orig_h = _img.size
                    else:
                        orig_w, orig_h = w, h
                except Exception:
                    orig_w, orig_h = w, h
                scale_x = w / float(orig_w)
                scale_y = h / float(orig_h)
                gt_bbox = bbox.float().view(1, 4)
                gt_bbox[:, 0] *= scale_x
                gt_bbox[:, 2] *= scale_x
                gt_bbox[:, 1] *= scale_y
                gt_bbox[:, 3] *= scale_y
                iou = box_iou(box_xyxy, gt_bbox).item()
                ious.append(iou)
                processed += 1

                # For mAP: use all queries and their scores
                # Convert GT bbox (scaled) to normalized cxcywh
                gt_cxcywh = xyxy_to_cxcywh_norm(gt_bbox, W=w, H=h)
                # Convert pred boxes to torch (already normalized cxcywh)
                pred_boxes_t = torch.from_numpy(pred_boxes[0]).float()
                scores_t = torch.from_numpy(probs[0]).float()
                pred_list.append({"boxes": pred_boxes_t, "scores": scores_t})
                tgt_list.append({"boxes": gt_cxcywh})
            except Exception as e:
                logger.warning(f"Error in detector eval loop: {e}")

        avg_inference_time = float(np.mean(inference_times) * 1000)
        std_inference_time = float(np.std(inference_times) * 1000)
        fps = float(1000.0 / avg_inference_time) if avg_inference_time > 0 else 0.0

        results.update(
            {
                "average_inference_time_ms": avg_inference_time,
                "std_inference_time_ms": std_inference_time,
                "fps": fps,
                "num_samples": processed,
                "input_shape": input_shape,
            }
        )
        if ious:
            results["iou_mean"] = float(np.mean(ious))
            results["iou_std"] = float(np.std(ious))

        # Compute mAP over SPEED test using single-class evaluator
        if pred_list and tgt_list:
            ap = evaluate_predictions_single_class(pred_list, tgt_list)
            results["mAP"] = ap

        logger.info("\nDetector TRT Inference Performance:")
        logger.info(f"Average inference time: {avg_inference_time:.2f} ms")
        logger.info(f"Inference time std: {std_inference_time:.2f} ms")
        logger.info(f"Throughput: {fps:.2f} FPS")
        if ious:
            logger.info(f"Mean IoU (best query): {np.mean(ious):.4f}")
            logger.info(f"IoU std (best query): {np.std(ious):.4f}")
        if pred_list and tgt_list:
            ap50 = results["mAP"].get("AP@0.50", 0.0)
            ap75 = results["mAP"].get("AP@0.75", 0.0)
            map_all = results["mAP"].get("AP@[0.50:0.95]", 0.0)
            logger.info("Detector mAP summary (single-class):")
            logger.info(f"AP@0.50: {ap50:.4f}")
            logger.info(f"AP@0.75: {ap75:.4f}")
            logger.info(f"AP@[0.50:0.95]: {map_all:.4f}")

    except Exception as e:
        logger.error(f"Error during detector evaluation: {e}")
        raise
    finally:
        if "trt_model" in locals():
            trt_model.cleanup()

    return results


def parse_tuple(s: str) -> tuple[int, int]:
    """Parse an image size tuple from a comma-separated string.

    Args:
        s: String in format 'height,width' (e.g., '384,384').

    Returns:
        Tuple of (height, width) as integers.

    Raises:
        argparse.ArgumentTypeError: If the string format is invalid.
    """
    try:
        # Split the string by comma and convert to integers
        return tuple(map(int, s.split(",")))
    except ValueError:
        raise argparse.ArgumentTypeError("Image size must be height,width (e.g., '384,384')")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate TensorRT model(s) performance")

    # Modified image_size argument to use the parse_tuple function
    parser.add_argument(
        "--image_size",
        type=parse_tuple,
        default=(384, 384),
        help="Input image size as height,width (e.g., '384,384')",
    )

    # Rest of your arguments
    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        choices=["pose", "detector", "both", "pipeline"],
        default="pose",
        help="What to evaluate: pose model, detector, or both",
    )
    # Engine paths
    parser.add_argument(
        "--trt_engine_path",
        type=str,
        help="TensorRT engine path (legacy: used as pose engine if --trt_engine_pose_path not set)",
    )
    parser.add_argument(
        "--trt_engine_pose_path",
        type=str,
        help="Path to the TensorRT engine file for pose model",
    )
    parser.add_argument(
        "--trt_engine_detector_path",
        type=str,
        help="Path to the TensorRT engine file for detector",
    )
    parser.add_argument(
        "--detector_images_dir",
        type=str,
        help="Directory with images to run the detector on (used in detector/both modes)",
    )
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

    # Detector eval options
    parser.add_argument(
        "--detector_num_samples",
        type=int,
        default=200,
        help="Number of random samples to run for detector performance",
    )
    # Pose postprocessing timing
    parser.add_argument(
        "--measure_postproc",
        action="store_true",
        help="Measure pose postprocessing time (from raw outputs to final translation+rotation)",
    )

    args = parser.parse_args()

    # Set dataset_root_dir based on dataset choice if not specified
    if args.dataset_root_dir is None:
        dataset_config = get_dataset_config(args.dataset)
        args.dataset_root_dir = f"./{dataset_config['folder']}"

    # Route by mode
    results = {}
    if args.mode == "pipeline":
        pipeline_results = evaluate_pipeline(args)
        results["pipeline"] = pipeline_results
    else:
        if args.mode in ("pose", "both"):
            if not (args.trt_engine_pose_path or args.trt_engine_path):
                raise SystemExit("Pose engine path is required for pose evaluation")
            pose_results = evaluate_pose(args)
            results["pose"] = pose_results

        if args.mode in ("detector", "both"):
            if not (args.trt_engine_detector_path or args.trt_engine_path):
                raise SystemExit("Detector engine path is required for detector evaluation")
            det_results = evaluate_detector(args)
            results["detector"] = det_results

    # Optionally save combined results
    if args.output_json and results:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved combined results to {args.output_json}")
