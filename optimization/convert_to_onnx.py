# This script converts the Torch model to ONNX format

import argparse
import logging

import onnx
import torch
import tensorrt as trt

from src.models import FastPoseViT, SUPPORTED_VIT_MODELS, VIT_MODELS

# Object detector (LW-DETR) imports
from object_detector.LWDETR.lwdetr_models import create_lwdetr_model
from object_detector.constants import LWDETR_IMG_SIZES

import os
from src.utils import setup_logging


def convert_to_onnx(args: argparse.Namespace) -> None:
    """Convert a PyTorch model to ONNX format.

    Supports both pose estimation (FastPoseViT) and object detection (LW-DETR) models.
    Optionally converts the ONNX model to TensorRT engine if requested.

    Args:
        args: Command line arguments containing model configuration and paths.
            Required attributes include model_type, model_weights, onnx_model_path,
            and model-specific parameters.

    Raises:
        ValueError: If an invalid rotation format or unsupported model type is specified.
        FileNotFoundError: If model weights file does not exist.
    """

    # Set up logging
    logger, _ = setup_logging(args)

    # If log_dir does not exist, create it
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.model_type == "pose":
        # Set up the dimensions
        if args.rotation_format == "quaternion":
            out_dim_rotation = 4
        elif args.rotation_format == "matrix":
            out_dim_rotation = 6
        else:
            raise ValueError("Invalid rotation format")
        out_dim_translation = 3

        # Load the pose model (FastPoseViT)
        model = FastPoseViT(
            vit_model=args.vit_model,
            num_hidden_layers=args.num_hidden_layers,
            hidden_layer_dim=args.hidden_layer_dim,
            out_dim_translation=out_dim_translation,
            out_dim_rotation=out_dim_rotation,
            vit_weights=args.vit_weights,
            merge_outputs=args.merge_outputs,
            nb_class_tokens=args.nb_class_tokens,
            use_layer_norm=args.use_layer_norm,
            dropout_rate=args.dropout_rate,
            use_residual=args.use_residual,
            no_mlp=args.no_mlp,
        ).to(device)

        model.load_from_pretrained(args.model_weights)

        # Get the image size for the ViT model
        image_size = VIT_MODELS[args.vit_model][3] if args.vit_model in VIT_MODELS else (224, 224)

        # Create a dummy input tensor
        dummy_input = torch.randn(1, 3, *image_size).to(device)

        logger.info(f"Exporting ONNX model to {args.onnx_model_path}")

        torch.onnx.export(
            model,  # The model
            dummy_input,  # Input tensor (dummy input)
            args.onnx_model_path,  # Output file name
            export_params=True,  # Store the trained parameter weights inside the model file
            opset_version=22,  # Use modern opset version
            dynamo=True,  # Use modern PyTorch 2.0+ export path
            do_constant_folding=True,  # Whether to apply optimizations like constant folding
            input_names=["input"],  # Input tensor names (optional)
            output_names=["output"]  # Give a name to the output node
            if model.merge_outputs
            else ["translation", "rotation"],  # Output tensor names (optional)
        )

    elif args.model_type == "detector":
        # Load the LW-DETR model via wrapper
        logger.info(f"Initializing LW-DETR variant='{args.detector_variant}' for ONNX export")
        wrapper = create_lwdetr_model(variant=args.detector_variant, num_classes=1)

        if not os.path.exists(args.model_weights):
            raise FileNotFoundError(f"Model weights not found: {args.model_weights}")

        # Load weights and move to device
        wrapper.load_from_pretrained(args.model_weights)
        wrapper = wrapper.to(device)
        wrapper.eval()

        # Get underlying LW-DETR model instance
        if hasattr(wrapper, "lwdetr_model"):
            model = wrapper.lwdetr_model
        else:
            # Fallback: try to export the wrapper directly (will likely return dicts)
            logger.warning("LW-DETR export() not found; exporting wrapper forward (dict outputs)")
            model = wrapper

        # Determine input size from constants
        if args.detector_variant not in LWDETR_IMG_SIZES:
            raise ValueError(
                f"Unknown LW-DETR variant '{args.detector_variant}'. Supported: {list(LWDETR_IMG_SIZES.keys())}"
            )
        size = LWDETR_IMG_SIZES[args.detector_variant]
        image_size = (size, size)

        # If static export, set static spatial shapes for transformer based on projector scales
        if args.static_export and hasattr(model, "transformer"):
            # Base grid size after ViT patch embed (patch size 16)
            base = size // 16
            # Retrieve projector scales from backbone to compute feature map sizes
            level2scalefactor = {"P3": 2.0, "P4": 1.0, "P5": 0.5, "P6": 0.25}
            scales = []
            try:
                scales = list(getattr(model.backbone, "projector_scale", ["P4"]))
            except Exception:
                scales = ["P4"]
            hw_list = []
            for lvl in scales:
                sf = level2scalefactor.get(lvl, 1.0)
                h = int(round(base * sf))
                w = int(round(base * sf))
                hw_list.append([h, w])
            static_shapes = torch.tensor(hw_list, dtype=torch.long)
            try:
                model.transformer.static_spatial_shapes = static_shapes
                model.transformer.static_spatial_shapes_list = [
                    (int(h), int(w)) for h, w in hw_list
                ]
            except Exception:
                pass

        # After static shapes are set, switch to export-friendly forward
        if hasattr(model, "export"):
            model.export()
        if (
            args.static_export
            and hasattr(model, "transformer")
            and hasattr(model.transformer, "export")
        ):
            model.transformer.export()
        if hasattr(model, "backbone") and hasattr(model.backbone, "export"):
            model.backbone.export()
        if hasattr(model, "decoder") and hasattr(model.decoder, "export"):
            model.decoder.export()

        # Create dummy input (tensor only; NestedTensor not required in export mode)
        dummy_input = torch.randn(1, 3, *image_size, device=device)

        logger.info(f"Exporting LW-DETR ONNX model to {args.onnx_model_path}")
        if args.static_export:
            logger.info("Static export enabled: using integer spatial shapes and fixed grids")

        # Try PyTorch 2 dynamo exporter first; fall back to legacy exporter with ATen fallback.
        try:
            torch.onnx.export(
                model,
                dummy_input,
                args.onnx_model_path,
                export_params=True,
                opset_version=22,
                dynamo=True,
                do_constant_folding=True,
                input_names=["images"],
                output_names=["pred_boxes_all_layers", "pred_logits_all_layers"],
            )
        except Exception as e:
            logger.warning(
                f"Dynamo ONNX export failed for LW-DETR due to: {e.__class__.__name__}: {e}. "
                "Falling back to legacy exporter with ATen fallback on CPU."
            )
            # Move to CPU to avoid CUDA custom kernel tracing issues
            model_cpu = model.to("cpu")

            # Some LW-DETR modules cache position embeddings during export() in
            # a raw attribute (e.g., pos_embed_export) which isn't a registered
            # buffer/parameter. Ensure such cached tensors are moved to CPU.
            def _move_cached_attrs_to_device(mod: torch.nn.Module, device: torch.device):
                # Move known cached tensors if present
                for attr_name in ("pos_embed_export",):
                    if hasattr(mod, attr_name):
                        attr = getattr(mod, attr_name)
                        if isinstance(attr, torch.Tensor) and attr.device.type != device.type:
                            setattr(mod, attr_name, attr.to(device))
                for child in mod.children():
                    _move_cached_attrs_to_device(child, device)

            _move_cached_attrs_to_device(model_cpu, torch.device("cpu"))
            dummy_cpu = dummy_input.to("cpu")
            # Legacy exporter with ATen fallback keeps unsupported ops as ATen ops in ONNX
            operator_export_type = getattr(torch.onnx, "OperatorExportTypes", None)
            kwargs = {}
            if operator_export_type is not None:
                kwargs["operator_export_type"] = operator_export_type.ONNX_ATEN_FALLBACK
            torch.onnx.export(
                model_cpu,
                dummy_cpu,
                args.onnx_model_path,
                export_params=True,
                opset_version=17,  # more permissive for legacy path
                dynamo=False,
                do_constant_folding=True,
                input_names=["images"],
                output_names=["pred_boxes_all_layers", "pred_logits_all_layers"],
                **kwargs,
            )
    else:
        raise ValueError(f"Unsupported model_type '{args.model_type}'. Use 'pose' or 'detector'")

    onnx_model = onnx.load(args.onnx_model_path)
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX model exported successfully!")

    # Convert to TensorRT if requested
    if args.create_trt_engine:
        logger.info(f"Converting ONNX to TensorRT engine: {args.trt_engine_path}")
        convert_onnx_to_tensorrt(args, logger)


def convert_onnx_to_tensorrt(args: argparse.Namespace, logger: logging.Logger) -> None:
    """Convert an ONNX model to a TensorRT engine.

    Uses the TensorRT Python API to parse an ONNX model and build an optimized
    TensorRT engine with optional FP16 or INT8 precision.

    Args:
        args: Command line arguments containing ONNX path, TensorRT engine path,
            workspace size, and precision flags (trt_fp16, trt_int8).
        logger: Logger instance for status messages and error reporting.

    Raises:
        RuntimeError: If ONNX parsing fails or TensorRT engine build fails.
    """
    try:
        # Create TensorRT logger and builder
        trt_logger = trt.Logger(trt.Logger.INFO)
        builder = trt.Builder(trt_logger)

        # Create network and parser
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(network_flags)
        parser = trt.OnnxParser(network, trt_logger)

        # Parse ONNX model
        logger.info(f"Parsing ONNX model: {args.onnx_model_path}")
        with open(args.onnx_model_path, "rb") as model_file:
            if not parser.parse(model_file.read()):
                logger.error("Failed to parse ONNX model")
                for error_idx in range(parser.num_errors):
                    error = parser.get_error(error_idx)
                    logger.error(f"Parser error {error_idx}: {error}")
                raise RuntimeError("ONNX parsing failed")

        logger.info(
            f"Successfully parsed ONNX model with {network.num_inputs} inputs and {network.num_outputs} outputs"
        )

        # Log network inputs/outputs
        for i in range(network.num_inputs):
            input_tensor = network.get_input(i)
            logger.info(
                f"Input {i}: {input_tensor.name}, shape: {input_tensor.shape}, dtype: {input_tensor.dtype}"
            )

        for i in range(network.num_outputs):
            output_tensor = network.get_output(i)
            logger.info(
                f"Output {i}: {output_tensor.name}, shape: {output_tensor.shape}, dtype: {output_tensor.dtype}"
            )

        # Create builder configuration
        config = builder.create_builder_config()

        # Set workspace memory limit
        workspace_size = args.trt_workspace_size * (1 << 20)  # Convert MB to bytes
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)
        logger.info(f"Workspace memory limit: {args.trt_workspace_size} MB")

        # Enable precision modes
        if args.trt_fp16:
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                logger.info("FP16 precision enabled")
            else:
                logger.warning("FP16 not supported on this platform")

        if args.trt_int8:
            if builder.platform_has_fast_int8:
                config.set_flag(trt.BuilderFlag.INT8)
                logger.info("INT8 precision enabled")
            else:
                logger.warning("INT8 not supported on this platform")

        # Build the engine
        logger.info("Building TensorRT engine (this may take several minutes)...")
        serialized_engine = builder.build_serialized_network(network, config)

        if serialized_engine is None:
            raise RuntimeError("Failed to build TensorRT engine")

        # Save the engine
        with open(args.trt_engine_path, "wb") as f:
            f.write(serialized_engine)

        # Get engine size from the serialized bytes
        engine_bytes = bytes(serialized_engine)
        engine_size_mb = len(engine_bytes) / (1024 * 1024)
        logger.info(f"TensorRT engine saved to: {args.trt_engine_path}")
        logger.info(f"Engine size: {engine_size_mb:.2f} MB")

        # Verify the engine can be loaded
        runtime = trt.Runtime(trt_logger)
        engine = runtime.deserialize_cuda_engine(engine_bytes)
        if engine is None:
            raise RuntimeError("Failed to deserialize saved engine")

        logger.info("TensorRT engine conversion completed successfully!")

    except Exception as e:
        logger.error(f"TensorRT conversion failed: {e}")
        raise


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Convert Torch model to ONNX model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # General selection: which architecture to export
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["pose", "detector"],
        default="pose",
        help="Select architecture: 'pose' (ViT pose) or 'detector' (LW-DETR)",
    )

    # Pose model (FastPoseViT) specific arguments
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
        "--model_weights",
        type=str,
        help="Path to the weights checkpoints (pose or detector)",
        default="best_FastPoseViT.pth",
    )
    parser.add_argument(
        "--onnx_model_path", type=str, help="Path to the ONNX model", default="FastPoseViT.onnx"
    )
    parser.add_argument(
        "--vit_weights", type=str, default=None, help="Path to pretrained ViT weights"
    )
    parser.add_argument(
        "--vit_model",
        type=str,
        choices=SUPPORTED_VIT_MODELS,
        default="vit_b_16",
        help="ViT model to use",
    )
    parser.add_argument(
        "--log_dir", type=str, default="./runs", help="Directory to store TensorBoard logs"
    )
    parser.add_argument(
        "--merge_outputs",
        action="store_true",
        help="Merge translation and rotation outputs into a single MLP",
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
    parser.add_argument(
        "--new_Z",
        action="store_true",
        help="Use least squares solution for avg_ratio computation instead of geometric mean",
    )
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
    parser.add_argument(
        "--no_mlp",
        action="store_true",
        help="Skip MLP layers and use direct projection only",
    )
    parser.add_argument(
        "--no_crop_padding",
        action="store_true",
        help="Don't pad cropped images to make them square",
    )
    parser.add_argument(
        "--skip_tensorboard",
        action="store_true",
        help="Skip tensorboard logging to save time and reduce I/O",
    )

    # Detector (LW-DETR) specific arguments
    parser.add_argument(
        "--detector_variant",
        type=str,
        choices=list(LWDETR_IMG_SIZES.keys()),
        default="tiny",
        help="LW-DETR variant to export when --model_type=detector",
    )
    parser.add_argument(
        "--static_export",
        action="store_true",
        help="Enable static export path (fixed integer spatial shapes) for detector",
    )

    # TensorRT conversion arguments
    parser.add_argument(
        "--create_trt_engine",
        action="store_true",
        help="Convert ONNX model to TensorRT engine after ONNX export",
    )
    parser.add_argument(
        "--trt_engine_path",
        type=str,
        default="model.trt",
        help="Path to save the TensorRT engine file",
    )
    parser.add_argument(
        "--trt_workspace_size",
        type=int,
        default=1024,
        help="TensorRT workspace memory size in MB",
    )
    parser.add_argument(
        "--trt_fp16",
        action="store_true",
        help="Enable FP16 precision for TensorRT engine",
    )
    parser.add_argument(
        "--trt_int8",
        action="store_true",
        help="Enable INT8 precision for TensorRT engine",
    )

    args = parser.parse_args()

    convert_to_onnx(args)
