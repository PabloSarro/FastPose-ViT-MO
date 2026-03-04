# This file contains the main training script for the RF-DETR model on the SPEED dataset

import argparse
import os
import torch
from styleaug import StyleAugmentor

# Anomaly detection disabled after fixing inplace operations

# Fix torch.compile Docker UID issues by setting comprehensive cache directories
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/tmp/torchinductor_cache")
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_cache")
os.environ.setdefault("PYTORCH_TRITON_CACHE_DIR", "/tmp/triton_cache")
# Set user environment variables to avoid getpwuid lookups
os.environ.setdefault("USER", "pytorch")
os.environ.setdefault("LOGNAME", "pytorch")
os.environ.setdefault("HOME", "/tmp")
# Fix CUDA memory fragmentation issues
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler
from object_detector.optimizer import SingleDeviceMuonWithAuxAdam
import os
import numpy as np
from object_detector.LWDETR.lwdetr_models import create_lwdetr_model, get_lwdetr_loss_original
from object_detector.scheduler import (
    LogarithmicLR,
    CosineAnnealingLR,
    CosineAnnealingWithWarmRestarts,
)
import logging

from object_detector.constants import (
    LWDETR_IMG_SIZES,
    DEFAULT_T_0_EPOCHS,
    DEFAULT_T_MULT,
    DEFAULT_LR_ENCODER_FACTOR,
    DEFAULT_LR_COMPONENT_DECAY,
    DEFAULT_LR_VIT_LAYER_DECAY,
)
from object_detector.datasets import create_dataloaders, verify_dataset
from src.datasets import get_dataset_config

# LW-DETR uses Hungarian loss with set prediction
from object_detector.engine import train_epoch, val_epoch, TrainingConfig
from object_detector.metrics import format_metrics
from object_detector.utils import (
    setup_logging,
    create_output_dir,
    format_number,
)


def create_style_augmentor(device: torch.device | None = None) -> StyleAugmentor:
    """Create and initialize StyleAugmentor.

    Args:
        device: Device to move the augmentor to. If None, stays on default device.

    Returns:
        Initialized StyleAugmentor instance.
    """
    style_augmentor = StyleAugmentor()
    if device is not None:
        style_augmentor = style_augmentor.to(device)
    return style_augmentor


def train_model(
    model_variant: str,
    dataset_root_dir: str,
    bbox_json_path: str | None = None,
    epochs: int = 20,
    batch_size: int = 16,
    max_lr: float = 1e-5,
    min_lr: float = 1e-6,
    weight_decay: float = 0.0,
    optimizer_type: str = "adam",
    scheduler_type: str = "cosineannealinglr",
    output_dir: str = "./checkpoints",
    log_dir: str = "./runs",
    num_workers: int = 4,
    resume_from: str | None = None,
    model_save_path: str | None = None,
    no_pixel_augmentation: bool = False,
    no_spatial_augmentation: bool = False,
    domain_gap_pixel_augmentation: bool = False,
    do_style_aug: bool = False,
    no_compile_model: bool = False,
    max_grad_norm: float = 0.0,
    use_per_epoch_scheduling: bool = False,
    pretrain_weights: str | None = None,
    pretrained_encoder: str | None = None,
    num_queries: int | None = None,
    static_training: bool = False,
    convert_to_grayscale: bool = False,
    logger: "logging.Logger | None" = None,
) -> None:
    """Train the LW-DETR object detection model.

    Args:
        model_variant: Model variant to use ("tiny", "small", "medium").
        dataset_root_dir: Root directory of SPEED dataset.
        bbox_json_path: Path to bounding box JSON file. If None, uses
            dataset_root_dir/speed_bbox_annotations.json.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        max_lr: Maximum learning rate for scheduler.
        min_lr: Minimum learning rate for scheduler.
        weight_decay: Weight decay for optimizer (currently disabled).
        optimizer_type: Optimizer to use ("adam", "adamw", "muon").
        scheduler_type: Learning rate scheduler ("cosineannealinglr",
            "cosineannealingwarmrestarts", "logarithmiclr").
        output_dir: Directory to save outputs.
        log_dir: Directory to store TensorBoard logs.
        num_workers: Number of data loading workers.
        resume_from: Path to checkpoint to resume from.
        model_save_path: Path to save the best model.
        no_pixel_augmentation: Disable pixel data augmentation.
        no_spatial_augmentation: Disable spatial data augmentation.
        domain_gap_pixel_augmentation: Use domain gap pixel augmentation.
        do_style_aug: Enable neural style augmentation.
        no_compile_model: Disable torch.compile optimization.
        max_grad_norm: Maximum gradient norm for clipping (0 disables).
        use_per_epoch_scheduling: Use per-epoch instead of per-iteration scheduling.
        pretrain_weights: Path to pretrained model weights.
        pretrained_encoder: Path to pretrained encoder weights.
        num_queries: Override number of queries for LW-DETR.
        static_training: Enable static-shape training for torch.compile.
        convert_to_grayscale: Convert images to grayscale.
        logger: Logger instance to use.
    """

    # Setup logging - use passed logger if provided
    if logger is None:
        logger, _ = setup_logging()  # Fallback to simple logging

    logger.info("Starting training...")

    # Setup device - always use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Enable basic CUDA optimizations
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        # Remove empty_cache() call to stabilize GPU power usage during training
        logger.info("CUDA optimizations enabled")

    # Create output directory
    output_path = create_output_dir(output_dir)
    logger.info(f"Output directory: {output_path}")

    # Create directory for model_save_path if specified
    if model_save_path:
        model_save_folder = os.path.dirname(model_save_path)
        if model_save_folder and not os.path.exists(model_save_folder):
            os.makedirs(model_save_folder)
            logger.info(f"Created directory for model save path: {model_save_folder}")

    # Setup dataset paths
    images_dir = os.path.join(dataset_root_dir, "images")
    if bbox_json_path is None:
        bbox_json_path = os.path.join(dataset_root_dir, "speed_bbox_annotations.json")

    logger.info(f"Images directory: {images_dir}")
    logger.info(f"Bounding box JSON file: {bbox_json_path}")

    # Verify dataset
    logger.info("Verifying dataset...")
    if not verify_dataset(images_dir, bbox_json_path, split="train"):
        logger.error("Dataset verification failed!")
        return
    logger.info("Dataset verification passed!")

    # Get model configuration for LW-DETR
    if model_variant not in LWDETR_IMG_SIZES:
        raise ValueError(
            f"Unknown LW-DETR variant: {model_variant}. Choose from {list(LWDETR_IMG_SIZES.keys())}"
        )
    img_size_px = LWDETR_IMG_SIZES[model_variant]

    img_size = (img_size_px, img_size_px)
    logger.info(f"Using image size: {img_size} for LW-DETR-{model_variant}")
    logger.info(f"Convert to grayscale: {convert_to_grayscale}")

    # Create data loaders
    logger.info("Creating data loaders...")
    use_persistent_workers = num_workers > 1
    optimal_prefetch = min(4, max(2, num_workers))

    train_loader, val_loader = create_dataloaders(
        images_dir=images_dir,
        annotations_file=bbox_json_path,
        batch_size=batch_size,
        img_size=img_size,
        num_workers=num_workers,
        no_pixel_augmentation=no_pixel_augmentation,
        no_spatial_augmentation=no_spatial_augmentation,
        domain_gap_pixel_augmentation=domain_gap_pixel_augmentation,
        do_style_aug=do_style_aug,
        enable_caching=True,
        prefetch_factor=optimal_prefetch,
        persistent_workers=use_persistent_workers,
        convert_to_grayscale=convert_to_grayscale,
    )

    logger.info(
        f"Training samples: {len(train_loader.dataset)}, Validation samples: {len(val_loader.dataset)}"
    )

    # Initialize StyleAugmentor if needed (like src/)
    style_augmentor = None
    if domain_gap_pixel_augmentation and do_style_aug:
        style_augmentor = create_style_augmentor(device)

    # Always use Hungarian loss with multi-query configuration
    if num_queries is None or num_queries <= 1:
        # Set num_queries based on model variant defaults
        if model_variant == "tiny":
            num_queries = 100
        else:
            num_queries = 300

    # Create LW-DETR model for optimal speed
    logger.info(f"Creating LW-DETR--{model_variant} model for single satellite detection...")
    model = create_lwdetr_model(variant=model_variant, num_classes=1, num_queries=num_queries)

    # Handle pretrained weights for LW-DETR
    if pretrain_weights is not None:
        logger.info(f"Loading LW-DETR pretrained weights from: {pretrain_weights}")
        try:
            loaded, missing, unexpected = model.load_from_pretrained(pretrain_weights)
            logger.info(
                f"✅ LW-DETR pretrained weights loaded: {loaded} layers loaded, {missing} missing, {unexpected} skipped"
            )
        except Exception as e:
            logger.warning(f"Failed to load LW-DETR pretrained weights: {e}")
            logger.info("Continuing with random initialization...")
    elif pretrained_encoder is not None:
        logger.info(f"Loading LW-DETR encoder weights from: {pretrained_encoder}")
        try:
            # For encoder-only loading, we would need to implement this in the model
            logger.warning("Encoder-only loading not yet implemented for LW-DETR")
            logger.info("Continuing with random initialization...")
        except Exception as e:
            logger.warning(f"Failed to load LW-DETR encoder weights: {e}")
            logger.info("Continuing with random initialization...")
    else:
        logger.info(f"LW-DETR {model_variant} initialized with random weights")

    model = model.to(device)

    # Optional: enable static-shape training for improved torch.compile graph stability
    if static_training:
        try:
            # Compute static feature map shapes based on variant size and backbone projector scales
            base = img_size_px // 16  # ViT patch size 16
            level2scalefactor = {"P3": 2.0, "P4": 1.0, "P5": 0.5, "P6": 0.25}
            scales = list(getattr(model.lwdetr_model.backbone, "projector_scale", ["P4"]))
            hw_list = []
            for lvl in scales:
                sf = level2scalefactor.get(lvl, 1.0)
                h = int(round(base * sf))
                w = int(round(base * sf))
                hw_list.append([h, w])
            import torch as _torch

            static_shapes_tensor = _torch.tensor(hw_list, dtype=_torch.long, device=device)
            # Attach to transformer
            tr = model.lwdetr_model.transformer
            tr.static_spatial_shapes = static_shapes_tensor
            tr.static_spatial_shapes_list = [(int(h), int(w)) for h, w in hw_list]
            tr._static_training = True
            # Propagate to cross attention for constant offset normalizer and shape usage
            for layer in tr.decoder.layers:
                if hasattr(layer, "cross_attn"):
                    layer.cross_attn.static_hw = tr.static_spatial_shapes_list
            logger.info(
                f"Static training enabled; feature shapes set to {tr.static_spatial_shapes_list}"
            )
        except Exception as e:
            logger.warning(f"Failed to enable static training: {e}")

    # Always set return_all_queries to False for proper inference-style validation
    if hasattr(model, "return_all_queries"):
        model.return_all_queries = False

    criterion = get_lwdetr_loss_original()
    logger.info("Using LW-DETR Hungarian SetCriterion loss")
    # Ensure wrapper returns all queries for validation loss & metrics
    if hasattr(model, "return_all_queries"):
        model.return_all_queries = True

    # Apply torch.compile unless disabled
    if not no_compile_model:
        try:
            # With static training enabled, aot-autograd can be more stable
            compile_mode = "max-autotune-no-cudagraphs"
            model = torch.compile(model, mode=compile_mode)
            logger.info("Model compiled with torch.compile")
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}. Continuing with uncompiled model.")

    logger.info(f"Using direct prediction LW-DETR-{model_variant} model")

    # Log model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Model: {format_number(total_params)} total, {format_number(trainable_params)} trainable parameters"
    )

    # Calculate training parameters
    batches_per_epoch = len(train_loader)
    total_iterations = batches_per_epoch * epochs

    logger.info(
        f"Training: {batches_per_epoch} batches/epoch, {total_iterations} total iterations"
    )
    logger.info(f"Learning rates: max={max_lr:.2e}, min={min_lr:.2e}")

    # Initialize advanced scheduler settings (like src/)
    # Choose between per-iteration and per-epoch scheduling
    # Per-epoch scheduling is more conservative and may help with convergence
    use_per_iteration_scheduling = not use_per_epoch_scheduling
    steps_per_epoch_for_scheduler = batches_per_epoch if use_per_iteration_scheduling else 1

    # Set up optimizer and scheduler
    match optimizer_type:
        case "adamw" | "adam":
            # Create optimizer with RF-DETR's parameter grouping strategy
            logger.info("Setting up RF-DETR AdamW optimizer with differentiated learning rates...")
            param_groups = model.get_parameter_groups(
                lr=max_lr,
                lr_encoder=max_lr * DEFAULT_LR_ENCODER_FACTOR,
                lr_component_decay=DEFAULT_LR_COMPONENT_DECAY,
                lr_vit_layer_decay=DEFAULT_LR_VIT_LAYER_DECAY,
            )
            optimizer = optim.AdamW(param_groups, weight_decay=0.0)  # No weight decay

            if scheduler_type == "cosineannealinglr":
                scheduler = CosineAnnealingLR(
                    optimizer,
                    max_lr,
                    min_lr,
                    T_max_epochs=epochs,
                    steps_per_epoch=steps_per_epoch_for_scheduler,
                )
            elif scheduler_type == "cosineannealingwarmrestarts":
                scheduler = CosineAnnealingWithWarmRestarts(
                    optimizer,
                    max_lr,
                    min_lr,
                    T_0=DEFAULT_T_0_EPOCHS,
                    T_mult=DEFAULT_T_MULT,
                    steps_per_epoch=steps_per_epoch_for_scheduler,
                )
            elif scheduler_type == "logarithmiclr":
                scheduler = LogarithmicLR(
                    optimizer,
                    max_lr,
                    min_lr,
                    T_max_epochs=epochs,
                    steps_per_epoch=steps_per_epoch_for_scheduler,
                )
            else:
                raise ValueError(f"Unsupported scheduler for AdamW: {scheduler_type}")

        case "muon":
            logger.info("Setting up Muon optimizer with parameter separation...")

            # Separate parameters for Muon using vectorized filtering (like src/)
            model_params = list(model.named_parameters())
            param_tensors = [param for _, param in model_params]
            param_ndims = torch.tensor([p.ndim for p in param_tensors])

            # Vectorized parameter separation
            muon_mask = param_ndims >= 2
            muon_params = [p for p, is_muon in zip(param_tensors, muon_mask) if is_muon]
            adam_params = [p for p, is_muon in zip(param_tensors, muon_mask) if not is_muon]

            # Create parameter groups with correct MuonWithAuxAdam format
            param_groups = []
            if muon_params:
                param_groups.append(
                    {
                        "params": muon_params,
                        "use_muon": True,
                        "lr": max_lr,
                    }
                )
            if adam_params:
                param_groups.append(
                    {
                        "params": adam_params,
                        "use_muon": False,
                        "lr": max_lr * 0.1,  # Adam parameters use 10% of Muon LR
                        "betas": (0.9, 0.95),
                    }
                )

            optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
            scheduler = None  # Handle Muon scheduling manually

            logger.info(
                f"Muon optimizer: {len(muon_params)} matrix params, {len(adam_params)} other params"
            )

        case _:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")

    # Setup TensorBoard - use log_dir like src/train.py
    writer = SummaryWriter(log_dir)

    # Initialize mixed precision scaler with optimized settings for power stability
    scaler = (
        GradScaler(
            device="cuda",
        )
        if device.type == "cuda"
        else None
    )
    if scaler:
        logger.info("Mixed precision training enabled")

    logger.info(
        f"Using {'per-iteration' if use_per_iteration_scheduling else 'per-epoch'} scheduling"
    )

    # Cache parameter group indices for Muon optimizer to avoid dictionary lookups (like src/)
    muon_param_groups = []
    adam_param_groups = []
    if optimizer_type == "muon":
        for idx, group in enumerate(optimizer.param_groups):
            if group.get("use_muon", False):
                muon_param_groups.append(idx)
            else:
                adam_param_groups.append(idx)

    # Pre-compute Muon scheduler values for performance (like src/)
    muon_lrs = None
    adam_lrs = None
    if optimizer_type == "muon":
        # Use the same scheduling approach as the Adam optimizer
        total_steps_for_muon = total_iterations if use_per_iteration_scheduling else epochs
        steps_per_epoch_for_muon = batches_per_epoch if use_per_iteration_scheduling else 1

        # Pre-compute constants outside the loop for efficiency
        if scheduler_type == "cosineannealingwarmrestarts":
            # Use the static method from CosineAnnealingWithWarmRestarts
            muon_lrs = CosineAnnealingWithWarmRestarts.compute_all_lrs(
                max_lr,
                min_lr,
                total_steps_for_muon,
                T_0_epochs=DEFAULT_T_0_EPOCHS,
                T_mult=DEFAULT_T_MULT,
                steps_per_epoch=steps_per_epoch_for_muon,
            )
        elif scheduler_type == "logarithmiclr":
            # Use the static method from LogarithmicLR
            muon_lrs = LogarithmicLR.compute_all_lrs(
                max_lr,
                min_lr,
                T_max_epochs=epochs,
                steps_per_epoch=steps_per_epoch_for_muon,
            )
        elif scheduler_type == "cosineannealinglr":
            # Use the static method from CosineAnnealingLR
            muon_lrs = CosineAnnealingLR.compute_all_lrs(
                max_lr,
                min_lr,
                T_max_epochs=epochs,
                steps_per_epoch=steps_per_epoch_for_muon,
            )
        else:
            raise ValueError(f"Unsupported scheduler for Muon: {scheduler_type}")

        # Compute Adam LRs in one vectorized operation
        adam_lrs = (np.array(muon_lrs) * 0.1).tolist()

    # Initialize global iteration counter for proper learning rate scheduling (like src/)
    global_iteration = 0

    # Resume from checkpoint if specified
    start_epoch = 0
    best_ap_50_95 = 0.0  # Track best AP@[0.50:0.95] (mAP)

    if resume_from and os.path.exists(resume_from):
        logger.info(f"Resuming from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_ap_50_95 = checkpoint.get("best_ap_50_95", 0.0)
        global_iteration = start_epoch * batches_per_epoch  # Update global iteration
        logger.info(f"Resumed from epoch {start_epoch}")

    # Final preparation before training loop
    if device.type == "cuda":
        # Keep only reset_peak_memory_stats, remove empty_cache for power stability
        torch.cuda.reset_peak_memory_stats()
    logger.info("Ready to start training")

    # Training loop
    logger.info("Starting training...")

    for epoch in range(start_epoch, epochs):
        logger.info(f"\nEpoch {epoch + 1}/{epochs}")

        # Get learning rate information for logging (like src/)
        if optimizer_type == "muon" and muon_lrs is not None:
            # For Muon with per-iteration scheduling, show start and end LR
            epoch_start_iteration = epoch * batches_per_epoch
            epoch_end_iteration = (epoch + 1) * batches_per_epoch - 1

            if use_per_iteration_scheduling:
                start_lr_index = min(epoch_start_iteration, len(muon_lrs) - 1)
                end_lr_index = min(epoch_end_iteration, len(muon_lrs) - 1)
                start_lr = muon_lrs[start_lr_index]
                end_lr = muon_lrs[end_lr_index]

                # Format LR display for per-iteration scheduling
                if start_lr != end_lr:
                    current_lr_display = f"{start_lr:.6f} → {end_lr:.6f}"
                else:
                    current_lr_display = f"{start_lr:.6f}"
                current_lr = start_lr  # For compatibility with TensorBoard logging
            else:
                lr_index = min(epoch, len(muon_lrs) - 1)
                current_lr = muon_lrs[lr_index]
                current_lr_display = f"{current_lr:.6f}"

            logger.info(f"Learning rate (Muon): {current_lr_display}")
        else:
            # For Adam, get the current learning rate from optimizer
            current_lr = optimizer.param_groups[0]["lr"]
            logger.info(f"Learning rate: {current_lr:.6f}")

        # Create training configuration
        train_config = TrainingConfig(
            max_grad_norm=max_grad_norm,
            muon_lrs=muon_lrs if optimizer_type == "muon" else None,
            adam_lrs=adam_lrs if optimizer_type == "muon" else None,
            muon_param_groups=muon_param_groups if optimizer_type == "muon" else None,
            adam_param_groups=adam_param_groups if optimizer_type == "muon" else None,
            use_per_iteration_scheduling=use_per_iteration_scheduling,
            global_iteration_start=global_iteration,
        )

        # Training (with SimpleIoU loss)
        train_loss, _ = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,  # Use SimpleIoU loss for training
            device=device,
            epoch=epoch,
            writer=writer,
            config=train_config,
            scaler=scaler,
            style_augmentor=style_augmentor,
            train_dataset=train_loader.dataset,
            scheduler=scheduler,
        )

        # Validation configuration is now simplified

        # Validation (with SimpleIoU loss)
        val_loss, val_metrics = val_epoch(
            model=model,
            val_loader=val_loader,
            criterion=criterion,  # Use SimpleIoU loss for validation
            device=device,
            epoch=epoch,
            writer=writer,
            compute_ap_metrics=True,
            inference_style_loss=True,  # Always use inference-style validation loss
        )

        # Update global iteration counter for next epoch (like src/)
        global_iteration += batches_per_epoch

        # Step scheduler only for per-epoch scheduling. For per-iteration
        # scheduling we already stepped inside the training loop.
        if scheduler is not None and not use_per_iteration_scheduling:
            scheduler.step()

        # Log results
        logger.info(f"Train Loss: {train_loss:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}")

        # Log AP metrics if available
        if val_metrics and any(key.startswith("AP@") for key in val_metrics):
            logger.info(f"\n{format_metrics(val_metrics)}")

        # Current learning rate for TensorBoard logging is already set above
        if optimizer_type != "muon":
            current_lr = (
                scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]["lr"]
            )

        # TensorBoard logging (simplified for IoU-only loss)
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("LearningRate", current_lr, epoch)

        # Log AP metrics to TensorBoard
        if val_metrics:
            for metric_name, metric_value in val_metrics.items():
                if metric_name.startswith("AP@") and isinstance(metric_value, (int, float)):
                    writer.add_scalar(f"Metrics/{metric_name}", metric_value, epoch)

        # Check for early stopping based on validation loss trends
        # Save best model based on AP@[0.50:0.95] (mAP)
        current_ap_50_95 = val_metrics.get("AP@[0.50:0.95]", 0.0) if val_metrics else 0.0
        if current_ap_50_95 > best_ap_50_95:
            best_ap_50_95 = current_ap_50_95
            if model_save_path:
                model.save(model_save_path)
                logger.info(
                    f"New best model saved (AP@[0.50:0.95]: {current_ap_50_95:.4f}): {model_save_path}"
                )
            else:
                # Fallback to default naming if no model_save_path specified
                best_model_path = output_path / f"best_model_{model_variant}.pth"
                model.save(str(best_model_path))
                logger.info(
                    f"New best model saved (AP@[0.50:0.95]: {current_ap_50_95:.4f}): {best_model_path}"
                )

    # Save final model
    final_model_path = output_path / f"final_model_{model_variant}.pth"
    model.save(str(final_model_path))
    logger.info(f"Final model saved: {final_model_path}")

    # Close TensorBoard writer
    writer.close()

    logger.info("Training completed!")
    logger.info(f"Best validation AP@[0.50:0.95]: {best_ap_50_95:.4f}")

    # No distributed cleanup needed - using SingleDeviceMuonWithAuxAdam


def main() -> None:
    """Main function with argument parsing for training script."""
    parser = argparse.ArgumentParser(
        description="Train RF-DETR on SPEED dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["SPEED", "SPEED_PLUS_SYNTHETIC", "SPEED_PLUS"],
        default="SPEED",
        help="Dataset to use for training",
    )
    parser.add_argument(
        "--dataset_root_dir",
        type=str,
        default=None,
        help="Root directory of dataset (auto-determined from dataset choice if not specified)",
    )

    # Model arguments
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["lwdetr"],
        default="lwdetr",
        help="Model type to use (only lwdetr is supported)",
    )
    parser.add_argument(
        "--model_variant",
        type=str,
        default="tiny",
        help="Model variant to use (nano/small/medium for RF-DETR, tiny/small/medium for LW-DETR)",
    )

    # Training arguments
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train")
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for training and validation"
    )
    parser.add_argument(
        "--max_lr",
        type=float,
        default=2e-4,
        help="Maximum learning rate for cosine annealing (reduced for stability)",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=2e-5,
        help="Minimum learning rate for cosine annealing (reduced for stability)",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay (disabled)")
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["adamw", "adam", "muon"],
        default="muon",
        help="Optimizer to use for training (adam/adamw or muon)",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=[
            "cosineannealinglr",
            "cosineannealingwarmrestarts",
            "logarithmiclr",
        ],
        default="cosineannealinglr",
        help="Learning rate scheduler to use",
    )

    # Hardware arguments
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for data loading"
    )

    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Output directory")
    parser.add_argument(
        "--log_dir", type=str, default="./runs", help="Directory to store TensorBoard logs"
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        default=None,
        help="Path to save the best model",
    )

    # Resume training
    parser.add_argument("--resume_from", type=str, default=None, help="Checkpoint to resume from")
    # Pretrained weights
    parser.add_argument(
        "--pretrain_weights", type=str, default=None, help="Path to pretrained model weights"
    )
    parser.add_argument(
        "--pretrained_encoder", type=str, default=None, help="Path to pretrained encoder weights"
    )

    # Experiments arguments
    parser.add_argument(
        "--no_pixel_augmentation",
        action="store_true",
        help="Don't use pixel data augmentation (default: enabled for stability)",
    )
    parser.add_argument(
        "--no_spatial_augmentation",
        action="store_true",
        help="Don't use spatial data augmentation (default: enabled for stability)",
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
        "--convert_to_grayscale",
        action="store_true",
        help="Convert images to grayscale (replicated as 3-channel for model compatibility)",
    )

    # Training optimization arguments
    parser.add_argument(
        "--no_compile_model",
        action="store_true",
        help="Disable torch.compile optimization (enabled by default)",
    )
    parser.add_argument(
        "--num_queries",
        type=int,
        default=None,
        help="Override number of queries for LW-DETR (set >1 to restore base model)",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,  # Enable gradient clipping by default for stability
        help="Maximum gradient norm for gradient clipping (0.0 disables clipping)",
    )
    parser.add_argument(
        "--use_per_epoch_scheduling",
        action="store_true",
        help="Use per-epoch learning rate scheduling instead of per-iteration (more conservative, may help convergence)",
    )
    parser.add_argument(
        "--static_training",
        action="store_true",
        help="Enable static-shape training to stabilize torch.compile graphs (fixed per-variant sizes)",
    )

    args = parser.parse_args()

    # Set dataset_root_dir based on dataset choice if not specified
    if args.dataset_root_dir is None:
        dataset_config = get_dataset_config(args.dataset)
        args.dataset_root_dir = f"./{dataset_config['folder']}"

    # Automatically determine bbox_json_path based on dataset
    dataset_config = get_dataset_config(args.dataset)
    bbox_json_path = os.path.join(args.dataset_root_dir, dataset_config["bbox_annotations"])

    # Set up logging with args
    logger, _ = setup_logging(args)

    # Log the arguments provided in a clean format
    logger.info("=========================================")
    logger.info("Arguments provided:")
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")
    logger.info("=========================================")
    logger.info("")

    # Start training
    train_model(
        model_variant=args.model_variant,
        dataset_root_dir=args.dataset_root_dir,
        bbox_json_path=bbox_json_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_lr=args.max_lr,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        optimizer_type=args.optimizer,
        scheduler_type=args.scheduler,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        num_workers=args.num_workers,
        resume_from=args.resume_from,
        model_save_path=args.model_save_path,
        no_pixel_augmentation=args.no_pixel_augmentation,
        no_spatial_augmentation=args.no_spatial_augmentation,
        domain_gap_pixel_augmentation=args.domain_gap_pixel_augmentation,
        do_style_aug=args.do_style_aug,
        no_compile_model=args.no_compile_model,
        num_queries=args.num_queries,
        max_grad_norm=args.max_grad_norm,
        use_per_epoch_scheduling=args.use_per_epoch_scheduling,
        pretrain_weights=args.pretrain_weights,
        pretrained_encoder=args.pretrained_encoder,
        static_training=args.static_training,
        convert_to_grayscale=args.convert_to_grayscale,
        logger=logger,
    )


if __name__ == "__main__":
    main()
