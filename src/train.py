# This script is used to train the FastPoseViT model on the SPEED dataset.

import argparse
import os
import torch

# Fix torch.compile Docker UID issues by setting comprehensive cache directories
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/tmp/torchinductor_cache")
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_cache")
os.environ.setdefault("PYTORCH_TRITON_CACHE_DIR", "/tmp/triton_cache")
# Set user environment variables to avoid getpwuid lookups
os.environ.setdefault("USER", "pytorch")
os.environ.setdefault("LOGNAME", "pytorch")
os.environ.setdefault("HOME", "/tmp")
import torch.optim as optim
import torch._dynamo
from torch.amp import GradScaler
from src.optimizer import SingleDeviceMuonWithAuxAdam
import torch._inductor.config
from src.datasets import SPEEDDataset, get_dataset_config
from src.engine import train_epoch, val_epoch, TrainingConfig, ValidationConfig
from src.losses import (
    QuaternionGeodesicLoss,
    QuaternionSimplifiedGeodesicLoss,
    QuaternionMSELoss,
    QuaternionFrobeniusLoss,
    RelativeQuaternionFrobeniusLoss,
    RotationMatrixGeodesicLoss,
    RotationMatrixMSELoss,
    RotationMatrixFrobeniusLoss,
    RotationMatrixMAELoss,
    RotationMatrixLpNormLoss,
    RotationMatrixNuclearLoss,
    RelativeRotationFrobeniusLoss,
    Rotation6DLoss,
    Rotation6DMAELoss,
    Rotation6DFrobeniusLoss,
    Rotation6DMSELoss,
    Rotation6DLpNormLoss,
    Rotation6DNuclearLoss,
    RelativeRotation6DFrobeniusLoss,
    TranslationMSELoss,
    TranslationMAELoss,
    TranslationFrobeniusLoss,
    RelativeTranslationLoss,
    RelativeTranslationFrobeniusLoss,
    TranslationRelativeMSELoss,
    TranslationLpNormLoss,
    TranslationNuclearLoss,
)
from src.models import FastPoseViT, SUPPORTED_VIT_MODELS, VIT_MODELS
import numpy as np
from styleaug import StyleAugmentor

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from src.utils import setup_logging
from src.scheduler import LogarithmicLR, CosineAnnealingLR, CosineAnnealingWithWarmRestarts


def train(args: argparse.Namespace, log_file: str, bbox_json_path: str) -> None:
    """
    Training function

    Args:
        args (argparse.Namespace): Command line arguments
        log_file (str): Path to the log file

    Returns:
        None
    """
    # Set up logging
    logger, _ = setup_logging(args, initialized=True, log_file=log_file)

    # Set up loss functions
    match args.translation_loss:
        case "mse":
            translation_criterion = TranslationMSELoss()
        case "mae":
            translation_criterion = TranslationMAELoss()
        case "frobenius":
            translation_criterion = TranslationFrobeniusLoss()
        case "relative_mse":
            translation_criterion = TranslationRelativeMSELoss()
        case "relative_frobenius":
            translation_criterion = RelativeTranslationFrobeniusLoss()
        case "lp_norm_075":
            translation_criterion = TranslationLpNormLoss(p=0.75)
        case "lp_norm_3":
            translation_criterion = TranslationLpNormLoss(p=3.0)
        case "lp_norm_inf":
            translation_criterion = TranslationLpNormLoss(p=float("inf"))
        case "nuclear":
            translation_criterion = TranslationNuclearLoss()
        case "relative_translation":
            translation_criterion = RelativeTranslationLoss()
        case _:
            raise ValueError(f"Invalid translation loss: {args.translation_loss}")

    if args.rotation_format == "quaternion":
        match args.rotation_loss:
            case "mse":
                rotation_criterion = QuaternionMSELoss()
            case "geodesic":
                rotation_criterion = QuaternionGeodesicLoss()
            case "simplified":
                rotation_criterion = QuaternionSimplifiedGeodesicLoss()
            case "frobenius":
                rotation_criterion = QuaternionFrobeniusLoss()
            case "relative_frobenius":
                rotation_criterion = RelativeQuaternionFrobeniusLoss()
            case _:
                raise ValueError(f"Invalid rotation loss for quaternions: {args.rotation_loss}")
        # Quaternion has 4 dimensions
        out_dim_rotation = 4
    elif args.rotation_format == "matrix":
        match args.rotation_loss:
            case "mse":
                rotation_criterion = RotationMatrixMSELoss()
            case "geodesic":
                rotation_criterion = RotationMatrixGeodesicLoss()
            case "frobenius":
                rotation_criterion = RotationMatrixFrobeniusLoss()
            case "mae":
                rotation_criterion = RotationMatrixMAELoss()
            case "lp_norm_075":
                rotation_criterion = RotationMatrixLpNormLoss(p=0.75)
            case "lp_norm_3":
                rotation_criterion = RotationMatrixLpNormLoss(p=3.0)
            case "lp_norm_inf":
                rotation_criterion = RotationMatrixLpNormLoss(p=float("inf"))
            case "nuclear":
                rotation_criterion = RotationMatrixNuclearLoss()
            case "6d":
                rotation_criterion = Rotation6DLoss()
            case "6d_mae":
                rotation_criterion = Rotation6DMAELoss()
            case "6d_frobenius":
                rotation_criterion = Rotation6DFrobeniusLoss()
            case "6d_mse":
                rotation_criterion = Rotation6DMSELoss()
            case "6d_lp_norm_075":
                rotation_criterion = Rotation6DLpNormLoss(p=0.75)
            case "6d_lp_norm_3":
                rotation_criterion = Rotation6DLpNormLoss(p=3.0)
            case "6d_lp_norm_inf":
                rotation_criterion = Rotation6DLpNormLoss(p=float("inf"))
            case "6d_nuclear":
                rotation_criterion = Rotation6DNuclearLoss()
            case "6d_relative_frobenius":
                rotation_criterion = RelativeRotation6DFrobeniusLoss()
            case "relative_frobenius":
                rotation_criterion = RelativeRotationFrobeniusLoss()
            case _:
                raise ValueError(
                    f"Invalid rotation loss for rotation matrix: {args.rotation_loss}"
                )

        # We only need the first 2 columns of the 3x3 matrix and determine the last column from the first 2 by cross product
        out_dim_rotation = 6
    else:
        raise ValueError(f"Invalid rotation format: {args.rotation_format}")

    # Set up device - always use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Enable cuDNN benchmark and optimizations for better performance
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True  # Enable TF32 for better performance
        torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for matrix operations
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
            True  # BF16 optimizations
        )
        torch.set_float32_matmul_precision("medium")  # Set medium precision for matmul

        # Additional CUDA optimizations
        os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # Async kernel launches (faster)

        # OpenMP settings for better performance
        os.environ["OMP_SCHEDULE"] = "STATIC"  # Static scheduling for OpenMP
        os.environ["OMP_PROC_BIND"] = "CLOSE"  # Bind threads to cores for better locality
        # 8 threads is a good default for most GPUs
        os.environ["OMP_NUM_THREADS"] = "8"

        # Synchronize CUDA context but avoid cache clearing for power stability
        torch.cuda.synchronize()

        # Set memory fraction to prevent fragmentation
        torch.cuda.set_per_process_memory_fraction(0.9)

        logger.info("cuDNN benchmark, TF32 and memory optimizations enabled")

    # Get the image size for the ViT model
    image_size = VIT_MODELS[args.vit_model][3] if args.vit_model in VIT_MODELS else (224, 224)

    # Initialize the model
    model = FastPoseViT(
        vit_model=args.vit_model,
        num_hidden_layers=args.num_hidden_layers,
        hidden_layer_dim=args.hidden_layer_dim,
        out_dim_translation=3,
        out_dim_rotation=out_dim_rotation,
        vit_weights=args.vit_weights,
        merge_outputs=args.merge_outputs,
        nb_class_tokens=args.nb_class_tokens,
        use_layer_norm=args.use_layer_norm,
        dropout_rate=args.dropout_rate,
        use_residual=args.use_residual,
        no_mlp=args.no_mlp,
    ).to(device)

    # Set basic torch inductor optimization flags (always applied for better performance)
    if device.type == "cuda":
        torch._inductor.config.triton.unique_kernel_names = True
        torch._inductor.config.coordinate_descent_tuning = True
        logger.info("Torch inductor optimization flags enabled")

    # Apply torch.compile unless disabled
    if not args.no_compile_model:
        logger.info("Applying torch.compile optimization...")
        logger.info("Note: First few steps will be slower due to compilation warmup")

        # Simple warmup for torch.compile
        if device.type == "cuda":
            logger.info("Performing simple model warmup for torch.compile...")

            model.eval()
            with torch.no_grad():
                dummy_input = torch.randn(args.batch_size, 3, *image_size, device=device)
                logger.info(f"Warmup input shape: {dummy_input.shape}")

                # Single warmup run
                try:
                    _ = model(dummy_input)
                    logger.info("Model warmup successful")
                except Exception as e:
                    logger.warning(f"Model warmup failed: {e}")

            model.train()
            logger.info("Warmup completed")

        # Use max-autotune-no-cudagraphs mode (compatible with gradient accumulation)
        try:
            # Enable error suppression for Docker environments
            torch._dynamo.config.suppress_errors = True

            model = torch.compile(model, mode="max-autotune")
            logger.info(
                "Model compiled with torch.compile (max-autotune mode + advanced optimizations)"
            )

            # Extended warmup for torch.compile to stabilize GPU power usage
            if device.type == "cuda":
                logger.info("Performing extended model warmup for torch.compile stability...")

                model.eval()
                with torch.no_grad():
                    # Multiple warmup scenarios to trigger all compilation paths
                    warmup_scenarios = [
                        (1, "single sample"),
                        (args.batch_size // 2, "half batch"),
                        (args.batch_size, "full batch"),
                    ]

                    for warmup_batch_size, desc in warmup_scenarios:
                        dummy_input = torch.randn(warmup_batch_size, 3, *image_size, device=device)
                        logger.info(f"Warming up with {desc}: {dummy_input.shape}")

                        # Multiple forward passes to ensure compilation stability
                        for i in range(3):
                            try:
                                _ = model(dummy_input)
                            except Exception as e:
                                logger.warning(f"Warmup iteration {i + 1} failed: {e}")

                    # Synchronize to ensure all kernels are compiled
                    torch.cuda.synchronize()
                    logger.info("Extended model warmup completed - GPU power should stabilize")

                model.train()
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

    # Create datasets and data loaders
    train_dataset = SPEEDDataset(
        dataset_root_dir=args.dataset_root_dir,
        split="train",
        rotation_format=args.rotation_format,
        img_size=image_size,
        bbox_json_path=bbox_json_path,
        args=args,
        dataset_name=args.dataset,
    )
    val_dataset = SPEEDDataset(
        dataset_root_dir=args.dataset_root_dir,
        split="val",
        rotation_format=args.rotation_format,
        img_size=image_size,
        bbox_json_path=bbox_json_path,
        args=args,
        dataset_name=args.dataset,
    )
    logger.info(
        f"Datasets created. Train set size: {len(train_dataset)}, Val set size: {len(val_dataset)}"
    )

    # Optimized DataLoader configuration
    # Use persistent workers if we have enough workers to benefit from it
    use_persistent_workers = args.num_workers > 1

    # Increase prefetch factor for better pipeline efficiency
    optimal_prefetch = min(4, max(2, args.num_workers))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",  # Only pin memory when using GPU
        persistent_workers=use_persistent_workers,
        prefetch_factor=optimal_prefetch,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",  # Only pin memory when using GPU
        persistent_workers=use_persistent_workers,
        prefetch_factor=optimal_prefetch,
    )
    logger.info(
        f"DataLoaders created with batch size {args.batch_size} and {args.num_workers} workers, prefetch={optimal_prefetch}, persistent={use_persistent_workers}"
    )

    # Calculate total iterations for per-iteration scheduling
    batches_per_epoch = len(train_loader)
    total_iterations = batches_per_epoch * args.epochs

    logger.info(f"Training setup: {batches_per_epoch} batches/epoch")
    logger.info(f"Total iterations across {args.epochs} epochs: {total_iterations}")

    # Initialize StyleAugmentor if needed
    style_augmentor = None
    if args.domain_gap_pixel_augmentation and args.do_style_aug:
        try:
            style_augmentor = StyleAugmentor().to(device)
        except Exception as e:
            logger.warning(f"Could not initialize StyleAugmentor: {e}")
            style_augmentor = None
        else:
            logger.info("StyleAugmentor initialized for domain gap augmentation")

    # Log model parameters using vectorized computation
    param_counts = torch.tensor([p.numel() for p in model.parameters()])
    trainable_mask = torch.tensor([p.requires_grad for p in model.parameters()])

    total_params = param_counts.sum().item()
    trainable_params = param_counts[trainable_mask].sum().item()
    logger.info(
        f"Model initialized with {total_params} total parameters, {trainable_params} trainable parameters"
    )

    # Load full model weights if provided
    if args.model_weights:
        model.load_from_pretrained(args.model_weights)
        logger.info(f"Loaded pretrained model weights from {args.model_weights}")

    # Set up TensorBoard writer
    if args.skip_tensorboard:
        writer = None
        logger.info("TensorBoard logging disabled")
    else:
        writer = SummaryWriter(args.log_dir)
        logger.info(f"TensorBoard writer initialized at {args.log_dir}")

    logger.info("Model ready for training")

    # Apply learning rate scaling based on batch size
    # Without gradient accumulation, use batch size directly
    effective_batch_size = args.batch_size
    base_batch_size = 4  # Reference batch size for the given learning rates
    lr_scale_factor = effective_batch_size / base_batch_size

    scaled_max_lr = args.max_lr * lr_scale_factor
    scaled_min_lr = args.min_lr * lr_scale_factor

    logger.info(f"Batch size: {args.batch_size}")
    logger.info(
        f"Learning rate scaling factor: {lr_scale_factor:.2f} (based on base batch size {base_batch_size})"
    )
    logger.info(f"Scaled learning rates: max={scaled_max_lr:.2e}, min={scaled_min_lr:.2e}")

    # Set up optimizer and scheduler
    if args.optimizer == "adam":
        # Use fused AdamW for better performance
        optimizer = optim.AdamW(
            model.parameters(),
            lr=scaled_max_lr,
            fused=True if device.type == "cuda" else False,  # Use fused kernel implementation
        )
    elif args.optimizer == "muon":
        # Separate parameters for Muon using vectorized filtering
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
                    "lr": scaled_max_lr,
                }
            )
        if adam_params:
            param_groups.append(
                {
                    "params": adam_params,
                    "use_muon": False,
                    "lr": scaled_max_lr * 0.1,
                    "betas": (0.9, 0.95),
                }
            )

        optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    # Choose between per-iteration and per-epoch scheduling
    # Per-epoch scheduling is more conservative and may help with convergence
    use_per_iteration_scheduling = not getattr(args, "use_per_epoch_scheduling", False)
    steps_per_epoch_for_scheduler = batches_per_epoch if use_per_iteration_scheduling else 1

    logger.info(
        f"Using {'per-iteration' if use_per_iteration_scheduling else 'per-epoch'} learning rate scheduling"
    )

    # Log scheduler configuration details for debugging
    if args.optimizer != "muon":
        scheduler_steps = total_iterations if use_per_iteration_scheduling else args.epochs
        logger.info(
            f"Scheduler will be stepped {scheduler_steps} times over {args.epochs} epochs ({batches_per_epoch} batches/epoch)"
        )

    # Initialize scheduler - will be used differently for Muon vs Adam
    if args.optimizer == "muon":
        # For Muon, we'll handle scheduling manually in the epoch loop
        scheduler = None
    else:
        if args.scheduler == "cosineannealinglr":
            scheduler = CosineAnnealingLR(
                optimizer,
                scaled_max_lr,
                scaled_min_lr,
                T_max_epochs=args.epochs,
                steps_per_epoch=steps_per_epoch_for_scheduler,
            )
        elif args.scheduler == "cosineannealingwarmrestarts":
            # T_0 is specified in epochs (user-friendly)
            T_0_epochs = 10  # Restart every 10 epochs
            scheduler = CosineAnnealingWithWarmRestarts(
                optimizer,
                scaled_max_lr,
                scaled_min_lr,
                T_0=T_0_epochs,
                T_mult=2,
                steps_per_epoch=steps_per_epoch_for_scheduler,
            )
        elif args.scheduler == "logarithmiclr":
            scheduler = LogarithmicLR(
                optimizer,
                scaled_max_lr,
                scaled_min_lr,
                T_max_epochs=args.epochs,
                steps_per_epoch=steps_per_epoch_for_scheduler,
            )
        else:
            raise ValueError(
                f"Unsupported scheduler: {args.scheduler}. Choose from ['cosineannealinglr', 'cosineannealingwarmrestarts', 'logarithmiclr']"
            )
    scheduler_name = f"Manual {args.scheduler}" if args.optimizer == "muon" else args.scheduler
    logger.info(
        f"{args.optimizer.capitalize()} optimizer with {scheduler_name} scheduler and criterion initialized"
    )

    # If log_dir does not exist, create it
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # If folder of model_save_path does not exist, create it
    if args.model_save_path:
        model_save_folder = os.path.dirname(args.model_save_path)
        if not os.path.exists(model_save_folder):
            os.makedirs(model_save_folder)

    # Initialize mixed precision scaler with power-stability optimized settings
    scaler = (
        GradScaler(
            device="cuda",
            init_scale=2.0**12,  # Lower initial scale for power stability
            growth_factor=1.5,  # Conservative growth to reduce power spikes
            backoff_factor=0.8,  # Conservative backoff
            growth_interval=1000,  # Less frequent scaling adjustments for stability
        )
        if device.type == "cuda"
        else None
    )
    if scaler:
        logger.info("Mixed precision training enabled with power-stability optimized settings")

    # Initialize best validation loss
    best_val_loss = float("inf")

    # Cache validation loss criteria to avoid recreation every epoch
    val_translation_criterion = RelativeTranslationLoss()
    val_rotation_criterion = QuaternionGeodesicLoss()

    # Cache parameter group indices for Muon optimizer to avoid dictionary lookups
    muon_param_groups = []
    adam_param_groups = []
    if args.optimizer == "muon":
        for idx, group in enumerate(optimizer.param_groups):
            if group.get("use_muon", False):
                muon_param_groups.append(idx)
            else:
                adam_param_groups.append(idx)

    # Pre-compute Muon scheduler values for performance (if using Muon)
    muon_lrs = None
    adam_lrs = None
    if args.optimizer == "muon":
        # Use the same scheduling approach as the Adam optimizer
        total_steps_for_muon = total_iterations if use_per_iteration_scheduling else args.epochs
        steps_per_epoch_for_muon = batches_per_epoch if use_per_iteration_scheduling else 1

        # Pre-compute constants outside the loop for efficiency
        if args.scheduler == "cosineannealingwarmrestarts":
            # Use the static method from CosineAnnealingWithWarmRestarts
            T_0_epochs = 10  # Restart every 10 epochs (user-friendly)
            muon_lrs = CosineAnnealingWithWarmRestarts.compute_all_lrs(
                scaled_max_lr,
                scaled_min_lr,
                total_steps_for_muon,
                T_0_epochs=T_0_epochs,
                T_mult=2,
                steps_per_epoch=steps_per_epoch_for_muon,
            )
        elif args.scheduler == "logarithmiclr":
            # Use the static method from LogarithmicLR
            muon_lrs = LogarithmicLR.compute_all_lrs(
                scaled_max_lr,
                scaled_min_lr,
                T_max_epochs=args.epochs,
                steps_per_epoch=steps_per_epoch_for_muon,
            )
        elif args.scheduler == "cosineannealinglr":
            # Use the static method from CosineAnnealingLR
            muon_lrs = CosineAnnealingLR.compute_all_lrs(
                scaled_max_lr,
                scaled_min_lr,
                T_max_epochs=args.epochs,
                steps_per_epoch=steps_per_epoch_for_muon,
            )
        else:
            raise ValueError(f"Unsupported scheduler for Muon: {args.scheduler}")

        # Compute Adam LRs in one vectorized operation
        adam_lrs = (np.array(muon_lrs) * 0.1).tolist()

        # Log Muon scheduler configuration
        logger.info(f"Muon pre-computed learning rates: {total_steps_for_muon} steps")

    # Initialize global iteration counter for proper learning rate scheduling
    global_iteration = 0

    # Training loop
    for epoch in range(args.epochs):
        # Train and validate
        # Create training configuration
        train_config = TrainingConfig(
            rotation_format=args.rotation_format,
            rotation_loss_type=args.rotation_loss,
            merge_outputs=args.merge_outputs,
            max_grad_norm=args.max_grad_norm,
            muon_lrs=muon_lrs if args.optimizer == "muon" else None,
            adam_lrs=adam_lrs if args.optimizer == "muon" else None,
            muon_param_groups=muon_param_groups if args.optimizer == "muon" else None,
            adam_param_groups=adam_param_groups if args.optimizer == "muon" else None,
            use_per_iteration_scheduling=use_per_iteration_scheduling,
            global_iteration_start=global_iteration,
        )

        train_loss, train_loss_translation, train_loss_rotation = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            translation_criterion=translation_criterion,
            rotation_criterion=rotation_criterion,
            device=device,
            epoch=epoch,
            writer=writer,
            config=train_config,
            scaler=scaler,
            style_augmentor=style_augmentor,
            train_dataset=train_dataset,
            scheduler=scheduler if args.optimizer != "muon" else None,
        )
        # Create validation configuration
        val_config = ValidationConfig(
            rotation_format=args.rotation_format,
            merge_outputs=args.merge_outputs,
            no_rotation_compensation=args.no_rotation_compensation,
            use_absolute_translation=args.use_absolute_translation,
        )

        val_loss, val_loss_translation, val_loss_rotation, val_metrics = val_epoch(
            model=model,
            val_loader=val_loader,
            translation_criterion=val_translation_criterion,
            rotation_criterion=val_rotation_criterion,
            device=device,
            epoch=epoch,
            writer=writer,
            config=val_config,
            val_dataset=val_dataset,
        )

        # Update global iteration counter for next epoch
        global_iteration += batches_per_epoch

        # Step scheduler for per-epoch mode (per-iteration stepping is handled in train_epoch)
        if not use_per_iteration_scheduling and scheduler is not None and args.optimizer != "muon":
            old_lr = (
                scheduler.get_last_lr()[0]
                if hasattr(scheduler, "get_last_lr")
                else optimizer.param_groups[0]["lr"]
            )
            scheduler.step()
            new_lr = (
                scheduler.get_last_lr()[0]
                if hasattr(scheduler, "get_last_lr")
                else optimizer.param_groups[0]["lr"]
            )
            logger.debug(
                f"Per-epoch scheduler stepped after epoch {epoch + 1}: LR {old_lr:.8f} -> {new_lr:.8f}"
            )

        # Log the results
        # Get learning rate information for logging
        if args.optimizer == "muon" and muon_lrs is not None:
            # For Muon with per-iteration scheduling, show start and end LR
            epoch_start_iteration = (epoch) * batches_per_epoch
            epoch_end_iteration = (epoch + 1) * batches_per_epoch - 1

            if use_per_iteration_scheduling:
                start_lr_index = min(epoch_start_iteration, len(muon_lrs) - 1)
                end_lr_index = min(epoch_end_iteration, len(muon_lrs) - 1)
                start_lr = muon_lrs[start_lr_index]
                end_lr = muon_lrs[end_lr_index]

                # Format LR display for per-iteration scheduling
                if start_lr != end_lr:
                    current_lr_display = f"{start_lr:.8f} → {end_lr:.8f}"
                else:
                    current_lr_display = f"{start_lr:.8f}"
                current_lr = start_lr  # For compatibility with TensorBoard logging
            else:
                lr_index = min(epoch, len(muon_lrs) - 1)
                current_lr = muon_lrs[lr_index]
                current_lr_display = f"{current_lr:.8f}"
        else:
            # For Adam, get the current learning rate from optimizer
            current_lr = optimizer.param_groups[0]["lr"]
            current_lr_display = f"{current_lr:.8f}"

        scheduling_mode = "per-iteration" if use_per_iteration_scheduling else "per-epoch"
        log_message = (
            f"Epoch {epoch + 1}/{args.epochs} ({scheduling_mode} scheduling)\n"
            f"Train Loss: {train_loss:.4f} (Translation: {train_loss_translation:.4f}, Rotation: {train_loss_rotation:.4f})\n"
            f"Val Loss: {val_loss:.4f} (Relative Translation: {val_loss_translation:.4f}, Rotation (Geodesic): {val_loss_rotation:.4f})\n"
            f"Val Metrics: Trans: {val_metrics['translation_metric_mean']:.4f}, Rot: {val_metrics['rotation_metric_mean']:.4f}, Rel Trans: {val_metrics['relative_translation_metric_mean']:.4f}, SPEED Score: {val_metrics['relative_total_metric']:.4f}\n"
            f"Learning Rate: {current_lr_display}"
        )
        logger.info(log_message)

        if writer is not None:
            writer.add_scalar("LearningRate", current_lr, epoch)

        # Save the model if it's the best so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if args.model_save_path:
                model.save(args.model_save_path)
                best_model_message = f"New best model saved to {args.model_save_path} with validation loss: {best_val_loss:.4f}"
                logger.info(best_model_message)

    # Training completed
    final_message = f"Training completed. Best validation loss: {best_val_loss:.4f}"
    logger.info(final_message)

    logger.info("Training process completed")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Train FastPoseViT on SPEED dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # General parameters
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
        help="Root directory of the dataset (auto-determined from dataset choice if not specified)",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for data loading"
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        default=None,
        help="Path to save the best model",
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
        "--vit_weights", type=str, default=None, help="Path to pretrained ViT weights"
    )
    parser.add_argument(
        "--model_weights", type=str, default=None, help="Path to pretrained full model weights"
    )

    # Model parameters
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs to train")
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for training and validation"
    )
    parser.add_argument(
        "--max_lr", type=float, default=1e-4, help="Maximum learning rate for cosine annealing"
    )
    parser.add_argument(
        "--min_lr", type=float, default=1e-6, help="Minimum learning rate for cosine annealing"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["adam", "muon"],
        default="adam",
        help="Optimizer to use for training",
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
        help="Learning rate scheduler to use (ignored for muon optimizer)",
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
        "--translation_loss",
        type=str,
        choices=[
            "mse",
            "mae",
            "frobenius",
            "relative_mse",
            "relative_frobenius",
            "lp_norm_075",
            "lp_norm_3",
            "lp_norm_inf",
            "nuclear",
            "relative_translation",
        ],
        default="mse",
        help="Loss function for translation",
    )
    parser.add_argument(
        "--rotation_format",
        type=str,
        choices=["quaternion", "matrix"],
        default="quaternion",
        help="Format for representing rotations",
    )
    parser.add_argument(
        "--rotation_loss",
        type=str,
        choices=[
            "mse",
            "geodesic",
            "simplified",
            "frobenius",
            "mae",
            "lp_norm_075",
            "lp_norm_3",
            "lp_norm_inf",
            "nuclear",
            "6d",
            "6d_mae",
            "6d_frobenius",
            "6d_mse",
            "6d_lp_norm_075",
            "6d_lp_norm_3",
            "6d_lp_norm_inf",
            "6d_nuclear",
            "6d_relative_frobenius",
            "relative_frobenius",
        ],
        default="mse",
        help="Loss function for rotation",
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
    # Don't use bbox augmentation
    parser.add_argument(
        "--no_bbox_augmentation",
        action="store_true",
        help="Don't use bbox augmentation during spatial augmentation",
    )
    # Bbox augmentation crop percentage
    parser.add_argument(
        "--bbox_crop_percent",
        type=float,
        default=10.0,
        help="Maximum crop percentage for bbox augmentation (0-50 percent, default: 10.0)",
    )
    # Don't pad cropped images to make them square
    parser.add_argument(
        "--no_crop_padding",
        action="store_true",
        help="Don't pad cropped images to make them square",
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

    # Training optimization arguments
    parser.add_argument(
        "--no_compile_model",
        action="store_true",
        help="Disable torch.compile optimization (enabled by default)",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=0.0,
        help="Maximum gradient norm for gradient clipping (0.0 disables clipping)",
    )
    parser.add_argument(
        "--use_per_epoch_scheduling",
        action="store_true",
        help="Use per-epoch learning rate scheduling instead of per-iteration (more conservative, may help convergence)",
    )

    args = parser.parse_args()

    # Validate bbox_crop_percent
    if not (0.0 <= args.bbox_crop_percent <= 50.0):
        parser.error("--bbox_crop_percent must be between 0 and 50 (inclusive)")

    # Set dataset_root_dir based on dataset choice if not specified
    if args.dataset_root_dir is None:
        dataset_config = get_dataset_config(args.dataset)
        args.dataset_root_dir = f"./{dataset_config['folder']}"

    # Automatically determine bbox_json_path based on dataset
    dataset_config = get_dataset_config(args.dataset)
    bbox_json_path = os.path.join(args.dataset_root_dir, dataset_config["bbox_annotations"])

    # Set up logging
    logger, log_file = setup_logging(args)

    # Log the arguments provided in a clean format
    logger.info("=========================================")
    logger.info("Arguments provided:")
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")
    logger.info("=========================================")
    logger.info("")

    # Start training process
    logger.info("Starting training")
    train(args, log_file, bbox_json_path)
