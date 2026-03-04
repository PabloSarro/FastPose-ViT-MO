# This file contains the training and validation loops for the model.

import logging
from collections import deque
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from src.utils import (
    smooth_loss,
    process_rotation,
    bbox_relative_translation_to_translation,
    rotation_matrix_to_quaternion,
    get_absolute_orientation,
    rotation_matrix_to_6d,
)
from src.metrics import compute_metrics


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""

    rotation_format: str
    rotation_loss_type: Optional[str] = None
    smooth_window: int = 500
    merge_outputs: bool = False
    max_grad_norm: float = 1.0
    muon_lrs: Optional[list[float]] = None
    adam_lrs: Optional[list[float]] = None
    muon_param_groups: Optional[list[int]] = None
    adam_param_groups: Optional[list[int]] = None
    use_per_iteration_scheduling: bool = True
    global_iteration_start: int = 0


@dataclass
class ValidationConfig:
    """Configuration for validation parameters."""

    rotation_format: str
    smooth_window: int = 500
    merge_outputs: bool = False
    no_rotation_compensation: bool = False
    use_absolute_translation: bool = False


class ProgressTracker:
    """Handles progress tracking and smoothed loss display."""

    def __init__(self, smooth_window: int, desc: str) -> None:
        self.loss_deque: deque[float] = deque(maxlen=smooth_window)
        self.loss_translation_deque: deque[float] = deque(maxlen=smooth_window)
        self.loss_rotation_deque: deque[float] = deque(maxlen=smooth_window)
        self.pbar: tqdm | None = None
        self.desc = desc

    def setup_progress_bar(self, total_batches: int) -> tqdm:
        """Set up progress bar for the epoch.

        Args:
            total_batches: Total number of batches in the epoch.

        Returns:
            The initialized tqdm progress bar.
        """
        self.pbar = tqdm(total=total_batches, desc=self.desc)
        return self.pbar

    def update_losses(
        self, total_loss: float, trans_loss: float, rot_loss: float, batch_idx: int
    ) -> None:
        """Update loss tracking and display every 10 batches.

        Args:
            total_loss: Total loss value for the batch.
            trans_loss: Translation loss value for the batch.
            rot_loss: Rotation loss value for the batch.
            batch_idx: Current batch index.
        """
        update_interval = 10
        if batch_idx % update_interval == 0:
            self.loss_deque.append(total_loss)
            self.loss_translation_deque.append(trans_loss)
            self.loss_rotation_deque.append(rot_loss)

            smoothed_loss = smooth_loss(self.loss_deque)
            smoothed_loss_translation = smooth_loss(self.loss_translation_deque)
            smoothed_loss_rotation = smooth_loss(self.loss_rotation_deque)

            self.pbar.set_postfix(
                {
                    "loss": f"{smoothed_loss:.4f}",
                    "trans_loss": f"{smoothed_loss_translation:.4f}",
                    "rot_loss": f"{smoothed_loss_rotation:.4f}",
                }
            )

            self.pbar.update(update_interval)

    def close(self) -> None:
        """Close the progress bar."""
        if self.pbar:
            self.pbar.close()


class LossAccumulator:
    """Efficiently accumulates losses with minimal GPU-CPU transfers."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.total_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        self.translation_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        self.rotation_loss = torch.tensor(0.0, device=device, dtype=torch.float32)

    def add_batch_losses(
        self, total: torch.Tensor, translation: torch.Tensor, rotation: torch.Tensor
    ) -> None:
        """Add losses from a batch keeping everything on GPU.

        Args:
            total: Total loss tensor for the batch.
            translation: Translation loss tensor for the batch.
            rotation: Rotation loss tensor for the batch.
        """
        self.total_loss.add_(total.detach())
        self.translation_loss.add_(translation.detach())
        self.rotation_loss.add_(rotation.detach())

    def get_averages(self, num_batches: int) -> tuple[float, float, float]:
        """Get average losses across all accumulated batches.

        Args:
            num_batches: Number of batches accumulated.

        Returns:
            Tuple of (average_total_loss, average_translation_loss, average_rotation_loss).
        """
        inv_batches = torch.tensor(1.0 / num_batches, device=self.device, dtype=torch.float32)

        # Batch the final GPU-CPU transfer
        loss_tensor = torch.stack(
            [
                self.total_loss * inv_batches,
                self.translation_loss * inv_batches,
                self.rotation_loss * inv_batches,
            ]
        ).cpu()

        return loss_tensor[0].item(), loss_tensor[1].item(), loss_tensor[2].item()


class CoordinateTransformer:
    """Handles coordinate space transformations for validation."""

    def __init__(
        self,
        rotation_format: str,
        val_camera,
        no_rotation_compensation: bool,
        use_absolute_translation: bool = False,
    ) -> None:
        self.rotation_format = rotation_format
        self.val_camera = val_camera
        self.no_rotation_compensation = no_rotation_compensation
        self.use_absolute_translation = use_absolute_translation
        self.is_matrix_format = rotation_format == "matrix"

    def transform_to_absolute(
        self,
        pred_translations: torch.Tensor,
        pred_rotations: torch.Tensor,
        translations: torch.Tensor,
        rotations: torch.Tensor,
        bboxes: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Transform predictions and targets to absolute coordinate space.

        Args:
            pred_translations: Predicted translation vectors.
            pred_rotations: Predicted rotation tensors.
            translations: Ground truth translation vectors.
            rotations: Ground truth rotation tensors.
            bboxes: Bounding box tensors or None.

        Returns:
            Tuple of (abs_pred_translations, abs_pred_rotations, abs_translations, abs_rotations).
        """
        device = pred_translations.device
        device_bboxes = bboxes.to(device, non_blocking=True) if bboxes is not None else None

        # Check if transformations are needed
        has_bbox = device_bboxes is not None and not torch.all(device_bboxes.eq(0))
        needs_rotation_transform = self.is_matrix_format or not self.no_rotation_compensation
        needs_translation_transform = has_bbox and not self.use_absolute_translation

        # Only clone if we actually need to modify the tensors
        if needs_translation_transform:
            abs_pred_translations = pred_translations.clone()
            abs_translations = translations.clone()
        else:
            abs_pred_translations = pred_translations
            abs_translations = translations

        if needs_rotation_transform:
            abs_pred_rotations = pred_rotations.clone()
            abs_rotations = rotations.clone()
        else:
            abs_pred_rotations = pred_rotations
            abs_rotations = rotations

        # Convert matrix rotations to quaternions
        if self.is_matrix_format:
            abs_pred_rotations = rotation_matrix_to_quaternion(abs_pred_rotations)
            abs_rotations = rotation_matrix_to_quaternion(abs_rotations)

        # Convert relative translations to absolute
        if has_bbox:
            abs_pred_translations = bbox_relative_translation_to_translation(
                abs_pred_translations,
                device_bboxes,
                self.val_camera,
            )
            abs_translations = bbox_relative_translation_to_translation(
                abs_translations,
                device_bboxes,
                self.val_camera,
            )

        # Apply rotation compensation
        if not self.no_rotation_compensation:
            abs_pred_rotations = get_absolute_orientation(
                translation=abs_pred_translations,
                rotation_quat=abs_pred_rotations,
                no_rotation_compensation=self.no_rotation_compensation,
            )
            abs_rotations = get_absolute_orientation(
                translation=abs_translations,
                rotation_quat=abs_rotations,
                no_rotation_compensation=self.no_rotation_compensation,
            )

        return abs_pred_translations, abs_pred_rotations, abs_translations, abs_rotations


def apply_style_augmentation(
    images: torch.Tensor,
    style_augmentor,
    train_dataset,
    device: torch.device,
) -> torch.Tensor:
    """Apply style augmentation to images if configured.

    Args:
        images: Batch of input images tensor.
        style_augmentor: StyleAugmentor instance or None.
        train_dataset: Training dataset with style augmentation methods.
        device: Target device for computation.

    Returns:
        Processed images tensor on the target device.
    """
    if (
        style_augmentor is not None
        and train_dataset is not None
        and hasattr(train_dataset, "domain_gap_pixel_augmentation")
        and train_dataset.domain_gap_pixel_augmentation
        and train_dataset.do_style_aug
    ):
        try:
            images = images.to(device, non_blocking=True)
            styled_images = style_augmentor(images, alpha=0.5)
            return train_dataset.finalize_style_augmented_batch(styled_images)
        except Exception as e:
            logging.warning(f"Style augmentation failed: {e}")
            return train_dataset.finalize_style_augmented_batch(
                images.to(device, non_blocking=True)
            )

    return images.to(device, non_blocking=True)


def update_learning_rates(
    optimizer: optim.Optimizer,
    current_iteration: int,
    muon_lrs: list[float] | None,
    adam_lrs: list[float] | None,
    muon_param_groups: list[int] | None,
    adam_param_groups: list[int] | None,
    scheduler,
    use_per_iteration_scheduling: bool = True,
    epoch: int = 0,
) -> None:
    """Update learning rates for optimizers.

    Args:
        optimizer: The optimizer to update.
        current_iteration: Current training iteration.
        muon_lrs: Pre-computed learning rates for Muon parameters.
        adam_lrs: Pre-computed learning rates for Adam parameters.
        muon_param_groups: Indices of Muon parameter groups.
        adam_param_groups: Indices of Adam parameter groups.
        scheduler: Learning rate scheduler or None.
        use_per_iteration_scheduling: Whether to step per iteration.
        epoch: Current epoch number.
    """
    if muon_lrs is not None:
        # Choose index based on scheduling mode
        if use_per_iteration_scheduling:
            lr_index = min(current_iteration, len(muon_lrs) - 1)
        else:
            # For per-epoch scheduling, use epoch as index
            lr_index = min(epoch, len(muon_lrs) - 1)

        muon_lr = muon_lrs[lr_index]
        adam_lr = adam_lrs[lr_index] if adam_lrs else muon_lr * 0.1

        if muon_param_groups:
            for idx in muon_param_groups:
                optimizer.param_groups[idx]["lr"] = muon_lr
        if adam_param_groups:
            for idx in adam_param_groups:
                optimizer.param_groups[idx]["lr"] = adam_lr
    elif scheduler is not None and use_per_iteration_scheduling:
        # Only step scheduler per-iteration if explicitly using per-iteration scheduling
        scheduler.step()


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    translation_criterion: nn.Module,
    rotation_criterion: nn.Module,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    config: TrainingConfig,
    scaler: GradScaler = None,
    style_augmentor: nn.Module = None,
    train_dataset=None,
    scheduler=None,
) -> tuple[float, float, float]:
    """
    Training loop with simplified interface using configuration objects.

    Args:
        model: Model to train
        train_loader: Training data loader
        optimizer: Optimizer to use for training
        translation_criterion: Translation loss criterion
        rotation_criterion: Rotation loss criterion
        device: Device to use for training
        epoch: Current epoch
        writer: Tensorboard writer
        config: Training configuration object
        scaler: Mixed precision gradient scaler
        style_augmentor: StyleAugmentor for neural style transfer
        train_dataset: Training dataset for finalizing style-augmented images
        scheduler: Learning rate scheduler for per-iteration stepping

    Returns:
        tuple[float, float, float]: Total loss, translation loss, rotation loss
    """
    model.train()

    # Initialize helper objects
    loss_accumulator = LossAccumulator(device)
    progress_tracker = ProgressTracker(config.smooth_window, f"Epoch {epoch + 1} [Train]")
    total_batches = len(train_loader)
    progress_tracker.setup_progress_bar(total_batches)

    # Use global iteration counter from config
    current_iteration = config.global_iteration_start
    is_6d_rotation_loss = config.rotation_loss_type and "6d" in config.rotation_loss_type

    for batch_idx, (images, translations, rotations, bboxes) in enumerate(train_loader):
        # Apply style augmentation
        images = apply_style_augmentation(images, style_augmentor, train_dataset, device)

        # Move data to device
        translations = translations.to(device, non_blocking=True)
        rotations = rotations.to(device, non_blocking=True)

        # Forward pass with conditional autocast
        use_amp = scaler is not None
        with autocast("cuda", enabled=use_amp):
            if config.merge_outputs:
                output = model(images)
                pred_translations = output[:, :3]
                pred_rotations = output[:, 3:]
            else:
                pred_translations, pred_rotations = model(images)

            # Process rotations based on loss type
            if is_6d_rotation_loss:
                target_rotations_6d = rotation_matrix_to_6d(rotations)
                loss_rotation = rotation_criterion(pred_rotations, target_rotations_6d)
            else:
                pred_rotations = process_rotation(pred_rotations, config.rotation_format)
                loss_rotation = rotation_criterion(pred_rotations, rotations)

            loss_translation = translation_criterion(pred_translations, translations, bboxes)
            total_loss = loss_translation + loss_rotation

        # Backward pass
        if use_amp:
            scaler.scale(total_loss).backward()
            if config.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            if config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)

        # Update learning rates
        update_learning_rates(
            optimizer,
            current_iteration,
            config.muon_lrs,
            config.adam_lrs,
            config.muon_param_groups,
            config.adam_param_groups,
            scheduler,
            config.use_per_iteration_scheduling,
            epoch,
        )
        current_iteration += 1

        # Update tracking
        loss_accumulator.add_batch_losses(total_loss, loss_translation, loss_rotation)

        # Update progress and loss display every 10 batches
        progress_tracker.update_losses(
            total_loss.item(), loss_translation.item(), loss_rotation.item(), batch_idx
        )

    progress_tracker.close()

    # Get final averages
    train_loss, train_loss_translation, train_loss_rotation = loss_accumulator.get_averages(
        total_batches
    )

    # Log results
    if writer is not None:
        writer.add_scalar("Loss/Train/Total", train_loss, epoch)
        writer.add_scalar("Loss/Train/Translation", train_loss_translation, epoch)
        writer.add_scalar("Loss/Train/Rotation", train_loss_rotation, epoch)

    return train_loss, train_loss_translation, train_loss_rotation


def val_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    translation_criterion: nn.Module,
    rotation_criterion: nn.Module,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    config: ValidationConfig,
    val_dataset=None,
) -> tuple[float, float, float, dict]:
    """
    Validation loop with simplified interface using configuration objects.

    Args:
        model: Model to validate
        val_loader: Validation data loader
        translation_criterion: Translation loss criterion
        rotation_criterion: Rotation loss criterion
        device: Device to use for validation
        epoch: Current epoch
        writer: Tensorboard writer
        config: Validation configuration object
        val_dataset: Validation dataset for camera parameters

    Returns:
        tuple[float, float, float, dict]: Total loss, translation loss, rotation loss, validation metrics
    """
    model.eval()

    # Initialize helper objects
    val_camera = val_dataset.camera if val_dataset else None
    coordinate_transformer = CoordinateTransformer(
        config.rotation_format,
        val_camera,
        config.no_rotation_compensation,
        config.use_absolute_translation,
    )
    progress_tracker = ProgressTracker(config.smooth_window, f"Epoch {epoch + 1} [Val]")
    total_val_batches = len(val_loader)
    progress_tracker.setup_progress_bar(total_val_batches)

    # Initialize accumulators
    val_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
    val_loss_translation = torch.tensor(0.0, device=device, dtype=torch.float32)
    val_loss_rotation = torch.tensor(0.0, device=device, dtype=torch.float32)

    val_metrics = {
        "translation_metric_mean": 0.0,
        "rotation_metric_mean": 0.0,
        "relative_translation_metric_mean": 0.0,
        "relative_total_metric": 0.0,
    }

    inv_total_batches_tensor = torch.tensor(
        1.0 / total_val_batches, device=device, dtype=torch.float32
    )

    with torch.no_grad():
        for batch_idx, (images, translations, rotations, bboxes) in enumerate(val_loader):
            # Move data to device
            images = images.to(device, non_blocking=True)
            translations = translations.to(device, non_blocking=True)
            rotations = rotations.to(device, non_blocking=True)

            # Forward pass with conditional autocast
            with autocast("cuda", enabled=device.type == "cuda"):
                if config.merge_outputs:
                    output = model(images)
                    pred_translations = output[:, :3]
                    pred_rotations = output[:, 3:]
                else:
                    pred_translations, pred_rotations = model(images)

                pred_rotations = process_rotation(pred_rotations, config.rotation_format)

            # Transform to absolute coordinate space
            abs_pred_translations, abs_pred_rotations, abs_translations, abs_rotations = (
                coordinate_transformer.transform_to_absolute(
                    pred_translations, pred_rotations, translations, rotations, bboxes
                )
            )

            # Calculate loss and metrics on absolute coordinates
            loss_translation = translation_criterion(abs_pred_translations, abs_translations, None)
            loss_rotation = rotation_criterion(abs_pred_rotations, abs_rotations)
            total_loss = loss_translation + loss_rotation

            metrics = compute_metrics(
                abs_pred_translations, abs_translations, abs_pred_rotations, abs_rotations
            )

            # Accumulate losses
            val_loss.add_(total_loss.detach())
            val_loss_translation.add_(loss_translation.detach())
            val_loss_rotation.add_(loss_rotation.detach())

            # Accumulate metrics
            for key in val_metrics:
                val_metrics[key] += metrics.get(key, 0.0)

            # Update progress and loss display every 10 batches
            progress_tracker.update_losses(
                total_loss.item(), loss_translation.item(), loss_rotation.item(), batch_idx
            )

    progress_tracker.close()

    # Calculate final averages
    loss_tensor = torch.stack(
        [
            val_loss * inv_total_batches_tensor,
            val_loss_translation * inv_total_batches_tensor,
            val_loss_rotation * inv_total_batches_tensor,
        ]
    )
    loss_cpu = loss_tensor.cpu()
    val_loss = loss_cpu[0].item()
    val_loss_translation = loss_cpu[1].item()
    val_loss_rotation = loss_cpu[2].item()

    inv_total_batches_scalar = inv_total_batches_tensor.item()
    for key in val_metrics:
        val_metrics[key] *= inv_total_batches_scalar

    # Log results
    if writer is not None:
        writer.add_scalar("Loss/Validation/Total", val_loss, epoch)
        writer.add_scalar("Loss/Validation/Translation", val_loss_translation, epoch)
        writer.add_scalar("Loss/Validation/Rotation", val_loss_rotation, epoch)

    return val_loss, val_loss_translation, val_loss_rotation, val_metrics
