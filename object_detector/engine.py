# This file contains the engine for training and validation of the object detection model

from dataclasses import dataclass
from typing import List, Optional, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import torch.nn.functional as F

# For inference-style validation loss

# LW-DETR uses set criterion for loss calculation
from object_detector.metrics import compute_detection_metrics
from object_detector.constants import DEFAULT_UPDATE_INTERVAL
from object_detector.utils import prepare_model_input
from object_detector.utils import box_cxcywh_to_xyxy, box_iou


class ProgressTracker:
    """Simple progress tracker that updates at reasonable intervals.

    Args:
        total_batches: Total number of batches to process.
        desc: Description string for the progress bar.
        update_interval: How often to update the display.
    """

    def __init__(
        self, total_batches: int, desc: str, update_interval: int = DEFAULT_UPDATE_INTERVAL
    ) -> None:
        self.total_batches = total_batches
        self.update_interval = update_interval
        self.pbar = tqdm(total=total_batches, desc=desc)
        self.last_update = 0

    def update(self, batch_idx: int, loss: float) -> None:
        """Update progress bar at specified intervals.

        Args:
            batch_idx: Current batch index.
            loss: Current loss value to display.
        """
        if batch_idx % self.update_interval == 0 or batch_idx == self.total_batches - 1:
            # Update by the number of batches since last update
            update_amount = (
                batch_idx - self.last_update + (1 if batch_idx == self.total_batches - 1 else 0)
            )
            self.pbar.update(update_amount)
            self.pbar.set_postfix({"loss": f"{loss:.4f}"})
            self.last_update = batch_idx

    def close(self) -> None:
        """Close the progress bar."""
        self.pbar.close()


@dataclass
class TrainingConfig:
    """Training configuration."""

    max_grad_norm: float = 0.0
    muon_lrs: Optional[List[float]] = None
    adam_lrs: Optional[List[float]] = None
    muon_param_groups: Optional[List[int]] = None
    adam_param_groups: Optional[List[int]] = None
    use_per_iteration_scheduling: bool = True
    global_iteration_start: int = 0


def apply_style_augmentation(
    images: torch.Tensor,
    style_augmentor: Optional[nn.Module],
    train_dataset: Any,
    device: torch.device,
) -> torch.Tensor:
    """Apply style augmentation to images if configured.

    Args:
        images: Batch of input images.
        style_augmentor: Optional style augmentation module.
        train_dataset: Training dataset instance with augmentation settings.
        device: Device to move images to.

    Returns:
        Augmented (or original) images on the target device.
    """
    images = images.to(device, non_blocking=True)

    if (
        style_augmentor
        and train_dataset
        and getattr(train_dataset, "domain_gap_pixel_augmentation", False)
        and getattr(train_dataset, "do_style_aug", False)
    ):
        try:
            styled_images = style_augmentor(images, alpha=0.5)
            return train_dataset.finalize_style_augmented_batch(styled_images)
        except Exception:
            return train_dataset.finalize_style_augmented_batch(images)

    return images


def update_learning_rates(
    optimizer: torch.optim.Optimizer,
    current_iteration: int,
    config: "TrainingConfig",
    scheduler: Optional[Any],
    epoch: int = 0,
) -> None:
    """Update learning rates for optimizers.

    Args:
        optimizer: The optimizer whose learning rates to update.
        current_iteration: Current training iteration count.
        config: Training configuration with LR schedules.
        scheduler: Optional LR scheduler for non-Muon optimizers.
        epoch: Current epoch number.
    """
    if config.muon_lrs is not None:
        lr_index = min(
            current_iteration if config.use_per_iteration_scheduling else epoch,
            len(config.muon_lrs) - 1,
        )

        muon_lr = config.muon_lrs[lr_index]
        adam_lr = config.adam_lrs[lr_index] if config.adam_lrs else muon_lr * 0.1

        for idx in config.muon_param_groups or []:
            optimizer.param_groups[idx]["lr"] = muon_lr
        for idx in config.adam_param_groups or []:
            optimizer.param_groups[idx]["lr"] = adam_lr
    elif scheduler is not None and config.use_per_iteration_scheduling:
        scheduler.step()


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    writer: Optional[SummaryWriter] = None,
    config: Optional[TrainingConfig] = None,
    scaler: Optional[GradScaler] = None,
    style_augmentor: Optional[nn.Module] = None,
    train_dataset: Any = None,
    scheduler: Any = None,
) -> tuple[float, dict[str, float]]:
    """Training loop for one epoch.

    Args:
        model: The neural network model to train.
        train_loader: DataLoader providing training batches.
        optimizer: Optimizer for parameter updates.
        criterion: Loss criterion (LW-DETR SetCriterion or similar).
        device: Device to run training on.
        epoch: Current epoch number.
        writer: Optional TensorBoard SummaryWriter for logging.
        config: Optional training configuration.
        scaler: Optional GradScaler for mixed precision training.
        style_augmentor: Optional style augmentation module.
        train_dataset: Optional dataset reference for style augmentation.
        scheduler: Optional learning rate scheduler.

    Returns:
        Tuple of (average_loss, empty_dict).
    """
    model.train()

    if config is None:
        config = TrainingConfig()

    total_loss = 0.0
    num_batches = 0
    current_iteration = config.global_iteration_start

    progress = ProgressTracker(len(train_loader), f"Epoch {epoch + 1} [Train]")

    for batch_idx, (images, bboxes) in enumerate(train_loader):
        # Apply style augmentation and move to device
        images = apply_style_augmentation(images, style_augmentor, train_dataset, device)
        bboxes = bboxes.to(device, non_blocking=True)

        # Forward pass with proper input format (NestedTensor for LW-DETR)
        model_inputs = prepare_model_input(images, model)

        use_amp = scaler is not None
        with autocast("cuda", enabled=use_amp):
            outputs = model(model_inputs)

            # Convert targets to appropriate format for LW-DETR loss
            if hasattr(criterion, "__class__") and ("SetCriterion" in str(criterion.__class__)):
                # Original DETR-style loss expects list of dicts with 'labels' and 'boxes'
                targets = []
                for i in range(len(bboxes)):
                    targets.append(
                        {
                            "labels": torch.tensor(
                                [0], dtype=torch.long, device=bboxes.device
                            ),  # single object class index = 0
                            "boxes": bboxes[i : i + 1],  # [1, 4] bbox in cxcywh format
                        }
                    )
                loss_dict = criterion(outputs, targets)
                # DETR loss returns dict with different keys

                loss = sum(v for k, v in loss_dict.items() if k.startswith("loss_"))
                loss_dict["total_loss"] = loss

            else:
                # Simple loss for single-query models
                loss_dict = criterion(outputs, bboxes)
                loss = loss_dict["total_loss"]

        # Backward pass
        if use_amp:
            scaler.scale(loss).backward()
            if config.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)

        # Update learning rates
        update_learning_rates(optimizer, current_iteration, config, scheduler, epoch)
        current_iteration += 1

        # Accumulate loss
        total_loss += loss.item()
        num_batches += 1

        # Update progress tracker
        avg_loss = total_loss / num_batches
        progress.update(batch_idx, avg_loss)

    progress.close()

    avg_total_loss = total_loss / num_batches if num_batches > 0 else 0.0

    # Log loss components to TensorBoard if using Hungarian loss
    if writer is not None:
        writer.add_scalar("Loss/Train/Total", avg_total_loss, epoch)

        # Log loss components for detailed monitoring
        if hasattr(criterion, "__class__") and "SetCriterion" in str(criterion.__class__):
            if "loss_bbox" in loss_dict:
                writer.add_scalar("Loss/Train/BBox", loss_dict["loss_bbox"].item(), epoch)
            if "loss_ce" in loss_dict:
                writer.add_scalar("Loss/Train/CE", loss_dict["loss_ce"].item(), epoch)

    return avg_total_loss, {}


def val_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    writer: Optional[SummaryWriter] = None,
    config: Optional[TrainingConfig] = None,
    compute_ap_metrics: bool = True,
    inference_style_loss: bool = False,
) -> tuple[float, dict[str, float]]:
    """Validation loop for one epoch.

    Args:
        model: The neural network model to validate.
        val_loader: DataLoader providing validation batches.
        criterion: Loss criterion (LW-DETR SetCriterion or similar).
        device: Device to run validation on.
        epoch: Current epoch number.
        writer: Optional TensorBoard SummaryWriter for logging.
        config: Optional training configuration.
        compute_ap_metrics: Whether to compute AP detection metrics.
        inference_style_loss: Whether to use inference-style loss computation.

    Returns:
        Tuple of (average_loss, metrics_dict) where metrics_dict contains AP scores.
    """
    model.eval()

    total_loss = 0.0
    num_batches = 0
    all_outputs = []
    all_targets = []

    progress = ProgressTracker(len(val_loader), f"Epoch {epoch + 1} [Val]")

    with torch.no_grad():
        for batch_idx, (images, bboxes) in enumerate(val_loader):
            images = images.to(device, non_blocking=True)
            bboxes = bboxes.to(device, non_blocking=True)

            # Forward pass with proper input format (NestedTensor for LW-DETR)
            model_inputs = prepare_model_input(images, model)

            with autocast("cuda", enabled=device.type == "cuda"):
                outputs = model(model_inputs)

            # Store for AP metrics - always use inference-style behavior
            if compute_ap_metrics:
                # Always extract single best prediction per image (inference style)
                if "pred_logits" in outputs and "pred_boxes" in outputs:
                    pred_logits = outputs["pred_logits"]  # [B, Q, C] or [B, 1, C]
                    pred_boxes = outputs["pred_boxes"]  # [B, Q, 4] or [B, 1, 4]

                    # If model is returning all queries, select the best one per image
                    if pred_logits.shape[1] > 1:
                        # Satellite class score is class 1 probability via sigmoid (LW-DETR convention)
                        if pred_logits.shape[-1] > 1:
                            probs = torch.sigmoid(pred_logits[..., 1])  # [B, Q]
                        else:
                            probs = torch.sigmoid(pred_logits[..., 0])
                        # Select single best query per image
                        best_query_idx = torch.argmax(probs, dim=1)  # [B]
                        batch_indices = torch.arange(pred_boxes.shape[0], device=pred_boxes.device)
                        best_boxes = pred_boxes[batch_indices, best_query_idx]  # [B, 4]
                        best_scores = probs[batch_indices, best_query_idx]  # [B]

                        batch_outputs = {
                            "boxes": best_boxes.unsqueeze(1),  # [B, 1, 4]
                            "scores": best_scores.unsqueeze(1),  # [B, 1]
                        }
                    else:
                        # Model already returning single prediction
                        batch_outputs = {
                            "boxes": pred_boxes.clone(),
                            "scores": torch.sigmoid(pred_logits[..., 0])
                            if pred_logits.shape[-1] == 1
                            else torch.sigmoid(pred_logits[..., 1]),
                        }
                else:
                    # Direct single-prediction path
                    batch_outputs = {
                        "boxes": outputs["boxes"].clone(),
                        "scores": outputs["scores"].clone(),
                    }
                batch_targets = {
                    "boxes": (bboxes.unsqueeze(1) if bboxes.dim() == 2 else bboxes).clone()
                }
                all_outputs.append(batch_outputs)
                all_targets.append(batch_targets)

            # Compute validation loss - always use inference-style behavior
            # Inference-style loss: use single best prediction (no Hungarian)
            if "pred_logits" in outputs and "pred_boxes" in outputs:
                pred_logits = outputs["pred_logits"]  # [B, Q, C] or [B, 1, C]
                pred_boxes = outputs["pred_boxes"]  # [B, Q, 4] or [B, 1, 4]

                # If model is returning all queries, select the best one per image
                if pred_logits.shape[1] > 1:
                    # Satellite prob via sigmoid
                    if pred_logits.shape[-1] > 1:
                        probs = torch.sigmoid(pred_logits[..., 1])  # [B, Q]
                        sat_logits = pred_logits[..., 1]
                    else:
                        probs = torch.sigmoid(pred_logits[..., 0])
                        sat_logits = pred_logits[..., 0]
                    # Select single best query per image
                    best_query_idx = torch.argmax(probs, dim=1)  # [B]
                    batch_indices = torch.arange(pred_boxes.shape[0], device=pred_boxes.device)
                    best_boxes = pred_boxes[batch_indices, best_query_idx]  # [B, 4]
                    best_logits = sat_logits[batch_indices, best_query_idx]  # [B]
                else:
                    # Model already returning single prediction
                    best_boxes = pred_boxes.squeeze(1)  # [B, 4]
                    best_logits = pred_logits.squeeze(1)  # [B, C]
                    if best_logits.dim() == 1:
                        best_logits = best_logits.unsqueeze(0)  # Handle single image case
                    if best_logits.shape[-1] > 1:
                        best_logits = best_logits[:, 1]  # Satellite class
                    else:
                        best_logits = best_logits[:, 0]  # Satellite class

                # Compute BCE over single best predictions
                sat_prob = torch.sigmoid(best_logits)  # [B]
                ce_loss = F.binary_cross_entropy(
                    sat_prob, torch.ones_like(sat_prob), reduction="none"
                )  # [B]

                # IoU for single predictions
                best_boxes = best_boxes.clamp(0.0, 1.0)
                bboxes_clamped = bboxes.clamp(0.0, 1.0)
                pred_xyxy = box_cxcywh_to_xyxy(best_boxes)
                target_xyxy = box_cxcywh_to_xyxy(bboxes_clamped)
                # Use simple IoU loss (1 - IoU)
                iou_matrix = box_iou(pred_xyxy, target_xyxy)  # [B, B]
                iou_vec = torch.diag(iou_matrix)  # [B] - IoU for each sample
                iou_loss_vec = 1 - iou_vec  # [B] - IoU loss for each sample

                # Combine losses for single predictions
                total_loss_vector = 2.0 * ce_loss + 5.0 * iou_loss_vec
                loss = total_loss_vector.mean()
                loss_dict = {"total_loss": loss}
            else:
                # Fallback: treat as single best prediction
                boxes = outputs.get("boxes", outputs.get("pred_boxes"))  # [B, 1, 4]
                scores = outputs.get("scores")  # [B, 1]
                if boxes is None or scores is None:
                    # As last resort, use criterion path
                    loss_dict = criterion(outputs, bboxes)
                    loss = loss_dict.get("total_loss", next(iter(loss_dict.values())))
                    total_loss += loss.item()
                    num_batches += 1
                    avg_loss = total_loss / num_batches
                    progress.update(batch_idx, avg_loss)
                    continue

                # Compute BCE over probabilities
                sat_prob = scores.squeeze(-1)  # [B]
                ce_loss = F.binary_cross_entropy(
                    sat_prob, torch.ones_like(sat_prob), reduction="none"
                )  # [B]

                # IoU for predictions
                boxes = boxes.squeeze(1).clamp(0.0, 1.0)  # [B, 4]
                bboxes_clamped = bboxes.clamp(0.0, 1.0)  # [B, 4]
                pred_xyxy = box_cxcywh_to_xyxy(boxes)
                target_xyxy = box_cxcywh_to_xyxy(bboxes_clamped)
                # Use simple IoU loss (1 - IoU)
                iou_matrix = box_iou(pred_xyxy, target_xyxy)  # [B, B]
                iou_vec = torch.diag(iou_matrix)  # [B] - IoU for each sample
                iou_loss_vec = 1 - iou_vec  # [B] - IoU loss for each sample

                # Combine losses
                total_loss_vector = 2.0 * ce_loss + 5.0 * iou_loss_vec
                loss = total_loss_vector.mean()
                loss_dict = {"total_loss": loss}

            total_loss += loss.item()
            num_batches += 1

            # Update progress tracker
            avg_loss = total_loss / num_batches
            progress.update(batch_idx, avg_loss)

    progress.close()

    avg_total_loss = total_loss / num_batches if num_batches > 0 else 0.0
    loss_components = {}

    # Compute AP metrics if requested
    if compute_ap_metrics and all_outputs and all_targets:
        individual_outputs = []
        individual_targets = []

        for batch_out, batch_tgt in zip(all_outputs, all_targets):
            batch_size = batch_out["boxes"].shape[0]
            for i in range(batch_size):
                # For single prediction models, use the single prediction directly
                boxes = batch_out["boxes"][i]  # [1, 4] or [K, 4]
                scores = batch_out["scores"][i]  # [1] or [K]

                # If we have multiple predictions, filter by confidence
                if len(scores) > 1:
                    # Keep only confident predictions (threshold = 0.1 for stability)
                    confidence_threshold = 0.1
                    valid_mask = scores >= confidence_threshold

                    if valid_mask.any():
                        filtered_boxes = boxes[valid_mask]
                        filtered_scores = scores[valid_mask]

                        # Limit to top-5 predictions per image for single-object detection
                        if len(filtered_scores) > 5:
                            top_indices = torch.topk(filtered_scores, k=5)[1]
                            filtered_boxes = filtered_boxes[top_indices]
                            filtered_scores = filtered_scores[top_indices]
                    else:
                        # If no confident predictions, keep the best one
                        best_idx = torch.argmax(scores)
                        filtered_boxes = boxes[best_idx : best_idx + 1]
                        filtered_scores = scores[best_idx : best_idx + 1]
                else:
                    # Single prediction - use directly
                    filtered_boxes = boxes
                    filtered_scores = scores

                individual_outputs.append({"boxes": filtered_boxes, "scores": filtered_scores})
                individual_targets.append({"boxes": batch_tgt["boxes"][i]})

        ap_metrics = compute_detection_metrics(individual_outputs, individual_targets)
        loss_components.update(ap_metrics)

    # Log results
    if writer is not None:
        writer.add_scalar("Loss/Validation/Total", avg_total_loss, epoch)
        if loss_components:
            for metric_name, metric_value in loss_components.items():
                if metric_name.startswith("AP@") and isinstance(metric_value, (int, float)):
                    writer.add_scalar(f"Metrics/{metric_name}", metric_value, epoch)

    return avg_total_loss, loss_components
