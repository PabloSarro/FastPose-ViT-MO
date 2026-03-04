# This file contains the metrics used in the object detection module

import torch
import numpy as np
from object_detector.utils import box_iou, box_cxcywh_to_xyxy
from object_detector.constants import EPSILON, DEFAULT_IOU_THRESHOLDS, DEFAULT_CONFIDENCE_THRESHOLD


def compute_ap(precision: np.ndarray, recall: np.ndarray) -> float:
    """
    Compute Average Precision (AP) from precision and recall arrays.
    Uses the COCO evaluation protocol with interpolation.

    Args:
        precision: Array of precision values
        recall: Array of recall values

    Returns:
        Average Precision value
    """
    if precision.size == 0:
        return 0.0

    # Add sentinel values at the beginning and end
    precision = np.concatenate(([0.0], precision, [0.0]))
    recall = np.concatenate(([0.0], recall, [1.0]))

    # Ensure precision is monotonically decreasing
    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])

    # Find points where recall changes
    indices = np.where(recall[1:] != recall[:-1])[0]

    # Compute area under curve using trapezoidal rule
    ap = np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1])

    return ap


def compute_iou_matrix(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU matrix between predicted and ground truth boxes.

    Args:
        pred_boxes: Tensor of shape (N, 4) with predicted boxes in cxcywh format
        gt_boxes: Tensor of shape (M, 4) with ground truth boxes in cxcywh format

    Returns:
        IoU matrix of shape (N, M)
    """
    if pred_boxes.numel() == 0 or gt_boxes.numel() == 0:
        return torch.zeros((pred_boxes.size(0), gt_boxes.size(0)), device=pred_boxes.device)

    # Convert to xyxy format using centralized function
    pred_xyxy = box_cxcywh_to_xyxy(pred_boxes)
    gt_xyxy = box_cxcywh_to_xyxy(gt_boxes)

    # Use centralized IoU computation
    return box_iou(pred_xyxy, gt_xyxy)


def evaluate_predictions_single_class(
    predictions: list[dict[str, torch.Tensor]],
    targets: list[dict[str, torch.Tensor]],
    iou_thresholds: list[float] | None = None,
) -> dict[str, float]:
    """Evaluate predictions for single class detection using COCO-style metrics.

    Optimized version with vectorized operations and reduced redundancy.

    Args:
        predictions: List of prediction dicts, each containing:
            - 'boxes': Tensor of shape (N, 4) with predicted boxes in cxcywh format
            - 'scores': Tensor of shape (N,) with confidence scores
        targets: List of target dicts, each containing:
            - 'boxes': Tensor of shape (M, 4) with ground truth boxes in cxcywh format
        iou_thresholds: List of IoU thresholds to evaluate. Defaults to standard
            COCO thresholds [0.50, 0.55, ..., 0.95].

    Returns:
        Dictionary with AP metrics including AP@0.50, AP@0.75, AP@[0.50:0.95],
        and prediction/target counts.
    """
    if iou_thresholds is None:
        iou_thresholds = DEFAULT_IOU_THRESHOLDS

    predictions_len = len(predictions)
    targets_len = len(targets)
    if predictions_len != targets_len:
        raise ValueError("Number of predictions and targets must match")

    # Pre-collect all data and compute total GT count
    pred_boxes_list = []
    pred_scores_list = []
    pred_img_ids = []
    gt_boxes_list = []
    gt_img_ids = []
    gt_start_indices = []

    num_gt_total = 0
    gt_current_idx = 0

    for img_id, (pred, tgt) in enumerate(zip(predictions, targets)):
        gt_boxes = tgt["boxes"]
        pred_boxes = pred["boxes"]
        pred_scores = pred["scores"]

        gt_boxes_count = len(gt_boxes)
        pred_boxes_count = len(pred_boxes)

        num_gt_total += gt_boxes_count

        if gt_boxes_count > 0:
            gt_boxes_list.append(gt_boxes)
            gt_img_ids.extend([img_id] * gt_boxes_count)
            gt_start_indices.append(gt_current_idx)
            gt_current_idx += gt_boxes_count

        if pred_boxes_count > 0:
            pred_boxes_list.append(pred_boxes)
            pred_scores_list.append(pred_scores)
            pred_img_ids.extend([img_id] * pred_boxes_count)

    if not pred_boxes_list:
        # No predictions made
        return {
            "AP@0.50": 0.0,
            "AP@0.75": 0.0,
            "AP@[0.50:0.95]": 0.0,
            "num_predictions": 0,
            "num_targets": num_gt_total,
        }

    # Fast single-GT-per-image evaluation: avoid global IoU matrix (O(N^2)).
    # For each prediction, compute IoU only with its own image GT.
    # Then sort predictions globally by score and compute TP/FP per IoU threshold.
    device = pred_boxes_list[0].device if pred_boxes_list else torch.device("cpu")
    # Build per-prediction arrays
    per_pred_iou = []  # IoU with that image's GT
    per_pred_scores = []

    # We also track a mapping from prediction index to image id to handle one-GT matching rule
    per_pred_img_id = []

    # Compute IoUs per image efficiently
    for img_id, (pred, tgt) in enumerate(zip(predictions, targets)):
        boxes = pred["boxes"]  # [K, 4] (cxcywh)
        scores = pred["scores"]  # [K]
        gt = tgt["boxes"]  # [1, 4]
        if boxes.numel() == 0:
            continue
        # Ensure shapes
        boxes = boxes if boxes.dim() == 2 else boxes.view(-1, 4)
        gt_box = gt[0].unsqueeze(0)  # [1, 4]
        # Vector IoU between K preds and single GT
        ious = compute_iou_matrix(boxes, gt_box).squeeze(1)  # [K]
        per_pred_iou.append(ious)
        per_pred_scores.append(scores)
        per_pred_img_id.append(
            torch.full((boxes.shape[0],), img_id, device=boxes.device, dtype=torch.long)
        )

    if not per_pred_scores:
        return {
            "AP@0.50": 0.0,
            "AP@0.75": 0.0,
            "AP@[0.50:0.95]": 0.0,
            "num_predictions": 0,
            "num_targets": num_gt_total,
        }

    all_iou = torch.cat(per_pred_iou, dim=0)  # [P]
    all_scores = torch.cat(per_pred_scores, dim=0)  # [P]
    all_img_ids = torch.cat(per_pred_img_id, dim=0)  # [P]

    # Sort predictions by score
    order = torch.argsort(all_scores, descending=True)
    all_iou = all_iou[order]
    all_img_ids = all_img_ids[order]
    num_predictions = all_iou.numel()

    ap_scores = {}
    for iou_thresh in iou_thresholds:
        # Track if GT of an image is already matched
        matched = torch.zeros(targets_len, dtype=torch.bool, device=device)
        tp = torch.zeros(num_predictions, device=device)
        fp = torch.zeros(num_predictions, device=device)
        # Linear pass
        for idx in range(num_predictions):
            img_id = int(all_img_ids[idx].item())
            if all_iou[idx] >= iou_thresh and not matched[img_id]:
                tp[idx] = 1
                matched[img_id] = True
            else:
                fp[idx] = 1
        # Cum sums and AP
        tp_c = torch.cumsum(tp, dim=0).cpu().numpy()
        fp_c = torch.cumsum(fp, dim=0).cpu().numpy()
        precision = tp_c / (tp_c + fp_c + EPSILON)
        recall = tp_c / (num_gt_total + EPSILON)
        ap_scores[f"AP@{iou_thresh:.2f}"] = compute_ap(precision, recall)

    # Compute mean AP over all thresholds
    mean_ap = np.mean(list(ap_scores.values()))

    return {
        "AP@0.50": ap_scores.get("AP@0.50", 0.0),
        "AP@0.75": ap_scores.get("AP@0.75", 0.0),
        "AP@[0.50:0.95]": mean_ap,
        "num_predictions": num_predictions,
        "num_targets": num_gt_total,
        **ap_scores,
    }


def _compute_tp_fp_vectorized(
    iou_matrix: torch.Tensor,
    pred_img_ids: torch.Tensor,
    gt_img_ids: torch.Tensor,
    iou_thresh: float,
    num_predictions: int,
) -> torch.Tensor:
    """Compute true positives and false positives using vectorized operations.

    Args:
        iou_matrix: IoU matrix of shape (num_predictions, num_gt).
        pred_img_ids: Image IDs for each prediction.
        gt_img_ids: Image IDs for each ground truth box.
        iou_thresh: IoU threshold for considering a match.
        num_predictions: Total number of predictions.

    Returns:
        Tensor of shape (num_predictions, 2) with [TP, FP] for each prediction.
    """
    device = iou_matrix.device
    tp_fp = torch.zeros((num_predictions, 2), device=device)

    if iou_matrix.size(1) == 0:
        # No ground truth boxes - all predictions are false positives
        tp_fp = tp_fp.clone()
        tp_fp[:, 1] = 1
        return tp_fp

    # Create mask for same image predictions and ground truth
    same_image_mask = pred_img_ids.unsqueeze(1) == gt_img_ids.unsqueeze(0)

    # Mask IoU matrix to only consider same image matches
    masked_iou = iou_matrix * same_image_mask.float()

    # Find best IoU for each prediction
    max_ious, max_indices = torch.max(masked_iou, dim=1)

    # Track which GT boxes have been matched
    gt_matched = torch.zeros(iou_matrix.size(1), dtype=torch.bool, device=device)

    for pred_idx in range(num_predictions):
        max_iou = max_ious[pred_idx]
        gt_idx = max_indices[pred_idx]

        # Check if IoU exceeds threshold and GT not already matched
        if max_iou >= iou_thresh and not gt_matched[gt_idx]:
            tp_fp[pred_idx, 0] = 1  # True positive
            gt_matched[gt_idx] = True
        else:
            tp_fp[pred_idx, 1] = 1  # False positive

    return tp_fp


def compute_detection_metrics(
    model_outputs: list[dict[str, torch.Tensor]],
    ground_truth: list[torch.Tensor | dict[str, torch.Tensor]],
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> dict[str, float]:
    """Compute detection metrics from model outputs and ground truth.

    Optimized version with reduced memory allocations.

    Args:
        model_outputs: List of model output dicts, each containing:
            - 'boxes': Tensor of predicted boxes
            - 'scores': Tensor of confidence scores
        ground_truth: List of ground truth box tensors (cxcywh format) or dicts
            with 'boxes' key.
        confidence_threshold: Minimum confidence threshold for predictions.

    Returns:
        Dictionary with computed AP metrics.
    """
    # Pre-allocate lists for better memory efficiency
    predictions = []
    targets = []

    # Process all data in one pass
    for outputs, gt_data in zip(model_outputs, ground_truth):
        # Extract ground truth boxes
        if isinstance(gt_data, dict):
            gt_boxes = gt_data["boxes"]
        else:
            gt_boxes = gt_data

        # Ensure proper tensor dimensions for ground truth
        if gt_boxes.dim() == 1 and gt_boxes.numel() > 0:
            gt_boxes = gt_boxes.unsqueeze(0)
        elif gt_boxes.numel() == 0:
            gt_boxes = gt_boxes.view(0, 4)

        # Process predictions with confidence filtering
        if "scores" in outputs and outputs["scores"].numel() > 0:
            scores = outputs["scores"]
            boxes = outputs["boxes"]

            # Ensure proper dimensions
            if scores.dim() == 0:
                scores = scores.unsqueeze(0)
            if boxes.dim() == 1 and boxes.numel() > 0:
                boxes = boxes.unsqueeze(0)

            # Apply confidence threshold efficiently
            if confidence_threshold > 0.0:
                mask = scores >= confidence_threshold
                filtered_boxes = boxes[mask]
                filtered_scores = scores[mask]
            else:
                filtered_boxes = boxes
                filtered_scores = scores
        else:
            # Handle case with no scores or empty predictions
            boxes = outputs.get("boxes", torch.empty((0, 4)))
            if boxes.dim() == 1 and boxes.numel() > 0:
                boxes = boxes.unsqueeze(0)
            elif boxes.numel() == 0:
                boxes = boxes.view(0, 4)

            filtered_boxes = boxes
            filtered_scores = torch.ones(len(filtered_boxes), device=boxes.device)

        # Append processed data
        predictions.append({"boxes": filtered_boxes, "scores": filtered_scores})
        targets.append({"boxes": gt_boxes})

    # Compute AP metrics using optimized function
    return evaluate_predictions_single_class(predictions, targets)


def format_metrics(metrics: dict[str, float]) -> str:
    """Format metrics for logging.

    Args:
        metrics: Dictionary of computed metrics.

    Returns:
        Formatted string suitable for logging output.
    """
    if not metrics:
        return "No metrics computed"

    lines = []
    lines.append("Detection Metrics:")
    lines.append("-" * 40)

    # Main metrics
    main_metrics = ["AP@0.50", "AP@0.75", "AP@[0.50:0.95]"]
    for metric in main_metrics:
        if metric in metrics:
            lines.append(f"{metric:<15}: {metrics[metric]:.4f}")

    # Additional info
    if "num_predictions" in metrics and "num_targets" in metrics:
        lines.append(f"{'Predictions':<15}: {metrics['num_predictions']}")
        lines.append(f"{'Targets':<15}: {metrics['num_targets']}")

    return "\n".join(lines)
