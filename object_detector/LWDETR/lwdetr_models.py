# Clean LW-DETR implementation using local files (no external dependencies)

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from types import SimpleNamespace

# Import LW-DETR components from local files, robust against PYTHONPATH collisions
import os as _os
import sys as _sys

_BASE_DIR = _os.path.dirname(__file__)
_PKG = "object_detector"
if _BASE_DIR not in _sys.path:
    _sys.path.insert(0, _BASE_DIR)

# Proactively purge conflicting LW-DETR modules that point to an external clone
for _name in list(_sys.modules.keys()):
    if _name.startswith("lwdetr_"):
        _mod = _sys.modules.get(_name)
        _file = getattr(_mod, "__file__", "") if _mod else ""
        if "LW-DETR" in _file:
            _sys.modules.pop(_name, None)

try:
    # Import vendored modules using the package name to avoid collisions
    from object_detector.LWDETR import lwdetr_core as _core
    from object_detector.LWDETR.lwdetr_utils import nested_tensor_from_tensor_list

    build = _core.build
    LWDETR = _core.LWDETR
    SetCriterion = _core.SetCriterion
    LWDETR_AVAILABLE = True
    print("✅ LW-DETR successfully imported from local files!")
except Exception as e:
    import traceback as _traceback

    print(f"❌ Failed to import local LW-DETR: {e}")
    _traceback.print_exc()
    LWDETR_AVAILABLE = False


class LWDETRWrapper(nn.Module):
    """
    Clean LW-DETR wrapper for single-class satellite detection using local files.
    Optimized for speed with single query configuration.
    """

    def __init__(self, variant="tiny", num_classes=1, num_queries: int | None = None):
        super().__init__()

        if not LWDETR_AVAILABLE:
            raise ImportError("LW-DETR local files not available.")

        self.variant = variant
        self.num_classes = num_classes  # number of object classes (no-object is implicit)

        # Create LW-DETR configuration. By default we use single-query for single
        # object detection, but allow overriding num_queries to restore the base
        # multi-query behavior with Hungarian matching.
        args = self._create_single_object_config(variant, override_num_queries=num_queries)

        # Build model using local LW-DETR build function
        self.lwdetr_model, self._lwdetr_criterion, self._lwdetr_post = build(args)
        # Expose num_select for validation top-K selection
        self.num_select = getattr(args, "num_select", 100)

        logging.info(
            f"LW-DETR {variant} initialized for single satellite detection (local implementation)"
        )

        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logging.info(f"LW-DETR parameters: {total_params:,} total, {trainable_params:,} trainable")

        # Control whether to return all queries even in eval (for validation with Hungarian loss)
        self.return_all_queries = False

    def _create_single_object_config(self, variant, override_num_queries: int | None = None):
        """Create LW-DETR configuration with variant-default query count.

        Defaults follow the LW-DETR reference scripts:
        - tiny:   num_queries=100, num_select=100
        - small:  num_queries=300, num_select=300
        - medium: num_queries=300, num_select=300
        - large:  num_queries=300, num_select=300
        - xlarge: num_queries=300, num_select=300
        group_detr defaults to 13 as in the reference.
        """

        args = SimpleNamespace()

        # Core settings using original DETR architecture
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        args.num_classes = self.num_classes
        # num_queries set per variant below
        args.aux_loss = True  # Enable auxiliary losses for better training
        # Note: group_detr, two_stage, lite_refpoint_refine, bbox_reparam set per variant

        # Set group_detr=1 for single-object detection to avoid train/eval inconsistency
        args.group_detr = 1

        # Loss configuration for single object
        args.cls_loss_coef = 1.0
        args.bbox_loss_coef = 5.0
        args.giou_loss_coef = 2.0
        args.focal_alpha = 0.25

        # Matcher configuration (simplified for single object)
        args.set_cost_class = 1.0
        args.set_cost_bbox = 5.0
        args.set_cost_giou = 2.0

        # Variant-specific configurations matching original LW-DETR
        if variant == "tiny":
            args.encoder = "vit_tiny"
            args.num_queries = 100
            args.projector_scale = ["P4"]
            args.projector_num_blocks = 3  # Original default
            args.hidden_dim = 256
            args.nheads = 8
            args.dec_layers = 3
            args.two_stage = True
            args.lite_refpoint_refine = True
            args.bbox_reparam = True
            args.vit_encoder_num_layers = 6
            args.window_block_indexes = [0, 2, 4]
            args.out_feature_indexes = [1, 3, 5]
            # Override group_detr for stability in single-object detection
            # args.group_detr = 13  # Original value, but causes train/eval inconsistency
            args.sa_nheads = 8
            args.ca_nheads = 16
            args.dec_n_points = 2

        elif variant == "small":
            args.encoder = "vit_tiny"
            args.num_queries = 300
            args.projector_scale = ["P4"]
            args.projector_num_blocks = 3  # Original default
            args.hidden_dim = 256
            args.nheads = 8
            args.dec_layers = 3
            args.two_stage = True
            args.lite_refpoint_refine = True
            args.bbox_reparam = True
            args.vit_encoder_num_layers = 10
            args.window_block_indexes = [0, 1, 3, 6, 7, 9]
            args.out_feature_indexes = [2, 4, 5, 9]
            # Override group_detr for stability in single-object detection
            # args.group_detr = 13  # Original value, but causes train/eval inconsistency
            args.sa_nheads = 8
            args.ca_nheads = 16
            args.dec_n_points = 2

        elif variant == "medium":
            args.encoder = "vit_small"
            args.num_queries = 300
            args.projector_scale = ["P4"]
            args.projector_num_blocks = 3  # Original default
            args.hidden_dim = 256
            args.nheads = 8
            args.dec_layers = 3
            args.two_stage = True
            args.lite_refpoint_refine = True
            args.bbox_reparam = True
            args.vit_encoder_num_layers = 10
            args.window_block_indexes = [0, 1, 3, 6, 7, 9]
            args.out_feature_indexes = [2, 4, 5, 9]
            # Override group_detr for stability in single-object detection
            # args.group_detr = 13  # Original value, but causes train/eval inconsistency
            args.sa_nheads = 8
            args.ca_nheads = 16
            args.dec_n_points = 2

        elif variant == "large":
            args.encoder = "vit_small"
            args.num_queries = 300
            args.projector_scale = ["P3", "P5"]
            args.projector_num_blocks = 3  # Original default
            args.hidden_dim = 384
            args.nheads = 12
            args.dec_layers = 3
            args.two_stage = True
            args.lite_refpoint_refine = True
            args.bbox_reparam = True
            args.vit_encoder_num_layers = 10
            args.window_block_indexes = [0, 1, 3, 6, 7, 9]
            args.out_feature_indexes = [2, 4, 5, 9]
            args.group_detr = 13
            args.sa_nheads = 12
            args.ca_nheads = 24
            args.dec_n_points = 4
            args.drop_path = 0.1

        elif variant == "xlarge":
            args.encoder = "vit_base"
            args.num_queries = 300
            args.projector_scale = ["P3", "P5"]
            args.projector_num_blocks = 3  # Original default
            args.hidden_dim = 384
            args.nheads = 12
            args.dec_layers = 3
            args.two_stage = True
            args.lite_refpoint_refine = True
            args.bbox_reparam = True
            args.vit_encoder_num_layers = 10
            args.window_block_indexes = [0, 1, 3, 6, 7, 9]
            args.out_feature_indexes = [2, 4, 5, 9]
            args.group_detr = 13
            args.sa_nheads = 12
            args.ca_nheads = 24
            args.dec_n_points = 4
            args.drop_path = 0.1

        else:
            raise ValueError(
                f"Unsupported LW-DETR variant: {variant}. Supported: tiny, small, medium, large, xlarge"
            )

        # Additional required parameters
        args.dropout = 0.1
        args.dim_feedforward = 2048
        args.pre_norm = False
        args.num_feature_levels = len(args.projector_scale)

        # Set default values for parameters not specified in variant configs
        if not hasattr(args, "drop_path"):
            args.drop_path = 0.0
        args.decoder_norm = "LN"

        # Backbone specific settings - already set in variant configs
        # Use no explicit checkpoint path by default; set to a str path to load
        args.pretrained_encoder = None

        # Position encoding settings
        args.position_embedding = "sine"  # Use sine position encoding
        args.lr_backbone = 1e-5
        # Loss/postprocess configuration
        # Default number selected for evaluation per reference scripts
        if variant == "tiny":
            args.num_select = 100
        else:
            args.num_select = 300
        args.use_varifocal_loss = False
        args.use_position_supervised_loss = False
        args.ia_bce_loss = False

        # Allow external override of num_queries if provided
        if isinstance(override_num_queries, int) and override_num_queries > 0:
            args.num_queries = int(override_num_queries)

        return args

    def forward(self, images):
        """
        Forward pass for LW-DETR with Hungarian matching for single satellite detection

        Args:
            images: Batch of images [B, 3, H, W] or NestedTensor

        Returns:
            Dict with all query predictions for Hungarian matching during training,
            and best single prediction during inference
        """
        # Convert to NestedTensor format if needed
        if isinstance(images, torch.Tensor):
            # Create NestedTensor from regular tensor batch
            nested_images = nested_tensor_from_tensor_list(
                [images[i] for i in range(images.shape[0])]
            )
        else:
            nested_images = images

        # Forward through LW-DETR
        outputs = self.lwdetr_model(nested_images)

        # During training or when explicitly requested, return all predictions for Hungarian matching
        if self.training or self.return_all_queries:
            return outputs

        # During inference, extract the best prediction per image
        pred_boxes = outputs["pred_boxes"]  # [B, num_queries, 4]
        pred_logits = outputs["pred_logits"]  # [B, num_queries, num_classes]

        # Find the query with highest confidence for satellite class.
        # LW-DETR uses sigmoid over classes (no softmax across classes).
        if pred_logits.shape[-1] > 1:
            confidence_scores = torch.sigmoid(pred_logits[..., 1])  # [B, num_queries]
        else:
            confidence_scores = torch.sigmoid(pred_logits[..., 0])  # [B, num_queries]

        # Get the query with maximum confidence per batch
        best_query_idx = torch.argmax(confidence_scores, dim=1)  # [B]
        batch_indices = torch.arange(pred_boxes.shape[0], device=pred_boxes.device)

        # Extract best predictions
        best_boxes = pred_boxes[batch_indices, best_query_idx]  # [B, 4]
        best_logits = pred_logits[batch_indices, best_query_idx]  # [B, num_classes]

        # Extract satellite class confidence for metrics
        if best_logits.shape[-1] > 1:
            satellite_scores = torch.sigmoid(best_logits[:, 1])  # [B]
        else:
            satellite_scores = torch.sigmoid(best_logits[:, 0])  # [B]

        return {
            "pred_boxes": best_boxes.unsqueeze(1),  # [B, 1, 4]
            "pred_logits": best_logits.unsqueeze(1),  # [B, 1, num_classes]
            "boxes": best_boxes.unsqueeze(1),  # For SimpleIoULoss path
            "scores": satellite_scores.unsqueeze(1),  # [B, 1] - single confidence score
        }

    def load_from_pretrained(self, path: str):
        """Load model weights from pretrained checkpoint with automatic variant detection"""
        logging.info(f"Loading pretrained weights from: {path}")

        # Check if the pretrained weights match the current model variant
        variant_in_path = None
        for variant in ["tiny", "small", "medium", "large", "xlarge"]:
            if variant in path.lower():
                variant_in_path = variant
                break

        if variant_in_path and variant_in_path != self.variant:
            logging.warning(
                f"⚠️  Loading {variant_in_path} weights into {self.variant} model - expect size mismatches"
            )
            logging.warning(
                f"   Recommended: Use LWDETR_{self.variant}_60e_coco.pth for best compatibility"
            )

        # Load with weights_only=False to handle argparse.Namespace in checkpoint
        # This is safe for trusted pretrained weights from official LW-DETR repository
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Detect if this checkpoint was saved from our wrapper (training in this repo)
        # In that case, keys already start with 'lwdetr_model.' and should be loaded directly.
        keys = list(state_dict.keys())
        is_wrapper_checkpoint = any(k.startswith("lwdetr_model.") for k in keys)

        if is_wrapper_checkpoint:
            logging.info(
                "Detected wrapper-style checkpoint; loading keys directly without remapping"
            )
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            total_keys = len(state_dict)
            loaded_keys = total_keys - len(unexpected_keys)
        else:
            # Map keys from checkpoint format to our wrapper format and filter incompatible layers
            # Pretrained weights use direct keys, but our model has 'lwdetr_model.' prefix
            mapped_state_dict = {}
            skipped_keys: list[str] = []

            for key, value in state_dict.items():
                # Skip classification layers that have different num_classes (e.g., COCO) when loading external weights
                if "class_embed" in key:
                    skipped_keys.append(key)
                    continue

                # Skip query embeddings when sizes do not match our configuration
                if "refpoint_embed.weight" in key or "query_feat.weight" in key:
                    # External checkpoints usually have many queries; we keep ours as-is
                    logging.info(f"Skipping {key} (query embeddings handled by current config)")
                    skipped_keys.append(key)
                    continue

                # Add the lwdetr_model prefix to match our wrapper structure
                mapped_key = f"lwdetr_model.{key}"
                mapped_state_dict[mapped_key] = value

            if skipped_keys:
                logging.info(
                    f"Skipped {len(skipped_keys)} incompatible classification/query layers from external weights"
                )

            # Load with strict=False to handle potential mismatches
            missing_keys, unexpected_keys = self.load_state_dict(mapped_state_dict, strict=False)

            # Count successfully loaded keys
            total_keys = len(state_dict)
            loaded_keys = total_keys - len(unexpected_keys)

        logging.info("Pretrained weight loading summary:")
        logging.info(f"  Total keys in checkpoint: {total_keys}")
        logging.info(f"  Successfully loaded: {loaded_keys}")
        logging.info(f"  Missing keys: {len(missing_keys)}")
        logging.info(f"  Unexpected keys: {len(unexpected_keys)}")

        if len(missing_keys) > 0:
            logging.info("Missing keys (common when num_queries differs):")
            for key in missing_keys[:5]:  # Show first 5
                logging.info(f"  - {key}")
            if len(missing_keys) > 5:
                logging.info(f"  ... and {len(missing_keys) - 5} more")

        if len(unexpected_keys) > 0:
            logging.info("Unexpected keys (skipped due to size mismatch):")
            for key in unexpected_keys[:5]:  # Show first 5
                logging.info(f"  - {key}")
            if len(unexpected_keys) > 5:
                logging.info(f"  ... and {len(unexpected_keys) - 5} more")

        return loaded_keys, len(missing_keys), len(unexpected_keys)

    def save(self, path: str):
        """Save model weights"""
        torch.save(self.state_dict(), path)

    def get_parameter_groups(
        self,
        lr: float = 1e-4,
        lr_encoder: float | None = None,
        lr_component_decay: float | None = None,
        lr_vit_layer_decay: float | None = None,
    ):
        """Return optimizer param groups.

        We accept RF‑DETR-style keyword args for compatibility but use a single
        param group for LW‑DETR. This avoids special‑casing in the training
        script while keeping optimizer setup simple and fast.
        """
        return [{"params": self.parameters(), "lr": lr}]


class SimplifiedSingleQueryLoss(nn.Module):
    """
    Simplified loss for single query LW-DETR (no Hungarian matching needed)
    """

    def __init__(
        self, giou_weight=5.0, bbox_weight=5.0, focal_weight=2.0, focal_alpha=0.25, focal_gamma=2.0
    ):
        super().__init__()
        self.giou_weight = giou_weight
        self.bbox_weight = bbox_weight
        self.focal_weight = focal_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        # Import box ops for GIoU calculation
        from object_detector.LWDETR import lwdetr_box_ops as box_ops

        self.box_ops = box_ops

    def forward(self, outputs, targets):
        """Direct loss computation for single query"""
        # Extract single predictions (no matching needed)
        pred_logits = outputs["pred_logits"]  # [B, 1, num_classes]
        pred_boxes = outputs["pred_boxes"]  # [B, 1, 4]

        batch_size = pred_logits.shape[0]
        device = pred_logits.device

        # Handle both raw bboxes and DETR-style targets
        if isinstance(targets, torch.Tensor):
            # Raw bboxes tensor [B, 4]
            target_boxes = targets
        else:
            # DETR-style targets: list of dicts with 'boxes' key
            target_boxes = torch.stack([t["boxes"][0] for t in targets])  # [B, 4]

        # Create target labels for satellite class
        target_labels = torch.tensor([1], dtype=torch.long, device=device).expand(
            batch_size
        )  # Satellite class

        # Classification loss (focal loss)
        pred_logits_flat = pred_logits.squeeze(1)  # [B, num_classes]
        ce_loss = F.cross_entropy(pred_logits_flat, target_labels, reduction="mean")

        # Focal loss modification
        pt = torch.exp(-ce_loss)
        focal_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * ce_loss

        # Box losses
        pred_boxes_flat = pred_boxes.squeeze(1)  # [B, 4]

        # L1 loss
        l1_loss = F.l1_loss(pred_boxes_flat, target_boxes, reduction="mean")

        # GIoU loss
        pred_boxes_xyxy = self.box_ops.box_cxcywh_to_xyxy(pred_boxes_flat)
        target_boxes_xyxy = self.box_ops.box_cxcywh_to_xyxy(target_boxes)
        giou_values = self.box_ops.generalized_box_iou(pred_boxes_xyxy, target_boxes_xyxy)
        giou_loss = 1 - giou_values.diag().mean()

        # Combine losses
        total_loss = (
            self.focal_weight * focal_loss
            + self.bbox_weight * l1_loss
            + self.giou_weight * giou_loss
        )

        return {
            "loss_ce": focal_loss,
            "loss_bbox": l1_loss,
            "loss_giou": giou_loss,
            "total_loss": total_loss,
        }


def get_lwdetr_loss(
    giou_weight: float = 5.0,
    bbox_weight: float = 5.0,
    focal_weight: float = 2.0,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
):
    """
    Create simplified loss function for single query LW-DETR

    Returns:
        Simplified loss function that bypasses Hungarian matching
    """
    return SimplifiedSingleQueryLoss(
        giou_weight=giou_weight,
        bbox_weight=bbox_weight,
        focal_weight=focal_weight,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
    )


def get_lwdetr_loss_original(
    giou_weight: float = 5.0,
    bbox_weight: float = 5.0,
    focal_weight: float = 2.0,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
):
    """
    Create LW-DETR SetCriterion loss function (original Hungarian matching version)
    """
    if not LWDETR_AVAILABLE:
        raise ImportError("LW-DETR dependencies not available")

    # Import Hungarian matcher from LW-DETR
    from object_detector.LWDETR.lwdetr_matcher import HungarianMatcher

    # Create Hungarian matcher with appropriate costs
    matcher = HungarianMatcher(
        cost_class=focal_weight,  # Classification cost
        cost_bbox=bbox_weight,  # L1 bbox cost
        cost_giou=giou_weight,  # GIoU cost
        focal_alpha=focal_alpha,  # Focal loss alpha
    )

    # Create SetCriterion with appropriate weights
    weight_dict = {
        "loss_ce": focal_weight,
        "loss_bbox": bbox_weight,
        "loss_giou": giou_weight,  # LW-DETR uses GIoU
    }

    # SetCriterion with correct parameters based on LW-DETR signature
    # num_classes should match the model's output (2 = background + satellite)
    criterion = SetCriterion(
        num_classes=1,  # 1 object class (satellite). No-object is implicit
        matcher=matcher,  # Hungarian matcher
        weight_dict=weight_dict,
        focal_alpha=focal_alpha,  # Focal loss alpha
        losses=["labels", "boxes", "cardinality"],  # Standard DETR losses
    )

    return criterion


def create_lwdetr_model(
    variant: str = "tiny", num_classes: int = 2, num_queries: int | None = None
):
    """
    Create LW-DETR wrapper model for single-class detection

    Args:
        variant: Model variant ("tiny", "small", "medium")
        num_classes: Number of classes (should be 2: background + satellite)

    Returns:
        LWDETRWrapper model optimized for single satellite detection
    """
    if not LWDETR_AVAILABLE:
        raise ImportError("LW-DETR dependencies not available")

    return LWDETRWrapper(variant=variant, num_classes=num_classes, num_queries=num_queries)
