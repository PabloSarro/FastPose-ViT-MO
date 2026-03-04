# Simplified LW-DETR implementation for single satellite detection
# Uses the original LW-DETR with single query modification

import torch
import torch.nn as nn
import sys
from pathlib import Path


def get_lwdetr_model(variant="tiny", num_classes=2, num_queries=1):
    """
    Get LW-DETR model with single query optimization.

    This function uses the original LW-DETR code with minimal modifications.
    """
    # Add LW-DETR to Python path
    lwdetr_path = str(Path(__file__).parent.parent / "LW-DETR")
    if lwdetr_path not in sys.path:
        sys.path.insert(0, lwdetr_path)

    try:
        # Import original LW-DETR
        from models.lwdetr import build
        from util.misc import NestedTensor, nested_tensor_from_tensor_list

        # Create minimal args for LW-DETR
        class Args:
            def __init__(self):
                # Model architecture
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.num_classes = num_classes
                self.num_queries = num_queries  # Single query for single object
                self.dataset_file = "satellite"  # Custom dataset

                # Variant configuration (keep original LW-DETR settings)
                if variant == "tiny":
                    self.encoder = "vit_tiny"
                elif variant == "small":
                    self.encoder = "vit_small"
                elif variant == "medium":
                    self.encoder = "vit_base"
                else:
                    raise ValueError(f"Unknown variant: {variant}")

                # Standard LW-DETR configuration
                self.projector_scale = [0.5, 1.0]
                self.hidden_dim = 256
                self.nheads = 8
                self.enc_layers = 6  # Keep original
                self.dec_layers = 6  # Keep original
                self.dropout = 0.1
                self.dim_feedforward = 2048  # Keep original
                self.pre_norm = False
                self.num_feature_levels = len(self.projector_scale)

                # Single object optimizations
                self.aux_loss = False
                self.group_detr = 1
                self.two_stage = False
                self.lite_refpoint_refine = False
                self.bbox_reparam = False

                # Loss configuration
                self.cls_loss_coef = 1.0
                self.bbox_loss_coef = 5.0
                self.giou_loss_coef = 2.0
                self.focal_alpha = 0.25

                # Matcher configuration
                self.set_cost_class = 1.0
                self.set_cost_bbox = 5.0
                self.set_cost_giou = 2.0

                # Position encoding
                self.position_embedding = "sine"

                # Backbone settings
                self.vit_encoder_num_layers = 12
                self.pretrained_encoder = True
                self.window_block_indexes = []
                self.drop_path = 0.1
                self.out_feature_indexes = [3, 6, 9, 12]
                self.lr_backbone = 1e-5

        args = Args()

        # Build the model using original LW-DETR
        model = build(args)

        print(f"✅ LW-DETR {variant} created with {num_queries} query")
        return model, NestedTensor, nested_tensor_from_tensor_list

    except ImportError as e:
        print(f"❌ LW-DETR import failed: {e}")
        return None, None, None


class SimpleLWDETR(nn.Module):
    """Simplified wrapper for LW-DETR with single satellite detection"""

    def __init__(self, variant="tiny"):
        super().__init__()

        self.variant = variant
        self.lwdetr_model, self.NestedTensor, self.nested_tensor_from_tensor_list = (
            get_lwdetr_model(
                variant=variant,
                num_classes=2,  # background + satellite
                num_queries=1,  # Single query for single object
            )
        )

        if self.lwdetr_model is None:
            raise ImportError("Failed to create LW-DETR model")

        print(f"LW-DETR {variant} ready for single satellite detection")

    def forward(self, images):
        """Forward pass handling NestedTensor conversion"""
        # Convert to NestedTensor if needed
        if isinstance(images, torch.Tensor):
            if self.nested_tensor_from_tensor_list:
                nested_images = self.nested_tensor_from_tensor_list(
                    [images[i] for i in range(images.shape[0])]
                )
            else:
                nested_images = images  # Fallback
        else:
            nested_images = images

        # Forward through LW-DETR
        outputs = self.lwdetr_model(nested_images)

        # Extract single prediction
        pred_boxes = outputs["pred_boxes"][:, 0]  # [B, 4] - single query
        pred_logits = outputs["pred_logits"][:, 0, 1]  # [B] - satellite class

        return {
            "pred_boxes": pred_boxes.unsqueeze(1),  # [B, 1, 4]
            "pred_logits": pred_logits.unsqueeze(1),  # [B, 1]
            "boxes": pred_boxes.unsqueeze(1),  # Compatibility
            "scores": torch.sigmoid(pred_logits).unsqueeze(1),
        }

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load_from_pretrained(self, path):
        checkpoint = torch.load(path, map_location="cpu")
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
        self.load_state_dict(state_dict, strict=False)

    def get_parameter_groups(self, lr=1e-4):
        return [{"params": self.parameters(), "lr": lr}]


def create_simple_lwdetr(variant="tiny"):
    """Create simplified LW-DETR for single satellite detection"""
    try:
        return SimpleLWDETR(variant=variant)
    except Exception as e:
        print(f"Failed to create LW-DETR: {e}")
        return None


# Test if this approach works
LWDETR_SIMPLE_AVAILABLE = False
try:
    test_model = get_lwdetr_model("tiny", 2, 1)
    if test_model[0] is not None:
        LWDETR_SIMPLE_AVAILABLE = True
        print("✅ Simple LW-DETR is available!")
except Exception as e:
    print(f"❌ Simple LW-DETR test failed: {e}")
