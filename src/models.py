# This file contains the models used in this project

import warnings

import torch
import torch.nn as nn

from torchvision.models import convnext_base, ConvNeXt_Base_Weights

from src.vit_models import (
    vit_b_16,
    vit_b_32,
    vit_l_16,
    vit_l_32,
    vit_h_14,
    ViT_B_16_Weights,
    ViT_B_32_Weights,
    ViT_L_16_Weights,
    ViT_L_32_Weights,
    ViT_H_14_Weights,
)

# Supported backbone models (ViT and ConvNeXt variants)
SUPPORTED_VIT_MODELS = [
    "vit_b_16",
    "vit_b_16_no_weights",
    "vit_b_16_384",
    "vit_b_16_384_no_weights",
    "vit_b_32",
    "vit_l_16",
    "vit_l_16_512",
    "vit_l_32",
    "vit_h_14",
    "vit_h_14_518",
    "convnext_base",
]

# Dictionary mapping model names to their corresponding functions, weights, outputs sizes, and image sizes in format (height, width)
VIT_MODELS = {
    # ViT models
    "vit_b_16": (vit_b_16, ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1, 768, (224, 224)),
    "vit_b_16_no_weights": (vit_b_16, None, 768, (224, 224)),
    "vit_b_16_384": (vit_b_16, ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1, 768, (384, 384)),
    "vit_b_16_384_no_weights": (vit_b_16, None, 768, (384, 384)),
    "vit_b_32": (vit_b_32, ViT_B_32_Weights.IMAGENET1K_V1, 768, (224, 224)),
    "vit_l_16": (vit_l_16, ViT_L_16_Weights.IMAGENET1K_SWAG_LINEAR_V1, 1024, (224, 224)),
    "vit_l_16_512": (vit_l_16, ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1, 1024, (512, 512)),
    "vit_l_32": (vit_l_32, ViT_L_32_Weights.IMAGENET1K_V1, 1024, (224, 224)),
    "vit_h_14": (vit_h_14, ViT_H_14_Weights.IMAGENET1K_SWAG_LINEAR_V1, 1280, (224, 224)),
    "vit_h_14_518": (vit_h_14, ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1, 1280, (518, 518)),
    # ConvNeXt models
    "convnext_base": (convnext_base, ConvNeXt_Base_Weights.IMAGENET1K_V1, 1024, (224, 224)),
}


class MLPWithProjection(nn.Module):
    """
    MLP with a linear projection layer at the end, with optional normalization and dropout
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: list[int],
        activation_function: nn.Module,
        out_channels: int,
        use_layer_norm: bool = False,
        dropout_rate: float = 0.0,
        use_residual: bool = True,
    ):
        """
        Initialize the MLP with projection layer

        Args:
            in_channels (int): Number of input channels
            hidden_channels (list): List of hidden layer dimensions
            activation_function (nn.Module): Activation function to use
            out_channels (int): Number of output channels
            use_layer_norm (bool): Whether to use layer normalization after each hidden layer
            dropout_rate (float): Dropout rate (0.0 means no dropout)
            use_residual (bool): Whether to use residual connections
        """
        super().__init__()
        self.use_residual = use_residual

        # Build custom MLP with normalization and dropout support
        layers = []
        prev_dim = in_channels

        for hidden_dim in hidden_channels:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(activation_function())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)
        self.projection = nn.Linear(hidden_channels[-1], out_channels)

        # Residual connection projection if dimensions don't match
        if self.use_residual and in_channels != out_channels:
            self.residual_proj = nn.Linear(in_channels, out_channels)
        else:
            self.residual_proj = None

        # Optimized initialization
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform forward pass through MLP with optional residual connection.

        Args:
            x: Input tensor of shape (batch_size, in_channels).

        Returns:
            Output tensor of shape (batch_size, out_channels).
        """
        if self.use_residual:
            residual = x

        x = self.mlp(x)
        x = self.projection(x)

        if self.use_residual:
            if self.residual_proj is not None:
                residual = self.residual_proj(residual)
            x = x + residual

        return x

    def _initialize_weights(self) -> None:
        """Initialize weights optimized for pose estimation MLPs."""
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                # He initialization for ReLU activations
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Smaller initialization for final projection to improve training stability
        nn.init.normal_(self.projection.weight, mean=0.0, std=0.01)
        if self.projection.bias is not None:
            nn.init.zeros_(self.projection.bias)

        # Identity-like initialization for residual projection
        if self.residual_proj is not None:
            nn.init.kaiming_normal_(
                self.residual_proj.weight, mode="fan_in", nonlinearity="linear"
            )
            if self.residual_proj.bias is not None:
                nn.init.zeros_(self.residual_proj.bias)


class FastPoseViT(nn.Module):
    """
    6D pose estimation model using Vision Transformer or ConvNeXt backbone with MLPs
    """

    def __init__(
        self,
        vit_model: str,
        num_hidden_layers: int,
        hidden_layer_dim: int,
        out_dim_translation: int,
        out_dim_rotation: int,
        vit_weights: str = None,
        merge_outputs: bool = False,
        nb_class_tokens: int = 1,
        use_layer_norm: bool = False,
        dropout_rate: float = 0.0,
        use_residual: bool = True,
        no_mlp: bool = False,
    ):
        """
        Model initialization

        Args:
            vit_model (str): Backbone model name (ViT or ConvNeXt variant)
            num_hidden_layers (int): Number of hidden layers in the MLPs
            hidden_layer_dim (int): Dimension of hidden layers in the MLPs
            out_dim_translation (int): Dimension of the output for translation
            out_dim_rotation (int): Dimension of the output for rotation
            vit_weights (str, optional): Path to custom backbone weights. Defaults to None.
            merge_outputs (bool, optional): Whether to merge translation and rotation MLP heads. Defaults to False.
            nb_class_tokens (int, optional): Number of class tokens (ViT only, ignored for ConvNeXt). Defaults to 1.
            use_layer_norm (bool, optional): Whether to use layer normalization in MLPs. Defaults to False.
            dropout_rate (float, optional): Dropout rate for MLPs. Defaults to 0.0.
            use_residual (bool, optional): Whether to use residual connections in MLPs. Defaults to True.
            no_mlp (bool, optional): Whether to skip MLP layers and use direct projection only. Defaults to False.
        """
        super().__init__()

        if vit_model not in VIT_MODELS:
            raise ValueError(
                f"Unsupported model: {vit_model}. Choose from {list(VIT_MODELS.keys())}"
            )

        # Determine backbone type
        self.backbone_type = "convnext" if vit_model.startswith("convnext") else "vit"

        model_func, default_weights, backbone_output_dim, img_size = VIT_MODELS[vit_model]

        if self.backbone_type == "vit":
            # Multiply output_dim by the number of class tokens for ViT
            backbone_output_dim *= nb_class_tokens

            # Load Vision Transformer
            self.backbone = model_func(
                weights=None,
                nb_class_tokens=nb_class_tokens,
                image_size=img_size[0],
                dropout=dropout_rate,
                attention_dropout=dropout_rate,
            )

            if vit_weights:
                # Load custom weights
                state_dict = torch.load(vit_weights)
                adapted_state_dict = self._adapt_state_dict_for_nb_class_tokens(
                    state_dict, nb_class_tokens
                )
                self.backbone.load_state_dict(adapted_state_dict)
            else:
                # Load default weights if available
                if default_weights is not None:
                    state_dict = default_weights.get_state_dict(progress=True, check_hash=True)
                    adapted_state_dict = self._adapt_state_dict_for_nb_class_tokens(
                        state_dict, nb_class_tokens
                    )
                    self.backbone.load_state_dict(adapted_state_dict)
                else:
                    for name, param in self.backbone.named_parameters():
                        if "weight" in name and param.dim() >= 2:
                            nn.init.xavier_uniform_(param)
                        elif "bias" in name and param.dim() == 1:
                            nn.init.constant_(param, 0)

            # Remove classification head
            self.backbone.heads = nn.Identity()

        else:  # ConvNeXt
            if nb_class_tokens > 1:
                warnings.warn(
                    f"nb_class_tokens={nb_class_tokens} is ignored for ConvNeXt backbone "
                    "(only applicable to ViT models)"
                )

            # Load ConvNeXt with pretrained weights
            if vit_weights:
                # Load custom weights
                self.backbone = model_func(weights=None)
                state_dict = torch.load(vit_weights, map_location="cpu", weights_only=True)
                self.backbone.load_state_dict(state_dict)
            else:
                self.backbone = model_func(weights=default_weights)

            # Remove classification head, keep only feature extractor
            self.backbone.classifier = nn.Identity()

            # Global average pooling to convert feature maps to feature vector
            self.gap = nn.AdaptiveAvgPool2d(1)

        # Add translation and rotation heads
        self.merge_outputs = merge_outputs
        self.no_mlp = no_mlp

        if no_mlp:
            # Direct projection without MLP layers
            if merge_outputs:
                out_dim = out_dim_translation + out_dim_rotation
                self.output = nn.Linear(backbone_output_dim, out_dim)
            else:
                self.translation = nn.Linear(backbone_output_dim, out_dim_translation)
                self.rotation = nn.Linear(backbone_output_dim, out_dim_rotation)
        else:
            # Use MLP with projection layers
            if merge_outputs:
                out_dim = out_dim_translation + out_dim_rotation
                self.output = MLPWithProjection(
                    backbone_output_dim,
                    [hidden_layer_dim] * num_hidden_layers,
                    nn.ReLU,
                    out_dim,
                    use_layer_norm=use_layer_norm,
                    dropout_rate=dropout_rate,
                    use_residual=use_residual,
                )
            else:
                self.translation = MLPWithProjection(
                    backbone_output_dim,
                    [hidden_layer_dim] * num_hidden_layers,
                    nn.ReLU,
                    out_dim_translation,
                    use_layer_norm=use_layer_norm,
                    dropout_rate=dropout_rate,
                    use_residual=use_residual,
                )
                self.rotation = MLPWithProjection(
                    backbone_output_dim,
                    [hidden_layer_dim] * num_hidden_layers,
                    nn.ReLU,
                    out_dim_rotation,
                    use_layer_norm=use_layer_norm,
                    dropout_rate=dropout_rate,
                    use_residual=use_residual,
                )

    def _adapt_state_dict_for_nb_class_tokens(
        self, state_dict: dict, nb_class_tokens: int
    ) -> dict:
        """
        Function to adapt the state dict of a Vision Transformer model to have a different number of class tokens.
        The standard torchvision weights have a single class token, but we can add more. We thus need to handle the case
        where we want to load these torchvision weights into a model with a different number of class tokens.

        Args:
            state_dict (dict): The state dict to adapt
            nb_class_tokens (int): The number of class tokens in the new model

        Returns:
            dict: The adapted state dict
        """
        # Get the original class token
        original_class_token = state_dict["class_token"]  # Shape: [1, 1, hidden_dim]
        hidden_dim = original_class_token.shape[-1]

        if original_class_token.shape[1] != nb_class_tokens:
            # Create new class tokens
            new_class_tokens = torch.zeros(1, nb_class_tokens, hidden_dim)
            # Copy the original token to the first position
            new_class_tokens[:, 0:1, :] = original_class_token
            # Initialize remaining tokens with xavier uniform
            if nb_class_tokens > 1:
                nn.init.xavier_uniform_(new_class_tokens[:, 1:, :])
            state_dict["class_token"] = new_class_tokens

            # Handle position embeddings
            pos_embed = state_dict["encoder.pos_embedding"]  # Shape: [1, N+1, hidden_dim]
            pos_token_embed = pos_embed[:, :1, :]  # Class token position embedding
            pos_img_embed = pos_embed[:, 1:, :]  # Image patches position embeddings

            # Create new position embeddings for additional class tokens
            new_pos_tokens = torch.zeros(1, nb_class_tokens, hidden_dim)
            new_pos_tokens[:, 0:1, :] = pos_token_embed
            # Initialize remaining position embeddings with xavier uniform
            if nb_class_tokens > 1:
                nn.init.xavier_uniform_(new_pos_tokens[:, 1:, :])

            # Concatenate new position embeddings
            new_pos_embed = torch.cat([new_pos_tokens, pos_img_embed], dim=1)
            state_dict["encoder.pos_embedding"] = new_pos_embed

        return state_dict

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model

        Args:
            x (torch.Tensor): Input image tensor

        Returns:
            torch.Tensor: Output tensor
        """
        if self.backbone_type == "vit":
            x = self.backbone(x)
        else:  # ConvNeXt
            x = self.backbone.features(x)  # [batch, channels, H, W]
            x = self.gap(x)  # [batch, channels, 1, 1]
            x = x.flatten(1)  # [batch, channels]

        if self.merge_outputs:
            return self.output(x)
        else:
            return self.translation(x), self.rotation(x)

    def load_from_pretrained(self, path: str) -> None:
        """
        Load model weights from a pretrained model

        Args:
            path (str): Path to the pretrained model weights

        Returns:
            None
        """
        state_dict = torch.load(path, map_location="cpu", weights_only=True)

        # Check if loading full FastPoseViT model state dict
        if "backbone.class_token" in state_dict or "backbone.features.0.0.weight" in state_dict:
            self.load_state_dict(state_dict)
        # Legacy support: loading old model with "vit." prefix
        elif "vit.class_token" in state_dict:
            # Rename vit.* keys to backbone.*
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("vit."):
                    new_key = "backbone." + key[4:]
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            self.load_state_dict(new_state_dict)
        # If loading only backbone state dict (ViT or ConvNeXt)
        else:
            if self.backbone_type == "vit":
                adapted_state_dict = self._adapt_state_dict_for_nb_class_tokens(
                    state_dict, self.backbone.class_token.shape[1]
                )
                self.backbone.load_state_dict(adapted_state_dict)
            else:
                self.backbone.load_state_dict(state_dict)

    def save(self, path: str) -> None:
        """
        Save the model weights

        Args:
            path (str): Path to save the model weights

        Returns:
            None
        """
        torch.save(self.state_dict(), path)
