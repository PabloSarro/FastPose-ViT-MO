# Download the ViT-B-16 weights and save them to the specified path

import os

import torch
import torchvision.models as models


def download_vit_weights(save_path: str = "vit_b_16_weights.pth") -> None:
    """
    Download the ViT-B-16 weights and save them to the specified path.

    Args:
        save_path (str): Path to save the weights

    Returns:
        None
    """
    # Create a ViT-B-16 model with default weights
    vit = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)

    # Save the state dict
    torch.save(vit.state_dict(), save_path)

    # Validation message and file size
    print(f"ViT-B-16 weights downloaded and saved to {save_path}")
    print(f"File size: {os.path.getsize(save_path) / (1024 * 1024):.2f} MB")


if __name__ == "__main__":
    download_vit_weights()
