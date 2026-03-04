#!/usr/bin/env python3
"""
Visualization script for the data augmentation pipeline including style augmentation.
Shows the effect of domain gap augmentations and neural style transfer on training images.
"""

import argparse
import os
import sys
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import random
from src.datasets import SPEEDDataset
from styleaug import StyleAugmentor

STYLE_AUG_AVAILABLE = True


def denormalize_imagenet_tensor(tensor):
    """Denormalize ImageNet normalized tensor back to [0,1] for visualization"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    # Denormalize: x * std + mean
    denormalized = tensor.cpu() * std + mean
    # Clamp to [0, 1] range
    return torch.clamp(denormalized, 0, 1)


def tensor_to_numpy_image(tensor):
    """Convert tensor [C, H, W] in [0,1] range to numpy [H, W, C] for display"""
    if tensor.dim() == 4:
        tensor = tensor[0]  # Take first image from batch
    return tensor.permute(1, 2, 0).cpu().numpy()


def visualize_augmentation_pipeline(dataset_root_dir, num_samples=4, device="cuda"):
    """
    Visualize the complete augmentation pipeline

    Args:
        dataset_root_dir: Path to dataset
        num_samples: Number of sample images to show
        device: Device for style augmentation
    """

    print("Setting up augmentation pipeline visualization...")

    # Create mock arguments for different augmentation scenarios
    scenarios = [
        {
            "name": "No Augmentation",
            "args": {
                "no_pixel_augmentation": True,
                "no_spatial_augmentation": True,
                "domain_gap_pixel_augmentation": False,
                "do_style_aug": False,
                "no_translation_compensation": False,
                "no_rotation_compensation": False,
                "new_Z": False,
            },
        },
        {
            "name": "Standard Pixel Augmentation",
            "args": {
                "no_pixel_augmentation": False,
                "no_spatial_augmentation": True,
                "domain_gap_pixel_augmentation": False,
                "do_style_aug": False,
                "no_translation_compensation": False,
                "no_rotation_compensation": False,
                "new_Z": False,
            },
        },
        {
            "name": "Domain Gap Augmentation",
            "args": {
                "no_pixel_augmentation": False,
                "no_spatial_augmentation": True,
                "domain_gap_pixel_augmentation": True,
                "do_style_aug": False,
                "no_translation_compensation": False,
                "no_rotation_compensation": False,
                "new_Z": False,
            },
        },
    ]

    # Add style augmentation scenario if available
    if STYLE_AUG_AVAILABLE:
        scenarios.append(
            {
                "name": "Domain Gap + Style Augmentation",
                "args": {
                    "no_pixel_augmentation": False,
                    "no_spatial_augmentation": True,
                    "domain_gap_pixel_augmentation": True,
                    "do_style_aug": True,
                    "no_translation_compensation": False,
                    "no_rotation_compensation": False,
                    "new_Z": False,
                },
            }
        )

        # Initialize StyleAugmentor for style aug scenario
        try:
            style_augmentor = StyleAugmentor().to(device)
            print(f"StyleAugmentor initialized on {device}")
        except Exception as e:
            print(f"Could not initialize StyleAugmentor: {e}")
            style_augmentor = None
            scenarios.pop()  # Remove style aug scenario
    else:
        print("StyleAugmentor not available")
        style_augmentor = None

    # Create datasets for each scenario
    datasets = {}
    for scenario in scenarios:
        args_namespace = argparse.Namespace(**scenario["args"])

        dataset = SPEEDDataset(
            dataset_root_dir=dataset_root_dir,
            split="train",
            rotation_format="quaternion",
            img_size=(224, 224),
            bbox_json_path=None,
            args=args_namespace,
            dataset_name="SPEED",
        )
        datasets[scenario["name"]] = dataset

    print(f"Created {len(datasets)} augmentation scenarios")

    # Select random samples to visualize
    base_dataset = datasets["No Augmentation"]
    sample_indices = random.sample(range(len(base_dataset)), min(num_samples, len(base_dataset)))

    # Create visualization
    fig, axes = plt.subplots(
        num_samples, len(scenarios), figsize=(4 * len(scenarios), 4 * num_samples)
    )
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    print(f"Processing {num_samples} samples with {len(scenarios)} augmentation types...")

    for sample_idx, data_idx in enumerate(sample_indices):
        for scenario_idx, (scenario_name, dataset) in enumerate(datasets.items()):
            # Get augmented sample
            image, translation, rotation, bbox = dataset[data_idx]

            # Handle different image formats
            if isinstance(image, torch.Tensor):
                if image.max() <= 1.0 and len(image.shape) == 3:
                    # Unnormalized [0,1] tensor format (from style aug pipeline)
                    if (
                        scenario_name == "Domain Gap + Style Augmentation"
                        and style_augmentor is not None
                    ):
                        # Apply style augmentation
                        try:
                            image_batch = image.unsqueeze(0).to(device)  # Add batch dim
                            styled_image = style_augmentor(image_batch, alpha=0.5)
                            # Apply final normalization
                            final_image = dataset.finalize_style_augmented_batch(styled_image)
                            # Denormalize for visualization
                            display_image = denormalize_imagenet_tensor(final_image[0])
                        except Exception as e:
                            print(f"Style augmentation failed for visualization: {e}")
                            # Fallback to original
                            final_image = dataset.finalize_style_augmented_batch(image_batch)
                            display_image = denormalize_imagenet_tensor(final_image[0])
                    else:
                        # Just normalize for other scenarios with unnormalized input
                        image_batch = image.unsqueeze(0).to(device)
                        final_image = dataset.finalize_style_augmented_batch(image_batch)
                        display_image = denormalize_imagenet_tensor(final_image[0])
                else:
                    # Already normalized ImageNet tensor - denormalize for display
                    display_image = denormalize_imagenet_tensor(image)

                # Convert to numpy for display
                display_image = tensor_to_numpy_image(display_image)
            else:
                # Numpy array - convert directly
                display_image = image
                if display_image.max() > 1.0:
                    display_image = display_image / 255.0

            # Plot the image
            axes[sample_idx, scenario_idx].imshow(display_image)
            axes[sample_idx, scenario_idx].set_title(
                f"{scenario_name}\nSample {sample_idx + 1}", fontsize=10
            )
            axes[sample_idx, scenario_idx].axis("off")

    plt.tight_layout()

    # Save visualization
    output_path = (
        Path(__file__).parent.parent / "other_utils" / "augmentation_pipeline_visualization.png"
    )
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Visualization saved to: {output_path}")

    plt.show()

    # Print summary
    print("\nAugmentation Pipeline Summary:")
    for scenario in scenarios:
        print(f"  • {scenario['name']}")
        if scenario["name"] == "Domain Gap + Style Augmentation":
            print("    → Traditional domain gap augmentations (blur, sun flare, noise)")
            print("    → Neural style transfer (artistic style variations)")
            print("    → ImageNet normalization")
        elif scenario["name"] == "Domain Gap Augmentation":
            print("    → Traditional domain gap augmentations (blur, sun flare, noise)")
            print("    → ImageNet normalization")
        elif scenario["name"] == "Standard Pixel Augmentation":
            print("    → Standard photometric augmentations (brightness, contrast)")
            print("    → ImageNet normalization")
        else:
            print("    → Only resize and ImageNet normalization")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize data augmentation pipeline")
    parser.add_argument(
        "--dataset_root_dir",
        type=str,
        default="SPEED_FIXED/",
        help="Root directory of the dataset",
    )
    parser.add_argument(
        "--num_samples", type=int, default=4, help="Number of sample images to visualize"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for style augmentation",
    )

    args = parser.parse_args()

    # Check if dataset exists
    if not os.path.exists(args.dataset_root_dir):
        print(f"Dataset directory not found: {args.dataset_root_dir}")
        print("Please specify correct --dataset_root_dir")
        sys.exit(1)

    print("Starting augmentation pipeline visualization")
    print(f"Dataset: {args.dataset_root_dir}")
    print(f"Samples: {args.num_samples}")
    print(f"Device: {args.device}")
    print(f"Style augmentation available: {STYLE_AUG_AVAILABLE}")
    print()

    visualize_augmentation_pipeline(
        dataset_root_dir=args.dataset_root_dir, num_samples=args.num_samples, device=args.device
    )
