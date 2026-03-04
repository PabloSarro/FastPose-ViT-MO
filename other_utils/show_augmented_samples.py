#!/usr/bin/env python3
"""
Show actual training samples with and without style augmentation.
"""

import argparse
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from src.datasets import SPEEDDataset
from styleaug import StyleAugmentor

STYLE_AUG_AVAILABLE = True


def show_training_samples(dataset_root_dir="SPEED_FIXED/", num_samples=2):
    """Show actual training samples with different augmentation settings"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"StyleAugmentor available: {STYLE_AUG_AVAILABLE}")

    # Create datasets with different augmentation settings
    scenarios = [
        ("No Augmentation", False, False),
        ("Domain Gap Only", True, False),
    ]

    if STYLE_AUG_AVAILABLE:
        scenarios.append(("Domain Gap + Style Aug", True, True))
        try:
            style_augmentor = StyleAugmentor().to(device)
            print("StyleAugmentor initialized")
        except Exception as e:
            print(f"StyleAugmentor failed: {e}")
            scenarios.pop()  # Remove style aug scenario
            style_augmentor = None
    else:
        style_augmentor = None

    print(f"Testing {len(scenarios)} scenarios")

    # Create figure
    fig, axes = plt.subplots(
        num_samples, len(scenarios), figsize=(4 * len(scenarios), 4 * num_samples)
    )
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for sample_idx in range(num_samples):
        for scenario_idx, (name, use_domain_gap, use_style_aug) in enumerate(scenarios):
            # Create args
            args = argparse.Namespace(
                no_pixel_augmentation=False,
                no_spatial_augmentation=True,
                domain_gap_pixel_augmentation=use_domain_gap,
                do_style_aug=use_style_aug,
                no_translation_compensation=False,
                no_rotation_compensation=False,
                new_Z=False,
            )

            # Create dataset
            dataset = SPEEDDataset(dataset_root_dir=dataset_root_dir, split="train", args=args)

            # Get sample
            image, translation, rotation, bbox = dataset[sample_idx]

            # Process image for display
            if isinstance(image, torch.Tensor):
                if image.max() <= 1.0 and use_style_aug and style_augmentor is not None:
                    # Apply style augmentation for visualization
                    try:
                        image_batch = image.unsqueeze(0).to(device)
                        styled_batch = style_augmentor(image_batch, alpha=0.5)
                        # Apply ImageNet normalization
                        final_batch = dataset.finalize_style_augmented_batch(styled_batch)
                        # Denormalize for display
                        display_image = denormalize_for_display(final_batch[0])
                    except Exception as e:
                        print(f"Style aug visualization failed: {e}")
                        final_batch = dataset.finalize_style_augmented_batch(image_batch)
                        display_image = denormalize_for_display(final_batch[0])
                else:
                    # Standard ImageNet normalized tensor - denormalize for display
                    display_image = denormalize_for_display(image)

                display_image = display_image.permute(1, 2, 0).cpu().numpy()
            else:
                display_image = image / 255.0 if image.max() > 1.0 else image

            # Plot
            axes[sample_idx, scenario_idx].imshow(display_image)
            axes[sample_idx, scenario_idx].set_title(f"{name}\nSample {sample_idx + 1}")
            axes[sample_idx, scenario_idx].axis("off")

    plt.tight_layout()
    output_path = Path(__file__).parent.parent / "other_utils" / "training_samples_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Sample comparison saved to: {output_path}")
    plt.show()


def denormalize_for_display(tensor):
    """Denormalize ImageNet tensor for display"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    denormalized = tensor.cpu() * std + mean
    return torch.clamp(denormalized, 0, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root_dir", default="SPEED_FIXED/", help="Dataset root directory"
    )
    parser.add_argument("--num_samples", type=int, default=2, help="Number of samples to show")

    args = parser.parse_args()
    show_training_samples(args.dataset_root_dir, args.num_samples)
