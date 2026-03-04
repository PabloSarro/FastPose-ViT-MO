#!/usr/bin/env python3
"""
SPEED Pose Estimation Comparison Visualization Script

This script generates comparison visualizations showing the best, average (median),
and worst performing samples from SPEED pose estimation results.

Outputs:
- Composite image with 3-panel comparison (best/average/worst)
- Individual images for each sample with satellite imagery and pose axes
- All saved to images/speed_comparison/ directory

Usage:
    python generate_speed_comparison.py --results results.json --dataset SPEED_PLUS
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from visualization import PoseVisualizationSystem


def find_representative_samples(viz_system):
    """
    Find representative samples: best, median (average), and worst performing
    based on SPEED score (the official competition metric).
    For SPEED_PLUS, use second worst instead of worst.

    Args:
        viz_system: PoseVisualizationSystem instance

    Returns:
        dict: Contains sample_id, filename, and metrics for best/median/worst samples
    """
    speed_scores = {
        sid: metrics["speed_score"] for sid, metrics in viz_system.error_metrics.items()
    }

    # Sort by SPEED score (lower is better)
    sorted_scores = sorted(speed_scores.items(), key=lambda x: x[1])

    # Get best, worst, and median samples
    best_sample = sorted_scores[0]  # Lowest SPEED score (best performance)

    # For SPEED_PLUS, use second worst instead of worst
    if viz_system.dataset_name == "SPEED_PLUS":
        worst_sample = sorted_scores[-2]  # Second highest SPEED score
        print(f"Using second worst for SPEED_PLUS: {worst_sample}")
    else:
        worst_sample = sorted_scores[-1]  # Highest SPEED score (worst performance)

    median_idx = len(sorted_scores) // 2
    median_sample = sorted_scores[median_idx]  # Median SPEED score

    # Get complete metrics for each sample
    def get_sample_info(sample_tuple, performance_type):
        sample_id, speed_score = sample_tuple
        metrics = viz_system.error_metrics[sample_id]
        return {
            "sample_id": sample_id,
            "filename": viz_system.get_filename(sample_id),
            "speed_score": speed_score,
            "translation_error": metrics["translation_error_m"],
            "rotation_error": metrics["rotation_error_deg"],
            "type": performance_type,
        }

    # Return structured information with complete metrics
    return {
        "best": get_sample_info(best_sample, "Best"),
        "median": get_sample_info(median_sample, "Average"),
        "worst": get_sample_info(worst_sample, "Worst"),
    }


def create_composite_visualization(viz_system, samples, output_path):
    """
    Create a composite visualization with 3 panels showing best/average/worst samples.

    Args:
        viz_system: PoseVisualizationSystem instance
        samples: Dictionary with best/median/worst sample information
        output_path: Path to save the composite image
    """
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Process each sample type
    sample_order = ["best", "median", "worst"]
    colors = ["green", "orange", "red"]  # Color coding for performance

    for i, (sample_key, color) in enumerate(zip(sample_order, colors)):
        sample_info = samples[sample_key]
        ax = axes[i]

        # Visualize the sample comparison with image
        viz_system.visualize_sample_comparison(sample_info["sample_id"], ax=ax, show_image=True)

        # Enhanced title with complete performance metrics
        title = f"{sample_info['type']} Performance\n{sample_info['filename']}"
        title += f"\nSPEED Score: {sample_info['speed_score']:.3f}"
        title += f"\nTrans: {sample_info['translation_error']:.3f}m | Rot: {sample_info['rotation_error']:.3f}°"

        ax.set_title(title, fontsize=11, fontweight="bold", color=color)

        # Add colored border to distinguish performance levels
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)

    # Overall title and layout
    fig.suptitle(
        "SPEED Pose Estimation: Performance Comparison (Official SPEED Score)\n"
        f"Dataset: {viz_system.dataset_name} | Total Samples: {len(viz_system.error_metrics)}",
        fontsize=14,
        fontweight="bold",
    )

    # Add summary statistics with complete metrics
    speed_scores = [m["speed_score"] for m in viz_system.error_metrics.values()]
    trans_errors = [m["translation_error_m"] for m in viz_system.error_metrics.values()]
    rot_errors = [m["rotation_error_deg"] for m in viz_system.error_metrics.values()]

    stats_text = (
        f"SPEED Score: Mean={np.mean(speed_scores):.3f}, Median={np.median(speed_scores):.3f} | "
    )
    stats_text += f"Trans Error: Mean={np.mean(trans_errors):.3f}m | "
    stats_text += f"Rot Error: Mean={np.mean(rot_errors):.3f}°"

    fig.text(0.5, 0.02, stats_text, ha="center", fontsize=10, style="italic")

    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.1)

    # Save the composite image
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Composite visualization saved: {output_path}")

    return fig


def save_individual_visualizations(viz_system, samples, output_dir):
    """
    Save individual visualizations for each sample (best/average/worst).
    Clean version with only satellite image, pose axes, and legend.

    Args:
        viz_system: PoseVisualizationSystem instance
        samples: Dictionary with best/median/worst sample information
        output_dir: Directory to save individual images
    """
    individual_files = []

    for sample_key, sample_info in samples.items():
        # Create individual figure
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        # Get the sample data
        sample_id = sample_info["sample_id"]
        data = viz_system.results[sample_id]
        pred = data["prediction"]
        gt = data["ground_truth"]

        # Load and display the satellite image
        try:
            from PIL import Image as PILImage

            filename = viz_system.get_filename(sample_id)

            # Map dataset names to their folders (relative to project root)
            dataset_folders = {
                "SPEED": "./SPEED_FIXED",
                "SPEED_PLUS": "./SPEED_PLUS_FIXED",
                "SPEED_PLUS_SYNTHETIC": "./SPEED_PLUS_SYNTHETIC_FIXED",
            }

            folder = dataset_folders.get(viz_system.dataset_name, "./SPEED_FIXED")

            # Try image paths for the specified dataset
            possible_paths = [
                f"{folder}/images/test/{filename}",
                f"images/test/{filename}",
                filename,
                f"{folder}/images/test/{filename.replace('.jpg', '.png')}",
                f"{folder}/images/test/{filename.replace('.png', '.jpg')}",
            ]

            image_loaded = False
            for img_path in possible_paths:
                if os.path.exists(img_path):
                    image = PILImage.open(img_path)
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    ax.imshow(image, cmap="gray" if len(image.getbands()) == 1 else None)
                    ax.set_xlim(0, image.width)
                    ax.set_ylim(image.height, 0)
                    image_loaded = True
                    break

            if not image_loaded:
                # Fallback: Set default limits for original SPEED camera resolution
                ax.set_xlim(0, 1920)
                ax.set_ylim(1200, 0)
                ax.set_facecolor("lightgray")

        except Exception:
            # Fallback
            ax.set_xlim(0, 1920)
            ax.set_ylim(1200, 0)
            ax.set_facecolor("lightgray")

        # Project and draw pose axes
        try:
            from visualization import project_axes

            gt_x, gt_y = project_axes(np.array(gt["rotation"]), np.array(gt["translation"]))
            pred_x, pred_y = project_axes(
                np.array(pred["rotation"]), np.array(pred["translation"])
            )

            # Define colors
            gt_colors = ["red", "green", "blue"]
            pred_colors = ["orange", "lightgreen", "cyan"]
            axis_names = ["X", "Y", "Z"]

            # Draw axes
            for i in range(1, min(4, len(gt_x), len(gt_y))):
                axis_names[i - 1]
                gt_color = gt_colors[i - 1]
                pred_color = pred_colors[i - 1]

                # Ground truth axis - solid lines (thicker)
                ax.annotate(
                    "",
                    xy=(gt_x[i], gt_y[i]),
                    xytext=(gt_x[0], gt_y[0]),
                    arrowprops=dict(
                        arrowstyle="->", lw=5, color=gt_color, alpha=0.9
                    ),  # Thicker arrows
                )

                # Prediction axis - dashed lines (thicker and more spaced)
                if i < len(pred_x) and i < len(pred_y):
                    ax.annotate(
                        "",
                        xy=(pred_x[i], pred_y[i]),
                        xytext=(pred_x[0], pred_y[0]),
                        arrowprops=dict(
                            arrowstyle="->",
                            lw=5,  # Thicker arrows
                            color=pred_color,
                            alpha=1.0,
                            linestyle=(0, (2.5, 3)),  # Fine-tuned dashes (2.5px dash, 3px gap)
                        ),
                    )

        except Exception as e:
            print(f"Error drawing axes for sample {sample_id}: {e}")

        # Remove title, axis labels, ticks, and all legends for clean individual images
        ax.set_title("")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")

        # Remove spines (axis borders)
        for spine in ax.spines.values():
            spine.set_visible(False)

        plt.tight_layout()

        # Save individual image
        filename = f"{sample_key}_{sample_info['sample_id']}_{sample_info['filename']}"
        filepath = output_dir / f"{filename}.png"
        plt.savefig(filepath, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0)

        individual_files.append(filepath)
        print(f"Clean individual visualization saved: {filepath}")

        plt.close(fig)  # Close to free memory

    return individual_files


def create_output_directory(base_dir="images"):
    """
    Create the output directory structure if it doesn't exist.

    Args:
        base_dir: Base directory name (default: "images")

    Returns:
        Path: Path object for the speed_comparison directory
    """
    output_dir = Path(base_dir) / "speed_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def main():
    """Main function to orchestrate the visualization generation."""
    parser = argparse.ArgumentParser(
        description="Generate SPEED pose estimation comparison visualizations"
    )
    parser.add_argument(
        "--results",
        type=str,
        default="results.json",
        help="Path to results.json file (default: results.json)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["SPEED", "SPEED_PLUS", "SPEED_PLUS_SYNTHETIC"],
        default="SPEED_PLUS",
        help="Dataset type (default: SPEED_PLUS)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="images",
        help="Output directory for generated images (default: images)",
    )

    args = parser.parse_args()

    try:
        print("=" * 60)
        print("SPEED Pose Estimation Comparison Visualization")
        print("=" * 60)

        # Check if results file exists
        if not os.path.exists(args.results):
            raise FileNotFoundError(f"Results file not found: {args.results}")

        print(f"Loading results from: {args.results}")
        print(f"Dataset type: {args.dataset}")

        # Initialize visualization system
        viz_system = PoseVisualizationSystem(results_path=args.results, dataset_name=args.dataset)

        print(f"Loaded {len(viz_system.error_metrics)} samples for analysis")

        # Find representative samples
        print("Finding representative samples...")
        samples = find_representative_samples(viz_system)

        # Display sample information
        print("\nSelected samples (ranked by SPEED score):")
        for key, sample in samples.items():
            print(
                f"  {sample['type']}: {sample['filename']} "
                f"(ID: {sample['sample_id']}, SPEED Score: {sample['speed_score']:.4f})"
            )
            print(
                f"    Trans: {sample['translation_error']:.3f}m, Rot: {sample['rotation_error']:.3f}°"
            )

        # Create output directory
        output_dir = create_output_directory(args.output)
        print(f"\nOutput directory: {output_dir}")

        # Generate composite visualization
        print("\nGenerating composite visualization...")
        composite_path = output_dir / "speed_comparison_composite.png"
        composite_fig = create_composite_visualization(viz_system, samples, composite_path)
        plt.close(composite_fig)

        # Generate individual visualizations
        print("Generating individual visualizations...")
        individual_files = save_individual_visualizations(viz_system, samples, output_dir)

        # Summary
        print("\n" + "=" * 60)
        print("VISUALIZATION GENERATION COMPLETE")
        print("=" * 60)
        print(f"Generated {len(individual_files) + 1} images:")
        print(f"  - Composite: {composite_path}")
        for file_path in individual_files:
            print(f"  - Individual: {file_path}")
        print(f"\nAll images saved to: {output_dir}")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
