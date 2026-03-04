"""
Enhanced SPEED Pose Visualization System
Based on ESA speed-utils but adapted for prediction vs ground truth comparison
"""

import numpy as np
import json
import os
from matplotlib import pyplot as plt

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
from typing import Dict, Optional

# Import your existing dataset class
from src.datasets import SPEEDDataset


class Camera:
    """Utility class for accessing camera parameters (from ESA speed-utils)"""

    fx = 0.0176  # focal length[m]
    fy = 0.0176  # focal length[m]
    nu = 1920  # number of horizontal[pixels]
    nv = 1200  # number of vertical[pixels]
    ppx = 5.86e-6  # horizontal pixel pitch[m / pixel]
    ppy = ppx  # vertical pixel pitch[m / pixel]
    fpx = fx / ppx  # horizontal focal length[pixels]
    fpy = fy / ppy  # vertical focal length[pixels]
    k = [[fpx, 0, nu / 2], [0, fpy, nv / 2], [0, 0, 1]]
    K = np.array(k)


def quat2dcm(q):
    """Computing direction cosine matrix from quaternion (from ESA speed-utils)"""

    # normalizing quaternion
    q = q / np.linalg.norm(q)

    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]

    dcm = np.zeros((3, 3))

    dcm[0, 0] = 2 * q0**2 - 1 + 2 * q1**2
    dcm[1, 1] = 2 * q0**2 - 1 + 2 * q2**2
    dcm[2, 2] = 2 * q0**2 - 1 + 2 * q3**2

    dcm[0, 1] = 2 * q1 * q2 + 2 * q0 * q3
    dcm[0, 2] = 2 * q1 * q3 - 2 * q0 * q2

    dcm[1, 0] = 2 * q1 * q2 - 2 * q0 * q3
    dcm[1, 2] = 2 * q2 * q3 + 2 * q0 * q1

    dcm[2, 0] = 2 * q1 * q3 + 2 * q0 * q2
    dcm[2, 1] = 2 * q2 * q3 - 2 * q0 * q1

    return dcm


def project_axes(q, r, camera=None):
    """Project coordinate axes to image frame (adapted from ESA speed-utils)"""

    if camera is None:
        camera = Camera()

    # reference points in satellite frame for drawing axes
    p_axes = np.array([[0, 0, 0, 1], [1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    points_body = np.transpose(p_axes)

    # transformation to camera frame
    pose_mat = np.hstack((np.transpose(quat2dcm(q)), np.expand_dims(r, 1)))
    p_cam = np.dot(pose_mat, points_body)

    # getting homogeneous coordinates
    points_camera_frame = p_cam / p_cam[2]

    # projection to image plane
    points_image_plane = camera.K.dot(points_camera_frame)

    x, y = (points_image_plane[0], points_image_plane[1])
    return x, y


def quaternion_distance(q1, q2):
    """Compute geodesic distance between two quaternions"""
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)

    # Compute the inner product
    inner_product = np.abs(np.dot(q1, q2))
    inner_product = np.clip(inner_product, 0.0, 1.0)

    # Geodesic distance
    distance = 2 * np.arccos(inner_product)
    return np.degrees(distance)


def translation_error(t_pred, t_gt):
    """Compute translation error in meters"""
    return np.linalg.norm(np.array(t_pred) - np.array(t_gt))


class PoseVisualizationSystem:
    """Enhanced visualization system for SPEED pose estimation results"""

    def __init__(
        self,
        results_path: str,
        dataset: Optional[SPEEDDataset] = None,
        dataset_name: str = "SPEED",
    ):
        """
        Initialize the visualization system

        Args:
            results_path: Path to results.json file
            dataset: Optional SPEEDDataset instance for accessing images
            dataset_name: Dataset type ("SPEED", "SPEED_PLUS", "SPEED_PLUS_SYNTHETIC")
        """
        self.results_path = results_path
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.camera = Camera()

        # Load results
        with open(results_path, "r") as f:
            self.results = json.load(f)

        # Load filename mapping from test.json if available
        self.filename_map = self._load_filename_mapping()

        # Compute error metrics
        self.compute_error_metrics()

    def _load_filename_mapping(self) -> Dict[str, str]:
        """Load mapping from sample ID to filename from test.json"""
        filename_map = {}

        # Map dataset names to their folders (relative to project root)
        dataset_folders = {
            "SPEED": "./SPEED_FIXED",
            "SPEED_PLUS": "./SPEED_PLUS_FIXED",
            "SPEED_PLUS_SYNTHETIC": "./SPEED_PLUS_SYNTHETIC_FIXED",
        }

        folder = dataset_folders.get(self.dataset_name, "./SPEED_FIXED")

        # Try to find test.json for the specified dataset
        possible_paths = [
            f"{folder}/test.json",
            os.path.join(os.path.dirname(self.results_path), "test.json"),
            "test.json",
            os.path.join("data", "test.json"),
        ]

        for test_path in possible_paths:
            try:
                with open(test_path, "r") as f:
                    test_data = json.load(f)
                    for idx, item in enumerate(test_data):
                        if "filename" in item:
                            filename_map[str(idx)] = item["filename"]
                    print(f"Loaded {len(filename_map)} filename mappings from {test_path}")
                    break
            except (FileNotFoundError, json.JSONDecodeError) as e:
                print(f"Could not load {test_path}: {e}")
                continue

        return filename_map

    def get_filename(self, sample_id: str) -> str:
        """Get filename for a sample ID, fallback to sample ID if not found"""
        return self.filename_map.get(sample_id, f"sample_{sample_id}.jpg")

    def compute_error_metrics(self):
        """Compute error metrics for all samples"""
        self.error_metrics = {}

        for sample_id, data in self.results.items():
            # Skip non-sample entries (like 'metrics' summary)
            if (
                not isinstance(data, dict)
                or "prediction" not in data
                or "ground_truth" not in data
            ):
                continue

            pred = data["prediction"]
            gt = data["ground_truth"]

            trans_error = translation_error(pred["translation"], gt["translation"])
            rot_error = quaternion_distance(np.array(pred["rotation"]), np.array(gt["rotation"]))

            # Calculate relative translation error (normalized by distance to satellite)
            distance_to_sat = np.linalg.norm(gt["translation"])
            rel_trans_error = trans_error / distance_to_sat if distance_to_sat > 0 else trans_error

            # Calculate SPEED score (relative translation + rotation in radians)
            # Convert rotation error from degrees to radians for SPEED score
            rot_error_rad = np.radians(rot_error)
            speed_score = rel_trans_error + rot_error_rad

            self.error_metrics[sample_id] = {
                "translation_error_m": trans_error,
                "rotation_error_deg": rot_error,
                "relative_translation_error": rel_trans_error,
                "rotation_error_rad": rot_error_rad,
                "speed_score": speed_score,
                "distance_to_satellite": distance_to_sat,
            }

    def visualize_sample_comparison(self, sample_id: str, ax=None, show_image=True):
        """
        Visualize prediction vs ground truth for a single sample

        Args:
            sample_id: Sample ID to visualize
            ax: Matplotlib axis (optional)
            show_image: Whether to show the actual satellite image
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        # Get data
        data = self.results[sample_id]
        pred = data["prediction"]
        gt = data["ground_truth"]

        # Show image if requested - ALWAYS load original full-resolution images
        if show_image:
            image_loaded = False

            # Load original full-resolution image directly from filesystem
            try:
                from PIL import Image as PILImage

                filename = self.get_filename(sample_id)

                # Map dataset names to their folders (relative to project root)
                dataset_folders = {
                    "SPEED": "./SPEED_FIXED",
                    "SPEED_PLUS": "./SPEED_PLUS_FIXED",
                    "SPEED_PLUS_SYNTHETIC": "./SPEED_PLUS_SYNTHETIC_FIXED",
                }

                folder = dataset_folders.get(self.dataset_name, "./SPEED_FIXED")

                # Try image paths for the specified dataset
                possible_paths = [
                    f"{folder}/images/test/{filename}",
                    f"images/test/{filename}",
                    filename,
                    # Also try without extension and with different extensions
                    f"{folder}/images/test/{filename.replace('.jpg', '.png')}",
                    f"{folder}/images/test/{filename.replace('.png', '.jpg')}",
                ]

                for img_path in possible_paths:
                    if os.path.exists(img_path):
                        image = PILImage.open(img_path)
                        # Convert to RGB if needed to avoid color issues
                        if image.mode != "RGB":
                            image = image.convert("RGB")
                        ax.imshow(image, cmap="gray" if len(image.getbands()) == 1 else None)
                        ax.set_xlim(0, image.width)
                        ax.set_ylim(image.height, 0)
                        image_loaded = True
                        print(f"Loaded original image: {img_path} ({image.width}x{image.height})")
                        break

                if not image_loaded:
                    print(
                        f"Warning: Image not found for sample {sample_id} (filename: {filename})"
                    )
                    print(f"Tried paths: {possible_paths}")

            except Exception as e:
                print(f"Could not load image: {e}")

            # Fallback: Set default limits for original SPEED camera resolution
            if not image_loaded:
                ax.set_xlim(0, 1920)  # Original SPEED camera width
                ax.set_ylim(1200, 0)  # Original SPEED camera height
                ax.set_facecolor("lightgray")
                print("Using default SPEED camera resolution (1920x1200)")
        else:
            # Not showing image, just set background to original resolution
            ax.set_xlim(0, 1920)
            ax.set_ylim(1200, 0)
            ax.set_facecolor("lightgray")

        # Define colors for ground truth and prediction with high contrast
        gt_colors = {"X": "#FF0000", "Y": "#00FF00", "Z": "#0000FF"}  # Bright RGB
        pred_colors = {"X": "#FF69B4", "Y": "#ADFF2F", "Z": "#00BFFF"}  # Bright alternative colors

        # Project and draw axes for ground truth and prediction
        try:
            # Project using original camera parameters (should match image resolution now)
            gt_x, gt_y = project_axes(np.array(gt["rotation"]), np.array(gt["translation"]))
            pred_x, pred_y = project_axes(
                np.array(pred["rotation"]), np.array(pred["translation"])
            )

            # Get image bounds
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            print(f"\nSample {sample_id}: Image bounds: x={xlim}, y={ylim}")
            print(f"GT quaternion: {gt['rotation']}")
            print(f"GT translation: {gt['translation']}")
            print(f"Pred quaternion: {pred['rotation']}")
            print(f"Pred translation: {pred['translation']}")
            print(f"GT projection: x={gt_x}, y={gt_y}")
            print(f"Pred projection: x={pred_x}, y={pred_y}")

            # Scale arrow sizes based on image size - make them smaller
            img_width = xlim[1] - xlim[0]
            img_height = ylim[0] - ylim[1]  # Note: ylim is inverted
            max(min(img_width, img_height) * 0.012, 8)  # Smaller arrows

            # Simple overlapping approach - GT bright colors, Pred lighter colors
            gt_colors = ["red", "green", "blue"]
            pred_colors = [
                "orange",
                "lightgreen",
                "cyan",
            ]  # Lighter/different colors for prediction
            axis_names = ["X", "Y", "Z"]

            # Draw axes - simple clean lines
            for i in range(1, min(4, len(gt_x), len(gt_y))):
                axis_name = axis_names[i - 1]
                gt_color = gt_colors[i - 1]
                pred_color = pred_colors[i - 1]

                print(
                    f"  {axis_name}-axis: GT({gt_x[i]:.1f}, {gt_y[i]:.1f}), Pred({pred_x[i]:.1f}, {pred_y[i]:.1f})"
                )

                # Ground truth axis - solid lines (bright colors, thicker)
                ax.annotate(
                    "",
                    xy=(gt_x[i], gt_y[i]),
                    xytext=(gt_x[0], gt_y[0]),
                    arrowprops=dict(
                        arrowstyle="->", lw=5, color=gt_color, alpha=0.9
                    ),  # Thicker arrows
                )

                # Prediction axis - dashed lines with dots spaced further apart (different lighter colors)
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

            # Check if the poses look reasonable
            gt_center = (gt_x[0], gt_y[0])
            pred_center = (pred_x[0], pred_y[0])
            center_dist = (
                (gt_center[0] - pred_center[0]) ** 2 + (gt_center[1] - pred_center[1]) ** 2
            ) ** 0.5
            print(f"  Pose center distance: {center_dist:.1f} pixels")
            print(f"  GT center: ({gt_center[0]:.1f}, {gt_center[1]:.1f})")
            print(f"  Pred center: ({pred_center[0]:.1f}, {pred_center[1]:.1f})")

        except Exception as e:
            print(f"Error drawing axes for sample {sample_id}: {e}")
            # Draw a simple marker at the center as fallback
            ax.plot(
                ax.get_xlim()[1] // 2, ax.get_ylim()[0] // 2, "ro", markersize=10, label="Center"
            )

        # Add error information
        metrics = self.error_metrics[sample_id]
        filename = self.get_filename(sample_id)
        error_text = f"{filename}\n"
        error_text += f"Trans Error: {metrics['translation_error_m']:.3f}m\n"
        error_text += f"Rot Error: {metrics['rotation_error_deg']:.1f}°\n"
        error_text += f"SPEED Score: {metrics['speed_score']:.3f}\n"
        error_text += f"Distance: {metrics['distance_to_satellite']:.1f}m"

        ax.text(
            0.02,
            0.98,
            error_text,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
        )

        ax.set_title(f"{filename}")

        # Create simple legend with color distinction
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D([0], [0], color="red", lw=3, label="GT X-axis"),
            Line2D([0], [0], color="green", lw=3, label="GT Y-axis"),
            Line2D([0], [0], color="blue", lw=3, label="GT Z-axis"),
            Line2D([0], [0], color="orange", lw=3, linestyle="--", label="Pred X-axis"),
            Line2D([0], [0], color="lightgreen", lw=3, linestyle="--", label="Pred Y-axis"),
            Line2D([0], [0], color="cyan", lw=3, linestyle="--", label="Pred Z-axis"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=8, framealpha=0.9)

        return ax

    def create_error_dashboard(self):
        """Create an interactive dashboard showing error analysis"""

        # Prepare data for plotting
        sample_ids = list(self.error_metrics.keys())
        trans_errors = [self.error_metrics[sid]["translation_error_m"] for sid in sample_ids]
        rot_errors = [self.error_metrics[sid]["rotation_error_deg"] for sid in sample_ids]
        distances = [self.error_metrics[sid]["distance_to_satellite"] for sid in sample_ids]

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Translation Error Distribution",
                "Rotation Error Distribution",
                "Error vs Distance",
                "Translation vs Rotation Error",
            ),
            specs=[
                [{"type": "histogram"}, {"type": "histogram"}],
                [{"type": "scatter"}, {"type": "scatter"}],
            ],
        )

        # Translation error histogram
        fig.add_trace(
            go.Histogram(
                x=trans_errors,
                nbinsx=20,
                name="Translation Error (m)",
                marker_color="blue",
                opacity=0.7,
            ),
            row=1,
            col=1,
        )

        # Rotation error histogram
        fig.add_trace(
            go.Histogram(
                x=rot_errors,
                nbinsx=20,
                name="Rotation Error (deg)",
                marker_color="red",
                opacity=0.7,
            ),
            row=1,
            col=2,
        )

        # Error vs distance
        fig.add_trace(
            go.Scatter(
                x=distances,
                y=trans_errors,
                mode="markers",
                name="Trans Error vs Distance",
                marker=dict(color="blue"),
                text=[f"Sample {sid}" for sid in sample_ids],
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=distances,
                y=rot_errors,
                mode="markers",
                name="Rot Error vs Distance",
                marker=dict(color="red"),
                text=[f"Sample {sid}" for sid in sample_ids],
                yaxis="y2",
            ),
            row=2,
            col=1,
        )

        # Translation vs rotation error
        fig.add_trace(
            go.Scatter(
                x=trans_errors,
                y=rot_errors,
                mode="markers",
                name="Trans vs Rot Error",
                text=[f"Sample {sid}" for sid in sample_ids],
            ),
            row=2,
            col=2,
        )

        # Update layout
        fig.update_layout(
            height=800, title_text="Pose Estimation Error Analysis Dashboard", showlegend=True
        )

        # Update axis labels
        fig.update_xaxes(title_text="Translation Error (m)", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)

        fig.update_xaxes(title_text="Rotation Error (deg)", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=1, col=2)

        fig.update_xaxes(title_text="Distance to Satellite (m)", row=2, col=1)
        fig.update_yaxes(title_text="Error", row=2, col=1)

        fig.update_xaxes(title_text="Translation Error (m)", row=2, col=2)
        fig.update_yaxes(title_text="Rotation Error (deg)", row=2, col=2)

        return fig

    def create_3d_trajectory_plot(self):
        """Create 3D plot showing predicted vs ground truth trajectories"""

        # Extract positions
        pred_positions = []
        gt_positions = []
        sample_ids = []

        for sample_id, data in self.results.items():
            # Skip non-sample entries (like 'metrics' summary)
            if (
                not isinstance(data, dict)
                or "prediction" not in data
                or "ground_truth" not in data
            ):
                continue

            pred_positions.append(data["prediction"]["translation"])
            gt_positions.append(data["ground_truth"]["translation"])
            sample_ids.append(f"Sample {sample_id}")

        pred_positions = np.array(pred_positions)
        gt_positions = np.array(gt_positions)

        # Create 3D scatter plot
        fig = go.Figure()

        # Ground truth positions
        fig.add_trace(
            go.Scatter3d(
                x=gt_positions[:, 0],
                y=gt_positions[:, 1],
                z=gt_positions[:, 2],
                mode="markers",
                marker=dict(size=8, color="blue", opacity=0.8),
                name="Ground Truth",
                text=sample_ids,
                hovertemplate="GT Position<br>X: %{x:.2f}m<br>Y: %{y:.2f}m<br>Z: %{z:.2f}m<br>%{text}<extra></extra>",
            )
        )

        # Predicted positions
        fig.add_trace(
            go.Scatter3d(
                x=pred_positions[:, 0],
                y=pred_positions[:, 1],
                z=pred_positions[:, 2],
                mode="markers",
                marker=dict(size=8, color="red", opacity=0.8),
                name="Predictions",
                text=sample_ids,
                hovertemplate="Pred Position<br>X: %{x:.2f}m<br>Y: %{y:.2f}m<br>Z: %{z:.2f}m<br>%{text}<extra></extra>",
            )
        )

        # Add lines connecting predictions to ground truth
        for i in range(len(sample_ids)):
            fig.add_trace(
                go.Scatter3d(
                    x=[gt_positions[i, 0], pred_positions[i, 0]],
                    y=[gt_positions[i, 1], pred_positions[i, 1]],
                    z=[gt_positions[i, 2], pred_positions[i, 2]],
                    mode="lines",
                    line=dict(color="gray", width=2),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        # Update layout
        fig.update_layout(
            title="3D Satellite Position Trajectories",
            scene=dict(
                xaxis_title="X (m)",
                yaxis_title="Y (m)",
                zaxis_title="Z (m)",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            ),
            height=700,
        )

        return fig

    def generate_summary_report(self) -> Dict:
        """Generate summary statistics of the pose estimation results"""

        trans_errors = [
            self.error_metrics[sid]["translation_error_m"] for sid in self.error_metrics.keys()
        ]
        rot_errors = [
            self.error_metrics[sid]["rotation_error_deg"] for sid in self.error_metrics.keys()
        ]

        summary = {
            "total_samples": len(self.results),
            "translation_error": {
                "mean": np.mean(trans_errors),
                "std": np.std(trans_errors),
                "median": np.median(trans_errors),
                "min": np.min(trans_errors),
                "max": np.max(trans_errors),
                "percentile_95": np.percentile(trans_errors, 95),
            },
            "rotation_error": {
                "mean": np.mean(rot_errors),
                "std": np.std(rot_errors),
                "median": np.median(rot_errors),
                "min": np.min(rot_errors),
                "max": np.max(rot_errors),
                "percentile_95": np.percentile(rot_errors, 95),
            },
        }

        return summary

    def create_sample_grid(self, n_samples=8, cols=4):
        """Create a grid showing multiple sample comparisons"""

        rows = n_samples // cols
        fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 4))
        axes = axes.flatten() if n_samples > 1 else [axes]

        sample_ids = list(self.error_metrics.keys())[
            :n_samples
        ]  # Use error_metrics keys to skip 'metrics'

        for i, sample_id in enumerate(sample_ids):
            self.visualize_sample_comparison(
                sample_id, ax=axes[i], show_image=True
            )  # Enable image display
            filename = self.get_filename(sample_id)
            axes[i].set_title(filename, fontsize=10)  # Show filename instead of sample ID
            axes[i].legend().set_visible(False)  # Hide individual legends

        # Hide unused subplots
        for i in range(n_samples, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        return fig

    def get_best_worst_samples(self):
        """Get best and worst performing samples with filenames"""
        trans_errors = {
            sid: metrics["translation_error_m"] for sid, metrics in self.error_metrics.items()
        }
        rot_errors = {
            sid: metrics["rotation_error_deg"] for sid, metrics in self.error_metrics.items()
        }

        # Best and worst translation
        best_trans = min(trans_errors, key=trans_errors.get)
        worst_trans = max(trans_errors, key=trans_errors.get)

        # Best and worst rotation
        best_rot = min(rot_errors, key=rot_errors.get)
        worst_rot = max(rot_errors, key=rot_errors.get)

        return {
            "best_translation": {
                "sample_id": best_trans,
                "filename": self.get_filename(best_trans),
                "error": trans_errors[best_trans],
            },
            "worst_translation": {
                "sample_id": worst_trans,
                "filename": self.get_filename(worst_trans),
                "error": trans_errors[worst_trans],
            },
            "best_rotation": {
                "sample_id": best_rot,
                "filename": self.get_filename(best_rot),
                "error": rot_errors[best_rot],
            },
            "worst_rotation": {
                "sample_id": worst_rot,
                "filename": self.get_filename(worst_rot),
                "error": rot_errors[worst_rot],
            },
        }


# Convenience functions for quick usage
def quick_visualize_sample(
    results_path: str, sample_id: str, dataset=None, dataset_name: str = "SPEED"
):
    """Quick function to visualize a single sample"""
    viz = PoseVisualizationSystem(results_path, dataset, dataset_name=dataset_name)
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    viz.visualize_sample_comparison(sample_id, ax)
    plt.show()
    return fig


def create_dashboard(results_path: str, output_html: str = None, dataset_name: str = "SPEED"):
    """Quick function to create and optionally export dashboard"""
    viz = PoseVisualizationSystem(results_path, dataset_name=dataset_name)

    # Show summary
    summary = viz.generate_summary_report()
    print("=== SPEED Pose Estimation Results Summary ===")
    print(f"Total Samples: {summary['total_samples']}")
    print(
        f"Translation Error - Mean: {summary['translation_error']['mean']:.4f}m, Std: {summary['translation_error']['std']:.4f}m"
    )
    print(
        f"Rotation Error - Mean: {summary['rotation_error']['mean']:.2f}°, Std: {summary['rotation_error']['std']:.2f}°"
    )
    print("=" * 50)

    # Create plots
    dashboard_fig = viz.create_error_dashboard()
    trajectory_fig = viz.create_3d_trajectory_plot()

    # Try to show plots, but handle missing dependencies gracefully
    try:
        dashboard_fig.show()
        trajectory_fig.show()
    except (ValueError, ImportError) as e:
        print(f"Note: Interactive plots cannot be displayed in this environment ({e})")
        print("Plots will be available in the exported HTML file.")

    return viz


def compare_samples_grid(
    results_path: str, n_samples: int = 8, dataset=None, dataset_name: str = "SPEED"
):
    """Quick function to create sample comparison grid"""
    viz = PoseVisualizationSystem(results_path, dataset=dataset, dataset_name=dataset_name)
    fig = viz.create_sample_grid(n_samples)
    plt.show()
    return fig
