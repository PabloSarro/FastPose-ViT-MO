# This script plots the error (in meters or degrees) in function of the some variables

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from scipy.ndimage import gaussian_filter1d
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze and plot pose estimation errors from JSON results"
    )
    parser.add_argument(
        "--json_path", type=str, required=True, help="Path to the JSON results file"
    )
    return parser.parse_args()


def plot_errors(x_data, trans_errors, rot_errors, ax1, ax2, x_label):
    # Sort data for line plot
    sort_idx = np.argsort(x_data)
    x_sorted = x_data[sort_idx]
    trans_errors_sorted = trans_errors[sort_idx]
    rot_errors_sorted = rot_errors[sort_idx]

    # Calculate moving averages
    window = len(x_data) // 20  # Adjust window size as needed
    if window < 2:
        window = 2

    trans_avg = gaussian_filter1d(trans_errors_sorted, sigma=window / 4)
    rot_avg = gaussian_filter1d(rot_errors_sorted, sigma=window / 4)

    # Translation error plot
    ax1.scatter(x_data, trans_errors, c="blue", alpha=0.5, label="Data points")
    ax1.plot(x_sorted, trans_avg, "r-", linewidth=2, label="Moving average")
    ax1.set_xlabel(x_label)
    ax1.set_ylabel("Translation Error (m)")
    ax1.set_title(f"Translation Error vs {x_label}")
    ax1.grid(True)
    ax1.legend()

    # Rotation error plot
    ax2.scatter(x_data, rot_errors, c="blue", alpha=0.5, label="Data points")
    ax2.plot(x_sorted, rot_avg, "r-", linewidth=2, label="Moving average")
    ax2.set_xlabel(x_label)
    ax2.set_ylabel("Rotation Error (degrees)")
    ax2.set_title(f"Rotation Error vs {x_label}")
    ax2.grid(True)
    ax2.legend()


def main():
    args = parse_args()

    # Load the JSON file
    with open(args.json_path, "r") as f:
        data = json.load(f)

    # Calculate errors, depths, and horizontal distances
    # The horizontal distance is sqrt(X^2 + Y^2), where X and Y are the translation values which are not the depth
    translation_errors = []
    rotation_errors = []
    depths = []
    horizontal_distances = []

    for key in data.keys():
        if key == "metrics":
            break
        # Get translations and rotations
        pred_trans = np.array(data[key]["prediction"]["translation"])
        gt_trans = np.array(data[key]["ground_truth"]["translation"])
        pred_rot = np.array(data[key]["prediction"]["rotation"])
        gt_rot = np.array(data[key]["ground_truth"]["rotation"])

        # Calculate translation error (Euclidean distance)
        trans_error = np.linalg.norm(pred_trans - gt_trans)

        # Calculate rotation error (angular difference in degrees)
        pred_rot_mat = Rotation.from_quat(pred_rot)
        gt_rot_mat = Rotation.from_quat(gt_rot)
        relative_rot = gt_rot_mat.inv() * pred_rot_mat
        rot_error = np.degrees(relative_rot.magnitude())

        translation_errors.append(trans_error)
        rotation_errors.append(rot_error)
        depths.append(gt_trans[2])  # Z component is depth
        horizontal_distances.append(
            np.sqrt(gt_trans[0] ** 2 + gt_trans[1] ** 2)
        )  # Horizontal distance

    # Convert to numpy arrays
    depths = np.array(depths)
    translation_errors = np.array(translation_errors)
    rotation_errors = np.array(rotation_errors)
    horizontal_distances = np.array(horizontal_distances)

    # Create the plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Plot errors vs depth
    plot_errors(depths, translation_errors, rotation_errors, ax1, ax2, "Depth (m)")

    # Plot errors vs horizontal distance
    plot_errors(
        horizontal_distances,
        translation_errors,
        rotation_errors,
        ax3,
        ax4,
        "Horizontal Distance (m)",
    )

    # Print the mean and standard deviation of the errors
    print("Mean translation error:", np.mean(translation_errors))
    print("Mean rotation error:", np.mean(rotation_errors))
    print("Std translation error:", np.std(translation_errors))
    print("Std rotation error:", np.std(rotation_errors))

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
