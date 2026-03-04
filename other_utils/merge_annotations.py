# This script merges bounding box annotations for SPEED_PLUS datasets
# Automatically determines dataset paths based on dataset type:
#   - synthetic: ../SPEED_PLUS_SYNTHETIC_FIXED/
#   - standard: ../SPEED_PLUS_FIXED/
# Expected prediction directory naming:
#   - SPEED_PLUS_SYNTHETIC: speedplus_synthetic_{train,val,test}
#   - SPEED_PLUS_FIXED: speedplus_{train,val,test}
# Creates bbox annotation files compatible with the training pipeline

import json
import argparse
from pathlib import Path


MIN_X, MAX_X = 0, 1920
MIN_Y, MAX_Y = 0, 1200


def clamp(value, lower, upper):
    """Clamp value to the inclusive range [lower, upper]."""
    return max(lower, min(value, upper))


def load_predictions_file(file_path):
    """Load a predictions JSON file and extract image->bbox mappings."""
    with open(file_path, "r") as f:
        data = json.load(f)

    annotations = {}

    # Extract predictions from the file
    predictions = data.get("predictions", [])

    for prediction in predictions:
        image_name = prediction.get("image_name")
        detections = prediction.get("detections", {})
        boxes = detections.get("boxes_xyxy", [])

        # Only process if we have exactly one detection (matching the target format)
        if image_name and len(boxes) == 1:
            box = boxes[0]
            x1 = clamp(int(box["x1"]), MIN_X, MAX_X)
            y1 = clamp(int(box["y1"]), MIN_Y, MAX_Y)
            x2 = clamp(int(box["x2"]), MIN_X, MAX_X)
            y2 = clamp(int(box["y2"]), MIN_Y, MAX_Y)

            annotations[image_name] = {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }

    return annotations


def create_bbox_annotations_from_file(input_file_path: str, output_file_path: str = None) -> None:
    """
    Create bbox annotations file from a single prediction file

    Args:
        input_file_path (str): Path to the input prediction JSON file
        output_file_path (str): Path for the output annotation file (optional)

    Returns:
        None
    """
    input_path = Path(input_file_path)

    if not input_path.exists():
        print(f"Error: Input file {input_path} does not exist")
        return

    # Set default output path if not provided
    if output_file_path is None:
        output_file_path = (
            input_path.parent / f"bbox_annotations_{input_path.stem.split('_')[-1]}.json"
        )
    else:
        output_file_path = Path(output_file_path)

    print(f"Processing file: {input_path}")

    try:
        # Load annotations from the single file
        annotations = load_predictions_file(input_path)
        print(f"Found {len(annotations)} annotations")

        if annotations:
            print(f"\nWriting {len(annotations)} total annotations to {output_file_path}")

            with open(output_file_path, "w") as f:
                json.dump(annotations, f, indent=4)

            print("Bbox annotations file created successfully!")

            # Print some statistics
            print("\nStatistics:")
            print(f"  Total unique images: {len(annotations)}")
            print(f"  Input file: {input_path}")
            print(f"  Output file: {output_file_path}")

            # Show a few examples
            print("\nFirst 3 entries:")
            for i, (image_name, bbox) in enumerate(annotations.items()):
                if i >= 3:
                    break
                print(f"  {image_name}: {bbox}")
        else:
            print("\nWarning: No annotations found. No output file created.")
            print("Make sure the input file contains valid prediction data.")

    except Exception as e:
        print(f"Error processing {input_path}: {e}")


def create_bbox_annotations(dataset_type: str) -> None:
    """
    Create bbox annotations file for SPEED_PLUS datasets

    Args:
        dataset_type (str): Type of dataset ('synthetic' or 'standard')

    Returns:
        None
    """

    # Automatically determine dataset path based on type
    base_dir = Path("./")  # From root directory

    if dataset_type == "synthetic":
        dataset_path = base_dir / "SPEED_PLUS_SYNTHETIC_FIXED"
    elif dataset_type == "standard":
        dataset_path = base_dir / "SPEED_PLUS_FIXED"
    else:
        print(f"Error: Unknown dataset type '{dataset_type}'. Use 'synthetic' or 'standard'")
        return

    if not dataset_path.exists():
        print(f"Error: Dataset path {dataset_path} does not exist")
        print(f"Make sure the {dataset_path.name} directory exists in the parent directory")
        return

    # Define source directories based on dataset type
    if dataset_type == "synthetic":
        # For SPEED_PLUS_SYNTHETIC: look in results directories with "synthetic" prefix
        base_dir = dataset_path.parent
        results_dir = base_dir / "results"

        source_dirs = [
            results_dir / "speedplus_synthetic_test",
            results_dir / "speedplus_synthetic_train",
            results_dir / "speedplus_synthetic_val",
        ]

        output_file = dataset_path / "speed_plus_synthetic_bbox_annotations.json"

    elif dataset_type == "standard":
        # For SPEED_PLUS_FIXED: look in results directories with standard naming
        base_dir = dataset_path.parent
        results_dir = base_dir / "results"

        source_dirs = [
            results_dir / "speedplus_test",
            results_dir / "speedplus_train",
            results_dir / "speedplus_val",
        ]

        output_file = dataset_path / "speed_plus_bbox_annotations.json"

    else:
        print(f"Error: Unknown dataset type '{dataset_type}'. Use 'synthetic' or 'standard'")
        return

    # Merged annotations dictionary
    merged_annotations = {}

    # Process each directory
    for source_dir in source_dirs:
        if not source_dir.exists():
            print(f"Skipping non-existent directory: {source_dir}")
            continue

        print(f"Processing directory: {source_dir}")

        # Look for JSON files in the directory
        json_files = list(source_dir.glob("*.json"))

        if not json_files:
            print(f"  No JSON files found in {source_dir}")
            continue

        for json_file in json_files:
            print(f"  Reading file: {json_file.name}")
            try:
                annotations = load_predictions_file(json_file)
                print(f"    Found {len(annotations)} annotations")

                # Merge annotations (later files will overwrite earlier ones if same image name)
                merged_annotations.update(annotations)

            except Exception as e:
                print(f"    Error processing {json_file}: {e}")

    # Write merged annotations
    if merged_annotations:
        print(f"\nWriting {len(merged_annotations)} total annotations to {output_file}")

        with open(output_file, "w") as f:
            json.dump(merged_annotations, f, indent=4)

        print("Bbox annotations file created successfully!")

        # Print some statistics
        print("\nStatistics:")
        print(f"  Total unique images: {len(merged_annotations)}")
        print(f"  Output file: {output_file}")

        # Show a few examples
        print("\nFirst 3 entries:")
        for i, (image_name, bbox) in enumerate(merged_annotations.items()):
            if i >= 3:
                break
            print(f"  {image_name}: {bbox}")
    else:
        print("\nWarning: No annotations found. No output file created.")
        print("Make sure the results directories contain valid prediction JSON files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create bbox annotations file for SPEED_PLUS datasets from prediction results"
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        choices=["synthetic", "standard"],
        help="Type of dataset: 'synthetic' for SPEED_PLUS_SYNTHETIC_FIXED (expects speedplus_synthetic_{split}), 'standard' for SPEED_PLUS_FIXED (expects speedplus_{split})",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        help="Path to input prediction JSON file (alternative to dataset-type)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        help="Path for output annotation file (optional, defaults to input directory with generated name)",
    )

    args = parser.parse_args()

    # Check which mode to use
    if args.input_file:
        # Use single file mode
        create_bbox_annotations_from_file(args.input_file, args.output_file)
    elif args.dataset_type:
        # Use original dataset mode
        create_bbox_annotations(args.dataset_type)
    else:
        print("Error: You must specify either --input-file or --dataset-type")
        parser.print_help()
