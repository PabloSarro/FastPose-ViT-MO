# This script merges the SPEED test and real_test datasets into a single prediction dataset
# for competition submission

import json
import os
import shutil
import argparse


def merge_speed_prediction_datasets(original_path: str, new_path: str) -> None:
    """
    Merge the SPEED test and real_test datasets into a single prediction dataset

    Args:
        original_path (str): Path to the original SPEED dataset
        new_path (str): Path to the SPEED_FIXED dataset where merged data will be saved

    Returns:
        None
    """

    # Create the new directory structure
    prediction_images_dir = os.path.join(new_path, "images", "to_predict")
    os.makedirs(prediction_images_dir, exist_ok=True)

    # Load the JSON files
    test_json_path = os.path.join(original_path, "test.json")
    real_test_json_path = os.path.join(original_path, "real_test.json")

    print(f"Loading {test_json_path}...")
    with open(test_json_path, "r") as f:
        test_data = json.load(f)

    print(f"Loading {real_test_json_path}...")
    with open(real_test_json_path, "r") as f:
        real_test_data = json.load(f)

    print(f"Test data: {len(test_data)} entries")
    print(f"Real test data: {len(real_test_data)} entries")

    # Check for filename conflicts
    test_filenames = set(item["filename"] for item in test_data)
    real_test_filenames = set(item["filename"] for item in real_test_data)
    conflicts = test_filenames.intersection(real_test_filenames)

    if conflicts:
        print(f"WARNING: Found {len(conflicts)} filename conflicts: {list(conflicts)[:5]}...")
        print("Proceeding anyway - real_test images will overwrite test images with same names")
    else:
        print("No filename conflicts found")

    # Add dummy pose fields to all entries for compatibility with SPEEDDataset
    for item in test_data:
        item["q_vbs2tango"] = [0.0, 0.0, 0.0, 1.0]  # Identity quaternion
        item["r_Vo2To_vbs_true"] = [0.0, 0.0, 0.0]  # Zero translation

    for item in real_test_data:
        item["q_vbs2tango"] = [0.0, 0.0, 0.0, 1.0]  # Identity quaternion
        item["r_Vo2To_vbs_true"] = [0.0, 0.0, 0.0]  # Zero translation

    # Merge the JSON data
    merged_data = test_data + real_test_data
    print(f"Total merged entries: {len(merged_data)}")

    # Copy images from test directory
    test_images_dir = os.path.join(original_path, "images", "test")
    copied_test = 0
    print("Copying test images...")
    for item in test_data:
        filename = item["filename"]
        src_path = os.path.join(test_images_dir, filename)
        dst_path = os.path.join(prediction_images_dir, filename)

        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            copied_test += 1
        else:
            print(f"WARNING: Missing test image: {src_path}")

    # Copy images from real_test directory
    real_test_images_dir = os.path.join(original_path, "images", "real_test")
    copied_real_test = 0
    print("Copying real_test images...")
    for item in real_test_data:
        filename = item["filename"]
        src_path = os.path.join(real_test_images_dir, filename)
        dst_path = os.path.join(prediction_images_dir, filename)

        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            copied_real_test += 1
        else:
            print(f"WARNING: Missing real_test image: {src_path}")

    # Save the merged JSON file
    merged_json_path = os.path.join(new_path, "to_predict.json")
    print(f"Saving merged JSON to {merged_json_path}...")
    with open(merged_json_path, "w") as f:
        json.dump(merged_data, f, indent=2)

    # Summary
    print("\n=== MERGE SUMMARY ===")
    print(f"Test images copied: {copied_test}/{len(test_data)}")
    print(f"Real test images copied: {copied_real_test}/{len(real_test_data)}")
    print(f"Total images in prediction dataset: {copied_test + copied_real_test}")
    print(f"Total JSON entries: {len(merged_data)}")
    print(f"Prediction dataset created at: {prediction_images_dir}")
    print(f"Prediction JSON created at: {merged_json_path}")

    # Verify final image count
    final_image_count = len(
        [
            f
            for f in os.listdir(prediction_images_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
    )
    print(f"Actual images in destination: {final_image_count}")

    if final_image_count == len(merged_data):
        print("✓ Image count matches JSON entries")
    else:
        print("✗ Image count does not match JSON entries")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge SPEED test and real_test datasets for competition submission"
    )
    parser.add_argument(
        "--original-path", type=str, default="./speed", help="Path to the original SPEED dataset"
    )
    parser.add_argument(
        "--new-path", type=str, default="./SPEED_FIXED", help="Path to the SPEED_FIXED dataset"
    )

    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.original_path):
        print(f"Error: Original path does not exist: {args.original_path}")
        exit(1)

    if not os.path.exists(args.new_path):
        print(f"Error: New path does not exist: {args.new_path}")
        exit(1)

    # Check required files exist
    required_files = [
        os.path.join(args.original_path, "test.json"),
        os.path.join(args.original_path, "real_test.json"),
        os.path.join(args.original_path, "images", "test"),
        os.path.join(args.original_path, "images", "real_test"),
    ]

    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"Error: Required file/directory does not exist: {file_path}")
            exit(1)

    print("Starting SPEED prediction dataset merge...")
    merge_speed_prediction_datasets(args.original_path, args.new_path)
    print("Merge completed successfully!")
