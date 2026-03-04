# This script creates the SPEED_PLUS_FIXED dataset
# Combines synthetic training/validation data with lightbox/sunlamp test data
# Training set: 47,966 synthetic images (unchanged)
# Validation set: 11,994 synthetic images (full validation set)
# Test set: 9,531 real images (6,740 lightbox + 2,791 sunlamp)

import json
import os
import shutil
import argparse

from tqdm import tqdm


def create_speedplus_fixed_dataset(speedplusv2_path: str, new_path: str) -> None:
    """
    Create SPEED_PLUS_FIXED dataset combining synthetic and real data
    Training: 47,966 synthetic images (unchanged)
    Validation: 11,994 synthetic images (full validation set)
    Test: 9,531 real images (lightbox + sunlamp with unique naming)

    Args:
        speedplusv2_path (str): Path to the original speedplusv2 dataset
        new_path (str): Path to save the new dataset

    Returns:
        None
    """

    # Create the new directory structure
    os.makedirs(os.path.join(new_path, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(new_path, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(new_path, "images", "test"), exist_ok=True)

    # Load the original synthetic JSON files
    with open(os.path.join(speedplusv2_path, "synthetic", "train.json"), "r") as f:
        train_data = json.load(f)

    with open(os.path.join(speedplusv2_path, "synthetic", "validation.json"), "r") as f:
        validation_data = json.load(f)

    # Load lightbox and sunlamp test data
    with open(os.path.join(speedplusv2_path, "lightbox", "test.json"), "r") as f:
        lightbox_data = json.load(f)

    with open(os.path.join(speedplusv2_path, "sunlamp", "test.json"), "r") as f:
        sunlamp_data = json.load(f)

    # Function to copy images and create new JSON files
    def process_synthetic_split(split_name, split_data, source_images_dir="images"):
        new_json_data = []
        print(f"Processing {split_name} set: {len(split_data)} images...")

        for item in tqdm(split_data, desc=f"Copying {split_name} images"):
            # Copy the image from the synthetic images directory
            src_image = os.path.join(
                speedplusv2_path, "synthetic", source_images_dir, item["filename"]
            )
            dst_image = os.path.join(new_path, "images", split_name, item["filename"])
            if os.path.exists(src_image):
                shutil.copy2(src_image, dst_image)

            # Add to new JSON data
            new_json_data.append(item)

        # Save the new JSON file
        with open(os.path.join(new_path, f"{split_name}.json"), "w") as f:
            json.dump(new_json_data, f, indent=2)

    # Function to process test data with suffix renaming
    def process_test_data(test_data, test_type, suffix):
        new_test_data = []
        images_dir = os.path.join(speedplusv2_path, test_type, "images")

        print(f"Processing {test_type} test images: {len(test_data)} images...")

        for item in tqdm(test_data, desc=f"Copying {test_type} images"):
            # Create new filename with suffix
            original_filename = item["filename"]
            name, ext = os.path.splitext(original_filename)
            new_filename = f"{name}_{suffix}{ext}"

            # Copy image with new name
            src_image = os.path.join(images_dir, original_filename)
            dst_image = os.path.join(new_path, "images", "test", new_filename)
            if os.path.exists(src_image):
                shutil.copy2(src_image, dst_image)

            # Update annotation with new filename
            new_item = item.copy()
            new_item["filename"] = new_filename
            new_test_data.append(new_item)

        return new_test_data

    # Process synthetic train and validation sets
    process_synthetic_split("train", train_data)
    process_synthetic_split("val", validation_data)

    # Process test sets with suffix renaming
    lightbox_test_data = process_test_data(lightbox_data, "lightbox", "lightbox")
    sunlamp_test_data = process_test_data(sunlamp_data, "sunlamp", "sunlamp")

    # Merge test data and save
    merged_test_data = lightbox_test_data + sunlamp_test_data
    with open(os.path.join(new_path, "test.json"), "w") as f:
        json.dump(merged_test_data, f, indent=2)

    print(f"Dataset creation complete. New dataset structure created at {new_path}")
    print(f"  - Training: {len(train_data)} images")
    print(f"  - Validation: {len(validation_data)} images")
    print(
        f"  - Test: {len(merged_test_data)} images ({len(lightbox_test_data)} lightbox + {len(sunlamp_test_data)} sunlamp)"
    )
    print(f"  - Total: {len(train_data) + len(validation_data) + len(merged_test_data)} images")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create SPEED_PLUS_FIXED dataset combining synthetic and real data. "
        "Training and validation use synthetic data, test uses lightbox + sunlamp real data."
    )
    parser.add_argument(
        "--speedplusv2-path",
        type=str,
        default="./speedplusv2",
        help="Path to the original speedplusv2 dataset",
    )
    parser.add_argument(
        "--new-path", type=str, default="./SPEED_PLUS_FIXED", help="Path to save the new dataset"
    )

    args = parser.parse_args()

    # Create the dataset
    create_speedplus_fixed_dataset(args.speedplusv2_path, args.new_path)
