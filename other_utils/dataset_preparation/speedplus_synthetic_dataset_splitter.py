# This script splits the SPEED+ dataset into train, val, and test sets
# The training set remains unchanged from the original dataset
# The validation set is split 50/50 into new validation and test sets

import json
import os
import shutil
import argparse

from sklearn.model_selection import train_test_split
from tqdm import tqdm


def split_speedplus_dataset(original_path: str, new_path: str) -> None:
    """
    Split the SPEED+ dataset into train, val, and test sets
    Training set remains unchanged (47,966 images)
    Validation set is split 50/50 into val (5,997) and test (5,997) sets

    Args:
        original_path (str): Path to the original SPEED+ dataset
        new_path (str): Path to save the new dataset

    Returns:
        None
    """

    # Create the new directory structure
    os.makedirs(os.path.join(new_path, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(new_path, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(new_path, "images", "test"), exist_ok=True)

    # Load the original JSON files
    with open(os.path.join(original_path, "train.json"), "r") as f:
        train_data = json.load(f)

    with open(os.path.join(original_path, "validation.json"), "r") as f:
        validation_data = json.load(f)

    # Split the validation data 50/50 into new val and test sets
    val_data, test_data = train_test_split(validation_data, test_size=0.5, random_state=42)

    # Function to copy images and create new JSON files
    def process_split(split_name, split_data, source_images_dir="images"):
        new_json_data = []
        for item in tqdm(split_data, desc=f"Copying {split_name} images"):
            # Copy the image from the original images directory
            src_image = os.path.join(original_path, source_images_dir, item["filename"])
            dst_image = os.path.join(new_path, "images", split_name, item["filename"])
            shutil.copy2(src_image, dst_image)

            # Add to new JSON data
            new_json_data.append(item)

        # Save the new JSON file
        with open(os.path.join(new_path, f"{split_name}.json"), "w") as f:
            json.dump(new_json_data, f, indent=2)

    # Process each split
    print(f"Processing training set: {len(train_data)} images...")
    process_split("train", train_data)

    print(f"Processing validation set: {len(val_data)} images...")
    process_split("val", val_data)

    print(f"Processing test set: {len(test_data)} images...")
    process_split("test", test_data)

    print(f"Dataset split complete. New dataset structure created at {new_path}")
    print(f"  - Training: {len(train_data)} images")
    print(f"  - Validation: {len(val_data)} images")
    print(f"  - Test: {len(test_data)} images")
    print(f"  - Total: {len(train_data) + len(val_data) + len(test_data)} images")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split SPEED+ dataset into train, val, and test sets. "
        "Training set remains unchanged, validation set is split 50/50."
    )
    parser.add_argument(
        "--original-path",
        type=str,
        default="./speedplusv2/synthetic",
        help="Path to the original SPEED+ dataset",
    )
    parser.add_argument(
        "--new-path",
        type=str,
        default="./SPEED_PLUS_SYNTHETIC_FIXED",
        help="Path to save the new dataset",
    )

    args = parser.parse_args()

    # Split the dataset
    split_speedplus_dataset(args.original_path, args.new_path)
