# This script splits the SPEED dataset into train, val, and test sets
# with the respective splits being 80%, 10%, and 10% of the original dataset

import json
import os
import shutil
import argparse

from sklearn.model_selection import train_test_split


def split_speed_dataset(original_path: str, new_path: str) -> None:
    """
    Split the SPEED dataset into train, val, and test sets
    Respectively, the splits are 80%, 10%, and 10% of the original dataset

    Args:
        original_path (str): Path to the original dataset
        new_path (str): Path to save the new dataset

    Returns:
        None
    """

    # Create the new directory structure
    os.makedirs(os.path.join(new_path, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(new_path, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(new_path, "images", "test"), exist_ok=True)

    # Load the original JSON file
    with open(os.path.join(original_path, "train.json"), "r") as f:
        data = json.load(f)

    # Split the data
    train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    # Function to copy images and create new JSON files
    def process_split(split_name, split_data):
        new_json_data = []
        for item in split_data:
            # Copy the image
            src_image = os.path.join(original_path, "images", "train", item["filename"])
            dst_image = os.path.join(new_path, "images", split_name, item["filename"])
            shutil.copy2(src_image, dst_image)

            # Add to new JSON data
            new_json_data.append(item)

        # Save the new JSON file
        with open(os.path.join(new_path, f"{split_name}.json"), "w") as f:
            json.dump(new_json_data, f, indent=2)

    # Process each split
    process_split("train", train_data)
    process_split("val", val_data)
    process_split("test", test_data)

    print(f"Dataset split complete. New dataset structure created at {new_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split SPEED dataset into train, val, and test sets"
    )
    parser.add_argument(
        "--original-path", type=str, default="./speed", help="Path to the original dataset"
    )
    parser.add_argument(
        "--new-path", type=str, default="./SPEED_FIXED", help="Path to save the new dataset"
    )

    args = parser.parse_args()

    # Split the dataset
    split_speed_dataset(args.original_path, args.new_path)
