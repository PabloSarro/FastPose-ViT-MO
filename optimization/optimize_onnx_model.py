"""ONNX model simplification and optimization script.

Uses onnx-simplifier to remove redundant nodes and optimize the model graph.
"""

import argparse

import onnx
from onnxsim import simplify


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for ONNX model simplification.

    Returns:
        Parsed command line arguments with input and output paths.
    """
    parser = argparse.ArgumentParser(
        description="Simplify ONNX model by removing redundant nodes."
    )
    parser.add_argument(
        "--input", "-i", type=str, default="FastPoseViT.onnx", help="Path to the input ONNX model"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="Simplified_FastPoseViT.onnx",
        help="Path to save the simplified ONNX model",
    )
    return parser.parse_args()


def main() -> None:
    """Run ONNX model simplification.

    Loads an ONNX model, applies graph optimizations using onnx-simplifier,
    validates the simplified model, and saves the result.

    Raises:
        AssertionError: If the simplified model fails validation.
    """
    # Parse command line arguments
    args = parse_args()

    # Load the original ONNX model
    model = onnx.load(args.input)

    # Simplify the model
    model_simp, check = simplify(model)

    # Verify simplification was successful
    assert check, "Simplified ONNX model could not be validated"

    # Save the simplified model
    onnx.save(model_simp, args.output)
    print(f"Model simplified successfully and saved to {args.output}")


if __name__ == "__main__":
    main()
