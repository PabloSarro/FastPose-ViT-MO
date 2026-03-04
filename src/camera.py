# This file contains the camera parameters for the datasets. In our case we just have SPEED, so we only define one camera.

import torch


class Camera:
    """
    Data class based on UrsoNet's implementation:
    https://github.com/pedropro/UrsoNet/blob/8e59d9b81dd3805aba1d773bd9b44f1a33745b05/speed.py
    Parameters verified with the original SPEED dataset paper:
    https://arxiv.org/pdf/1906.09868
    Also works for SPEED+ dataset, as they share the same camera parameters.
    """

    # Focal length[m]
    fwx = 0.0176
    fwy = 0.0176
    # Image size [pixels]
    width = 1920
    height = 1200
    # Size of the pixels [m / pixel]
    ppx = 5.86e-6
    ppy = ppx

    # Focal length[pixels]
    fx = fwx / ppx
    fy = fwy / ppy

    # Intrinsics matrix
    K = torch.tensor([[fx, 0, width / 2], [0, fy, height / 2], [0, 0, 1]])
    # Inverse of the intrinsics matrix
    K_inv = torch.inverse(K)

    # Centered intrinsics matrix
    K_c = torch.tensor([[fx, 0, 0], [0, fy, 0], [0, 0, 1]])
    # Inverse of the centered intrinsics matrix
    K_c_inv = torch.inverse(K_c)

    # Camera aspect ratio
    aspect_ratio = width / height

    # Pre-computed constants for performance optimization
    width_height_product = width * height
    width_height_over_aspect = width_height_product / aspect_ratio

    # Cache commonly used values for translation computations
    cx = width / 2  # Image center x
    cy = height / 2  # Image center y
