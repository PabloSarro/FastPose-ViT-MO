# This file contains diverse utility functions used in the project

import argparse
import logging
import os
from collections import deque
from datetime import datetime

import torch
import torch.nn.functional as F
from src.camera import Camera

# Simple numeric constants for stability
EPS = 1e-8

# Cached tensor constants (will be populated on first use)
_CACHED_TENSORS = {
    "identity_3x3": None,
    "z_axis": None,
    "x_rotation_180": None,
}


def _get_cached_tensor(key: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Get cached tensor constant, creating if needed.

    Args:
        key: Tensor identifier ('identity_3x3', 'z_axis', or 'x_rotation_180').
        device: Target device for the tensor.
        dtype: Data type for the tensor.

    Returns:
        The cached tensor constant.
    """
    cache_key = f"{key}_{device}_{dtype}"
    if cache_key not in _CACHED_TENSORS:
        if key == "identity_3x3":
            _CACHED_TENSORS[cache_key] = torch.eye(3, device=device, dtype=dtype)
        elif key == "z_axis":
            _CACHED_TENSORS[cache_key] = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
        elif key == "x_rotation_180":
            _CACHED_TENSORS[cache_key] = torch.tensor(
                [1.0, 0.0, 0.0, 0.0], device=device, dtype=dtype
            )
    return _CACHED_TENSORS[cache_key]


def _compute_skew_symmetric(axis: torch.Tensor) -> torch.Tensor:
    """Compute skew-symmetric matrix K from axis vector(s).

    Args:
        axis: Batch of axis vectors with shape (..., 3).

    Returns:
        Batch of skew-symmetric matrices with shape (..., 3, 3).
    """
    # Handle both single vector and batched inputs
    batch_dims = axis.shape[:-1]
    device, dtype = axis.device, axis.dtype

    K = torch.zeros(*batch_dims, 3, 3, device=device, dtype=dtype)
    K[..., 0, 1] = -axis[..., 2]
    K[..., 0, 2] = axis[..., 1]
    K[..., 1, 0] = axis[..., 2]
    K[..., 1, 2] = -axis[..., 0]
    K[..., 2, 0] = -axis[..., 1]
    K[..., 2, 1] = axis[..., 0]

    return K


def _rodrigues_rotation_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Compute rotation matrix using Rodrigues formula.

    Args:
        axis: Normalized rotation axis with shape (..., 3).
        angle: Rotation angle with shape (...).

    Returns:
        Rotation matrices with shape (..., 3, 3).
    """
    device, dtype = axis.device, axis.dtype
    batch_dims = axis.shape[:-1]

    # Pre-compute trigonometric values
    sin_angle = torch.sin(angle)
    cos_angle = torch.cos(angle)
    one_minus_cos = 1 - cos_angle

    # Get identity matrix
    identity = _get_cached_tensor("identity_3x3", device, dtype)
    if batch_dims:
        identity = identity.unsqueeze(0).expand(*batch_dims, -1, -1)

    # Compute skew-symmetric matrix
    K = _compute_skew_symmetric(axis)
    K_squared = K @ K

    # Rodrigues formula: R = I + sin(θ)K + (1-cos(θ))K²
    if batch_dims:
        sin_angle = sin_angle.view(*batch_dims, 1, 1)
        one_minus_cos = one_minus_cos.view(*batch_dims, 1, 1)

    R = identity + sin_angle * K + one_minus_cos * K_squared

    return R


def _safe_normalize(tensor: torch.Tensor, dim: int = -1, eps: float = EPS) -> torch.Tensor:
    """Safely normalize tensor with epsilon handling to prevent division by zero.

    Args:
        tensor: Input tensor to normalize.
        dim: Dimension along which to normalize.
        eps: Small epsilon value for numerical stability.

    Returns:
        Normalized tensor with unit norm along the specified dimension.
    """
    return F.normalize(tensor, p=2, dim=dim, eps=eps)


def _compute_orientation_transform(
    translation: torch.Tensor,
    rotation_quat: torch.Tensor,
    inverse: bool = False,
    no_rotation_compensation: bool = False,
) -> torch.Tensor:
    """
    Unified function for orientation transformations (absolute ↔ apparent).
    Pure batched implementation.

    Args:
        translation: Batch of translation vectors (B, 3)
        rotation_quat: Batch of quaternions (B, 4)
        inverse: If True, computes apparent from absolute (transpose R_rel)
        no_rotation_compensation: Skip rotation compensation

    Returns:
        Batch of transformed quaternions (B, 4)
    """
    if no_rotation_compensation:
        return rotation_quat

    device, dtype = translation.device, translation.dtype
    batch_size = translation.shape[0]

    # Convert quaternions to rotation matrices
    R_input = quaternion_to_rotation_matrix(rotation_quat).to(dtype)

    # Normalize translation vectors
    v_new = _safe_normalize(translation, dim=-1)

    # Get cached z-axis vector and expand for batch
    v_old = _get_cached_tensor("z_axis", device, dtype)
    v_old = v_old.unsqueeze(0).expand(batch_size, -1)

    # Compute dot products and rotation axes
    dot_products = torch.sum(v_old * v_new, dim=-1)
    axes = torch.cross(v_old, v_new, dim=-1)

    # Handle parallel/anti-parallel cases
    axis_norms = torch.norm(axes, dim=-1, keepdim=True)
    parallel_mask = (axis_norms < 1e-6).squeeze(-1)

    # Normalize axes safely
    axes = axes / torch.clamp(axis_norms, min=1e-6)

    # Compute angles
    angles = torch.acos(torch.clamp(dot_products, -1.0, 1.0))

    # Compute rotation matrices using Rodrigues formula
    R_rel = _rodrigues_rotation_matrix(axes, angles)

    # Handle special cases for parallel vectors
    identity = _get_cached_tensor("identity_3x3", device, dtype)
    antiparallel_rot = torch.diag(torch.tensor([1.0, -1.0, -1.0], device=device, dtype=dtype))

    identity = identity.unsqueeze(0).expand(batch_size, -1, -1)
    antiparallel_rot = antiparallel_rot.unsqueeze(0).expand(batch_size, -1, -1)

    positive_parallel = parallel_mask & (dot_products > 0)
    negative_parallel = parallel_mask & (dot_products <= 0)

    R_rel = torch.where(positive_parallel.unsqueeze(-1).unsqueeze(-1), identity, R_rel)
    R_rel = torch.where(negative_parallel.unsqueeze(-1).unsqueeze(-1), antiparallel_rot, R_rel)

    # Apply transformation (normal or inverse)
    if inverse:
        R_result = torch.transpose(R_rel, -2, -1) @ R_input  # Transpose for inverse
    else:
        R_result = R_rel @ R_input

    return rotation_matrix_to_quaternion(R_result)


def setup_logging(
    args: argparse.Namespace,
    initialized: bool = False,
    log_file: str = None,
    base_filename: str = "train",
) -> tuple[logging.Logger, str]:
    """
    Set up logging for the application

    Args:
        args (argparse.Namespace): Command line arguments
        initialized (bool): Whether the logger has already been initialized
        log_file (str): Path to the log file

    Returns:
        tuple[logging.Logger, str]: Logger object and path to the log file
    """
    # Create log directory if it doesn't exist
    if not initialized:
        os.makedirs(args.log_dir, exist_ok=True)

        # Create a unique log file name
        log_file = f"{args.log_dir}/{base_filename}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"

        # Create a logger
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(__name__)
        log_file = log_file

    # Clear any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.INFO)
    # Prevent the logger from propagating messages to the root logger
    logger.propagate = False

    # Create handlers
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()

    # Set level for handlers
    file_handler.setLevel(logging.INFO)
    console_handler.setLevel(logging.INFO)

    # Create a formatting configuration
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Set formatter for handlers
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger, log_file


def smooth_loss(loss_deque: deque) -> float:
    """
    Calculate the average loss over a deque

    Args:
        loss_deque (deque): Deque containing the losses

    Returns:
        float: Average loss
    """
    return sum(loss_deque) / len(loss_deque) if loss_deque else 0


def get_rotation_matrix(x: torch.Tensor) -> torch.Tensor:
    """
    Perform Gram-Schmidt orthogonalization on input vectors, then compute the third vector using cross product.
    Return rotation matrices in SO(3). Pure batched implementation.

    Args:
        x (torch.Tensor): Batch of 6D vectors (B, 6) containing [r1, r2] vectors

    Returns:
        torch.Tensor: Batch of rotation matrices (B, 3, 3)
    """
    # Split input efficiently
    v1, v2 = x[:, :3], x[:, 3:]

    # Normalize v1 in-place for memory efficiency
    v1 = _safe_normalize(v1, dim=1)

    # Use Gram-Schmidt to compute v2 (optimized dot product computation)
    v2_dot_v1 = torch.sum(v2 * v1, dim=1, keepdim=True)
    v2 = v2 - v2_dot_v1 * v1
    v2 = _safe_normalize(v2, dim=1)

    # Compute v3 as cross product of v1 and v2
    v3 = torch.cross(v1, v2, dim=1)

    # Stack the vectors to form the rotation matrix (memory efficient)
    return torch.stack([v1, v2, v3], dim=2)


def rotation_matrix_to_6d(rotation_matrices: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrices to 6D representation (first two columns).
    Pure batched implementation.

    Args:
        rotation_matrices (torch.Tensor): Batch of rotation matrices (B, 3, 3)

    Returns:
        torch.Tensor: Batch of 6D vectors (B, 6)
    """
    # Extract first two columns of the rotation matrix
    v1 = rotation_matrices[:, :, 0]  # Shape: (B, 3)
    v2 = rotation_matrices[:, :, 1]  # Shape: (B, 3)

    # Concatenate to form 6D representation
    return torch.cat([v1, v2], dim=1)  # Shape: (B, 6)


def do_gram_schmidt_2D(r1: torch.Tensor, r2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given two vectors r1 and r2, return an orthonormal basis for the plane they span following the Gram-Schmidt process.

    Args:
        r1 (torch.Tensor): First vector (3,)
        r2 (torch.Tensor): Second vector (3,)

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Orthonormal basis vectors for the plane spanned by r1 and r2
    """
    # Normalize r1 using safe normalization
    r1 = _safe_normalize(r1, dim=0)

    # Compute projection efficiently
    projection = torch.dot(r2, r1)
    r2 = r2 - projection * r1

    # Normalize r2
    r2 = _safe_normalize(r2, dim=0)

    return r1, r2


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """
    Converts unit quaternions to 3x3 rotation matrices using optimized operations.
    Pure batched implementation.
    Based on: https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation

    Args:
        q (torch.Tensor): Batch of unit quaternions (B, 4)

    Returns:
        torch.Tensor: Batch of rotation matrices (B, 3, 3)
    """
    # Extract quaternion components efficiently
    qr, qi, qj, qk = q.unbind(-1)

    # Compute squares and cross terms in batch
    qi2, qj2, qk2 = qi * qi, qj * qj, qk * qk
    qiqj, qiqk, qjqk = qi * qj, qi * qk, qj * qk
    qrqi, qrqj, qrqk = qr * qi, qr * qj, qr * qk

    # Compute rotation matrix elements
    r00 = 1 - 2 * (qj2 + qk2)
    r01 = 2 * (qiqj - qrqk)
    r02 = 2 * (qiqk + qrqj)

    r10 = 2 * (qiqj + qrqk)
    r11 = 1 - 2 * (qi2 + qk2)
    r12 = 2 * (qjqk - qrqi)

    r20 = 2 * (qiqk - qrqj)
    r21 = 2 * (qjqk + qrqi)
    r22 = 1 - 2 * (qi2 + qj2)

    # Stack efficiently into rotation matrices
    return torch.stack(
        [
            torch.stack([r00, r01, r02], dim=-1),
            torch.stack([r10, r11, r12], dim=-1),
            torch.stack([r20, r21, r22], dim=-1),
        ],
        dim=-2,
    )


def process_rotation(rotation_output: torch.Tensor, rotation_format: str) -> torch.Tensor:
    """
    Process the rotation output based on the desired format.

    Args:
        rotation_output (torch.Tensor): Raw rotation output
        rotation_format (str): Desired rotation format ("quaternion" or "matrix")

    Returns:
        torch.Tensor: Processed rotation output
    """
    # Do not act on quaternions
    if rotation_format == "quaternion":
        return rotation_output
    # Convert the rotation head output (two first column of rotation matrix) to rotation matrices (in SO(3))
    elif rotation_format == "matrix":
        return get_rotation_matrix(rotation_output)
    else:
        raise ValueError(f"Invalid rotation format: {rotation_format}")


def rotation_matrix_to_quaternion(rotation_matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of 3x3 rotation matrices to quaternions. Treats multiple cases for numerical stability.
    Based on: https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/

    Args:
        rotation_matrix (torch.Tensor): Batch of rotation matrices (N x 3 x 3) or single matrix (3 x 3)

    Returns:
        torch.Tensor: Batch of quaternions (N x 4) in format (w, x, y, z) or single quaternion (4,)
    """
    # Handle both batched and unbatched inputs efficiently
    if rotation_matrix.dim() == 2:
        rotation_matrix = rotation_matrix.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    batch_size = rotation_matrix.size(0)
    device = rotation_matrix.device

    # Compute trace more efficiently
    trace = torch.diagonal(rotation_matrix, dim1=1, dim2=2).sum(dim=1)

    # Initialize quaternions
    q = torch.zeros(batch_size, 4, device=device, dtype=rotation_matrix.dtype)

    # Extract matrix elements once for reuse
    R = rotation_matrix

    # Case 1: Trace > 0
    mask_1 = trace > 0
    if mask_1.any():
        r = torch.sqrt(torch.clamp(1 + trace[mask_1], min=EPS))
        s = 0.5 / r
        q[mask_1, 0] = 0.5 * r
        q[mask_1, 1] = (R[mask_1, 2, 1] - R[mask_1, 1, 2]) * s
        q[mask_1, 2] = (R[mask_1, 0, 2] - R[mask_1, 2, 0]) * s
        q[mask_1, 3] = (R[mask_1, 1, 0] - R[mask_1, 0, 1]) * s

    # Case 2: R[0,0] is largest diagonal element
    mask_2 = (~mask_1) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    if mask_2.any():
        r = torch.sqrt(
            torch.clamp(1 + R[mask_2, 0, 0] - R[mask_2, 1, 1] - R[mask_2, 2, 2], min=EPS)
        )
        s = 0.5 / r
        q[mask_2, 1] = 0.5 * r
        q[mask_2, 0] = (R[mask_2, 2, 1] - R[mask_2, 1, 2]) * s
        q[mask_2, 2] = (R[mask_2, 0, 1] + R[mask_2, 1, 0]) * s
        q[mask_2, 3] = (R[mask_2, 0, 2] + R[mask_2, 2, 0]) * s

    # Case 3: R[1,1] is largest diagonal element
    mask_3 = (~mask_1) & (~mask_2) & (R[:, 1, 1] > R[:, 2, 2])
    if mask_3.any():
        r = torch.sqrt(
            torch.clamp(1 - R[mask_3, 0, 0] + R[mask_3, 1, 1] - R[mask_3, 2, 2], min=EPS)
        )
        s = 0.5 / r
        q[mask_3, 2] = 0.5 * r
        q[mask_3, 0] = (R[mask_3, 0, 2] - R[mask_3, 2, 0]) * s
        q[mask_3, 1] = (R[mask_3, 0, 1] + R[mask_3, 1, 0]) * s
        q[mask_3, 3] = (R[mask_3, 1, 2] + R[mask_3, 2, 1]) * s

    # Case 4: R[2,2] is largest diagonal element
    mask_4 = (~mask_1) & (~mask_2) & (~mask_3)
    if mask_4.any():
        r = torch.sqrt(
            torch.clamp(1 - R[mask_4, 0, 0] - R[mask_4, 1, 1] + R[mask_4, 2, 2], min=EPS)
        )
        s = 0.5 / r
        q[mask_4, 3] = 0.5 * r
        q[mask_4, 0] = (R[mask_4, 1, 0] - R[mask_4, 0, 1]) * s
        q[mask_4, 1] = (R[mask_4, 0, 2] + R[mask_4, 2, 0]) * s
        q[mask_4, 2] = (R[mask_4, 1, 2] + R[mask_4, 2, 1]) * s

    # Normalize quaternions
    q = F.normalize(q, p=2, dim=1)

    return q.squeeze(0) if squeeze_output else q


def translation_to_bbox_relative_translation(
    translation: torch.Tensor,
    bbox: torch.Tensor,
    camera: Camera,
) -> torch.Tensor:
    """
    Transform translation vectors to relative translation vectors based on bounding boxes and camera parameters.
    Pure batched implementation.

    Args:
        translation: torch.Tensor - Batch of translation vectors (B, 3)
        bbox: torch.Tensor - Batch of bboxes (B, 4) containing [x1, y1, x2, y2]
        camera: Camera object with intrinsic parameters

    Returns:
        torch.Tensor - Batch of relative translations (B, 3)
    """
    # Compute bbox dimensions and centers in one go
    bbox_dims = bbox[:, 2:] - bbox[:, :2]  # [w, h] Shape: (B, 2)
    w, h = bbox_dims[:, 0], bbox_dims[:, 1]  # Shape: (B,) each
    bbox_centers = bbox[:, :2] + bbox_dims * 0.5  # Shape: (B, 2)
    bbox_center_x, bbox_center_y = bbox_centers[:, 0], bbox_centers[:, 1]

    # Cache camera parameters
    cx, cy, fx, fy = camera.cx, camera.cy, camera.fx, camera.fy

    # Cache translation components
    tx, ty, tz = translation[:, 0], translation[:, 1], translation[:, 2]

    # Use least squares solution for avg_ratio computation
    avg_ratio = 0.5 * ((w * camera.aspect_ratio) / camera.width + h / camera.height)
    new_target_z = tz * avg_ratio

    # Vectorized target computation
    inv_tz = 1.0 / tz
    new_target_x = (tx * fx * inv_tz + cx - bbox_center_x) / w
    new_target_y = (ty * fy * inv_tz + cy - bbox_center_y) / h

    # Use torch.stack for better performance than individual assignments
    return torch.stack([new_target_x, new_target_y, new_target_z], dim=1)


def bbox_relative_translation_to_translation(
    bbox_relative_translation: torch.Tensor,
    bbox: torch.Tensor,
    camera: Camera,
) -> torch.Tensor:
    """
    Transform bbox relative translation vectors to absolute translations based on bounding boxes and camera parameters.
    Pure batched implementation.

    Args:
        bbox_relative_translation: torch.Tensor - Batch of relative translations (B, 3)
        bbox: torch.Tensor - Batch of bboxes (B, 4) containing [x1, y1, x2, y2]
        camera: Camera object with intrinsic parameters

    Returns:
        torch.Tensor - Batch of absolute translations (B, 3)
    """
    # Compute bbox dimensions and centers efficiently
    bbox_dims = bbox[:, 2:] - bbox[:, :2]  # [w, h] Shape: (B, 2)
    w, h = bbox_dims[:, 0], bbox_dims[:, 1]  # Shape: (B,) each
    bbox_centers = bbox[:, :2] + bbox_dims * 0.5  # Shape: (B, 2)
    bbox_center_x, bbox_center_y = bbox_centers[:, 0], bbox_centers[:, 1]

    # Cache camera parameters
    cx, cy, fx, fy = camera.cx, camera.cy, camera.fx, camera.fy

    # Cache relative translation components
    rel_tx, rel_ty, rel_tz = (
        bbox_relative_translation[:, 0],
        bbox_relative_translation[:, 1],
        bbox_relative_translation[:, 2],
    )

    # Use least squares solution for avg_ratio computation
    avg_ratio = 0.5 * ((w * camera.aspect_ratio) / camera.width + h / camera.height)

    # Compute A and B coefficients
    A = (rel_tx * w - cx + bbox_center_x) / fx
    B = (rel_ty * h - cy + bbox_center_y) / fy

    # Compute translation_z using least squares method
    translation_z = rel_tz / avg_ratio

    # Compute x and y translations
    translation_x = A * translation_z
    translation_y = B * translation_z

    return torch.stack([translation_x, translation_y, translation_z], dim=1)


def get_absolute_orientation(
    translation: torch.Tensor, rotation_quat: torch.Tensor, no_rotation_compensation: bool = False
) -> torch.Tensor:
    """
    Convert apparent rotations to absolute rotations (centered on image).
    Pure batched implementation.

    Args:
        translation (torch.Tensor): Batch of translation vectors (B, 3)
        rotation_quat (torch.Tensor): Batch of quaternions (B, 4)
        no_rotation_compensation: bool to disable rotation compensation

    Returns:
        torch.Tensor: Batch of centered rotations as quaternions (B, 4)
    """
    return _compute_orientation_transform(
        translation,
        rotation_quat,
        inverse=False,
        no_rotation_compensation=no_rotation_compensation,
    )


def get_apparent_orientation(
    translation: torch.Tensor,
    centered_rotation_quat: torch.Tensor,
    no_rotation_compensation: bool = False,
) -> torch.Tensor:
    """
    Convert absolute rotations to apparent rotations (perturbed by translation).
    Pure batched implementation.

    Args:
        translation (torch.Tensor): Batch of translation vectors (B, 3)
        centered_rotation_quat (torch.Tensor): Batch of quaternions (B, 4)
        no_rotation_compensation: bool to disable rotation compensation

    Returns:
        torch.Tensor: Batch of apparent rotations as quaternions (B, 4)
    """
    return _compute_orientation_transform(
        translation,
        centered_rotation_quat,
        inverse=True,
        no_rotation_compensation=no_rotation_compensation,
    )
