# This file contains the loss functions used to train the models

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import (
    EPS,
    get_apparent_orientation,
    quaternion_to_rotation_matrix,
    translation_to_bbox_relative_translation,
)


class TranslationMSELoss(nn.Module):
    """
    L2 loss for translation vectors
    """

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, bbox: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute the mean squared error loss for translation vectors

        Args:
            pred (torch.Tensor): Predicted translation vectors
            target (torch.Tensor): Ground truth translation vectors
            bbox (torch.Tensor): Bounding box tensor

        Returns:
            torch.Tensor: Mean squared error loss
        """
        return (pred - target).pow(2).mean()


class TranslationMAELoss(nn.Module):
    """
    MAE (L1) loss for translation vectors
    """

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, bbox: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute the MAE loss for translation vectors

        Args:
            pred (torch.Tensor): Predicted translation vectors
            target (torch.Tensor): Ground truth translation vectors
            bbox (torch.Tensor): Bounding box tensor

        Returns:
            torch.Tensor: MAE loss
        """
        return (pred - target).abs().mean()


class RelativeTranslationLoss(nn.Module):
    """
    Relative translation loss: norm2(t_groundtruth - t_prediction) / norm2(t_groundtruth)
    Based on the SPEED+ evaluation metric
    """

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, bbox: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute the relative translation loss

        Args:
            pred (torch.Tensor): Predicted translation vectors
            target (torch.Tensor): Ground truth translation vectors
            bbox (torch.Tensor): Bounding box tensor (unused)

        Returns:
            torch.Tensor: Relative translation loss
        """
        diff = target - pred
        # Use optimized vector norm operations for better performance
        diff_norms = torch.linalg.vector_norm(diff, dim=1)
        target_norms = torch.linalg.vector_norm(target, dim=1)
        relative_errors = diff_norms / target_norms
        return relative_errors.mean()


class DecoupledTranslationLoss(nn.Module):
    """
    Decoupled translation loss on U = (U_x, U_y, U_z): L1(U_x, U_y) + lambda_z * L1(U_z)

    The relative translation loss is dominated by U_z (about 1 with the "site" depth), because
    U_x and U_y are small fractions of the box size. Here the in-plane and depth errors are
    weighted separately.
    """

    def __init__(self, lambda_z: float = 1.0) -> None:
        super().__init__()
        self.lambda_z = lambda_z

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, bbox: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute the decoupled translation loss

        Args:
            pred (torch.Tensor): Predicted U (B, 3)
            target (torch.Tensor): Ground truth U (B, 3)
            bbox (torch.Tensor): Bounding box tensor (unused)

        Returns:
            torch.Tensor: Mean over the batch of |dU_x| + |dU_y| + lambda_z * |dU_z|
        """
        diff = (pred - target).abs()
        return (diff[:, :2].sum(dim=1) + self.lambda_z * diff[:, 2]).mean()


class TranslationFrobeniusLoss(nn.Module):
    """
    Frobenius norm loss for translation vectors (equivalent to L2 norm for vectors)
    """

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, bbox: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute the Frobenius norm loss for translation vectors

        Args:
            pred (torch.Tensor): Predicted translation vectors
            target (torch.Tensor): Ground truth translation vectors
            bbox (torch.Tensor): Bounding box tensor

        Returns:
            torch.Tensor: Frobenius norm loss
        """
        # For vectors, Frobenius norm is equivalent to L2 norm
        return torch.norm(pred - target, p="fro", dim=1).mean()


class TranslationRelativeMSELoss(nn.Module):
    """
    L2 loss for translation vectors relative to the ground truth
    """

    def __init__(self, epsilon: float = 1e-4):
        super(TranslationRelativeMSELoss, self).__init__()
        self.epsilon = epsilon

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, bbox: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute the relative mean squared error loss for translation vectors

        Args:
            pred (torch.Tensor): Predicted translation vectors
            target (torch.Tensor): Ground truth translation vectors
            bbox (torch.Tensor): Bounding box tensor

        Returns:
            torch.Tensor: Relative mean squared error loss
        """
        # Add epsilon to avoid division by very small values
        denominator = target.pow(2).clamp(min=self.epsilon)
        return ((pred - target).pow(2) / denominator).mean()


class TranslationLpNormLoss(nn.Module):
    """
    L_p norm loss for translation vectors with configurable p value
    """

    def __init__(self, p: float = 2.0):
        super().__init__()
        self.p = p

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, bbox: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute the L_p norm loss for translation vectors

        Args:
            pred (torch.Tensor): Predicted translation vectors
            target (torch.Tensor): Ground truth translation vectors
            bbox (torch.Tensor): Bounding box tensor

        Returns:
            torch.Tensor: L_p norm loss
        """
        return torch.norm(pred - target, p=self.p, dim=1).mean()


class TranslationNuclearLoss(nn.Module):
    """
    Nuclear norm (trace norm) loss for translation vectors
    """

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, bbox: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute the nuclear norm loss for translation vectors

        Args:
            pred (torch.Tensor): Predicted translation vectors
            target (torch.Tensor): Ground truth translation vectors
            bbox (torch.Tensor): Bounding box tensor

        Returns:
            torch.Tensor: Nuclear norm loss
        """
        # For vectors, nuclear norm is equivalent to L1 norm
        return torch.norm(pred - target, p="nuc", dim=1).mean()


class RelativeTranslationFrobeniusLoss(nn.Module):
    """
    Relative Frobenius norm loss for translation vectors: ||t_pred - t_target||_F / ||t_target||_F
    """

    def __init__(self, epsilon: float = 1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, bbox: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute the relative Frobenius norm loss for translation vectors

        Args:
            pred (torch.Tensor): Predicted translation vectors (B, 3)
            target (torch.Tensor): Ground truth translation vectors (B, 3)
            bbox (torch.Tensor): Bounding box tensor (unused)

        Returns:
            torch.Tensor: Relative Frobenius norm loss
        """
        diff_frobenius = torch.norm(pred - target, p="fro", dim=1)
        target_frobenius = torch.norm(target, p="fro", dim=1)
        relative_errors = diff_frobenius / (target_frobenius + self.epsilon)
        return relative_errors.mean()


class QuaternionMSELoss(nn.Module):
    """
    L2 loss for quaternions with double cover handling (q ≡ -q)
    """

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, bbox: tuple = None
    ) -> torch.Tensor:
        """
        Compute the mean squared error loss for quaternions, accounting for double cover

        Args:
            pred (torch.Tensor): Predicted quaternions
            target (torch.Tensor): Ground truth quaternions
            bbox (tuple): Bounding box tensor

        Returns:
            torch.Tensor: Mean squared error loss
        """
        # Compute losses for both q and -q representations
        loss_q = (pred - target).pow(2).sum(dim=1)
        loss_neg_q = (pred + target).pow(2).sum(dim=1)

        # Choose the smaller loss (closer quaternion)
        min_loss = torch.min(loss_q, loss_neg_q)
        return min_loss.mean()


class QuaternionFrobeniusLoss(nn.Module):
    """
    Frobenius norm loss for quaternions with double cover handling (q ≡ -q)
    """

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, bbox: tuple = None
    ) -> torch.Tensor:
        """
        Compute the Frobenius norm loss for quaternions, accounting for double cover

        Args:
            pred (torch.Tensor): Predicted quaternions
            target (torch.Tensor): Ground truth quaternions
            bbox (tuple): Bounding box tensor

        Returns:
            torch.Tensor: Frobenius norm loss
        """
        # Compute Frobenius norm for both q and -q representations
        norm_q = torch.norm(pred - target, p="fro", dim=1)
        norm_neg_q = torch.norm(pred + target, p="fro", dim=1)

        # Choose the smaller norm (closer quaternion)
        min_norm = torch.min(norm_q, norm_neg_q)
        return min_norm.mean()


class RelativeQuaternionFrobeniusLoss(nn.Module):
    """
    Relative Frobenius norm loss for quaternions with double cover handling (q ≡ -q)
    """

    def __init__(self, epsilon: float = 1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, bbox: tuple = None
    ) -> torch.Tensor:
        """
        Compute the relative Frobenius norm loss for quaternions, accounting for double cover

        Args:
            pred (torch.Tensor): Predicted quaternions (B, 4)
            target (torch.Tensor): Ground truth quaternions (B, 4)
            bbox (tuple): Bounding box tensor

        Returns:
            torch.Tensor: Relative Frobenius norm loss
        """
        # Compute Frobenius norms for both q and -q representations
        diff_frobenius_q = torch.norm(pred - target, p="fro", dim=1)
        diff_frobenius_neg_q = torch.norm(pred + target, p="fro", dim=1)

        # Choose the smaller difference (closer quaternion)
        min_diff_frobenius = torch.min(diff_frobenius_q, diff_frobenius_neg_q)

        # Compute target norm (same for q and -q)
        target_frobenius = torch.norm(target, p="fro", dim=1)

        # Compute relative errors
        relative_errors = min_diff_frobenius / (target_frobenius + self.epsilon)
        return relative_errors.mean()


class QuaternionGeodesicLoss(nn.Module):
    """
    Geodesic loss for quaternions
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the geodesic loss for quaternions
        NOTE: The normalization of the quaternions is very important to avoid
        having a great loss with non-usable quaternions

        Args:
            pred (torch.Tensor): Predicted quaternions
            target (torch.Tensor): Ground truth quaternions

        Returns:
            torch.Tensor: Geodesic loss
        """
        # Normalize the quaternions (optimized with F.normalize)
        pred = F.normalize(pred, p=2, dim=1)
        target = F.normalize(target, p=2, dim=1)

        # Compute the dot product between the quaternions (optimized with einsum)
        dot_product = torch.einsum("bi,bi->b", pred, target)
        # Use out-of-place operations to preserve gradients
        dot_product = torch.abs(dot_product)
        dot_product = torch.clamp(dot_product, EPS, 1.0 - EPS)

        # Compute the angle between the quaternions
        angle = 2 * torch.acos(dot_product)
        return angle.mean()


class QuaternionSimplifiedGeodesicLoss(nn.Module):
    """
    Simplified eodesic loss for quaternions
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the geodesic loss for quaternions
        NOTE: The normalization of the quaternions is very important to avoid
        having a great loss with non-usable quaternions

        Args:
            pred (torch.Tensor): Predicted quaternions
            target (torch.Tensor): Ground truth quaternions

        Returns:
            torch.Tensor: Simplified geodesic loss
        """
        # Normalize the quaternions
        pred = pred / torch.norm(pred, p=2, dim=1, keepdim=True)
        target = target / torch.norm(target, p=2, dim=1, keepdim=True)

        # Compute the dot product between the quaternions
        dot_product = torch.sum(pred * target, dim=1)

        return (1 - torch.abs(dot_product)).mean()


class PenalizeNonUnitQuaternionLoss(nn.Module):
    """
    Penalize quaternions that are not unit norm
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Penalize quaternions that are not unit norm

        Args:
            x (torch.Tensor): Quaternions

        Returns:
            torch.Tensor: Penalty (norm - 1)^2
        """
        # Ensure that the quaternion has unit norm
        return (torch.norm(x, p=2, dim=1) - 1).pow(2).mean()


class RotationMatrixMSELoss(nn.Module):
    """
    MSE loss for rotation matrices
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the mean squared error loss for rotation matrices

        Args:
            pred (torch.Tensor): Predicted rotation matrices
            target (torch.Tensor): Ground truth rotation matrices

        Returns:
            torch.Tensor: Mean squared error loss
        """
        return (pred - target).pow(2).mean()


class RotationMatrixFrobeniusLoss(nn.Module):
    """
    Frobenius loss for rotation matrices
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the Frobenius loss for rotation matrices

        Args:
            pred (torch.Tensor): Predicted rotation matrices
            target (torch.Tensor): Ground truth rotation matrices

        Returns:
            torch.Tensor: Frobenius loss
        """
        return torch.norm(pred - target, p="fro", dim=(1, 2)).mean()


class RotationMatrixMAELoss(nn.Module):
    """
    MAE (L1) loss for rotation matrices
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the MAE loss for rotation matrices

        Args:
            pred (torch.Tensor): Predicted rotation matrices
            target (torch.Tensor): Ground truth rotation matrices

        Returns:
            torch.Tensor: MAE loss
        """
        return torch.norm(pred - target, p=1, dim=(1, 2)).mean()


class RotationMatrixLpNormLoss(nn.Module):
    """
    L_p norm loss for rotation matrices with configurable p value
    """

    def __init__(self, p: float = 2.0):
        super().__init__()
        self.p = p

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the L_p norm loss for rotation matrices

        Args:
            pred (torch.Tensor): Predicted rotation matrices (B, 3, 3)
            target (torch.Tensor): Ground truth rotation matrices (B, 3, 3)

        Returns:
            torch.Tensor: L_p norm loss
        """
        return torch.norm(pred - target, p=self.p, dim=(1, 2)).mean()


class RotationMatrixNuclearLoss(nn.Module):
    """
    Nuclear norm (trace norm) loss for rotation matrices
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the nuclear norm loss for rotation matrices

        Args:
            pred (torch.Tensor): Predicted rotation matrices (B, 3, 3)
            target (torch.Tensor): Ground truth rotation matrices (B, 3, 3)

        Returns:
            torch.Tensor: Nuclear norm loss
        """
        return torch.norm(pred - target, p="nuc", dim=(1, 2)).mean()


class Rotation6DLoss(nn.Module):
    """
    Loss computed directly on 6D rotation representation before matrix conversion.
    This avoids numerical instabilities in the Gram-Schmidt orthogonalization process.
    """

    def forward(self, pred_6d: torch.Tensor, target_6d: torch.Tensor) -> torch.Tensor:
        """
        Compute loss directly on 6D rotation vectors

        Args:
            pred_6d (torch.Tensor): Predicted 6D rotation vectors (batch_size, 6)
            target_6d (torch.Tensor): Ground truth 6D rotation vectors (batch_size, 6)

        Returns:
            torch.Tensor: L2 loss on 6D vectors
        """
        return nn.functional.mse_loss(pred_6d, target_6d)


class Rotation6DMAELoss(nn.Module):
    """
    MAE loss computed directly on 6D rotation representation before matrix conversion.
    """

    def forward(self, pred_6d: torch.Tensor, target_6d: torch.Tensor) -> torch.Tensor:
        """
        Compute MAE loss directly on 6D rotation vectors

        Args:
            pred_6d (torch.Tensor): Predicted 6D rotation vectors (batch_size, 6)
            target_6d (torch.Tensor): Ground truth 6D rotation vectors (batch_size, 6)

        Returns:
            torch.Tensor: MAE loss on 6D vectors
        """
        return nn.functional.l1_loss(pred_6d, target_6d)


class Rotation6DFrobeniusLoss(nn.Module):
    """
    Frobenius norm loss computed directly on 6D rotation representation before matrix conversion.
    """

    def forward(self, pred_6d: torch.Tensor, target_6d: torch.Tensor) -> torch.Tensor:
        """
        Compute Frobenius norm loss directly on 6D rotation vectors

        Args:
            pred_6d (torch.Tensor): Predicted 6D rotation vectors (batch_size, 6)
            target_6d (torch.Tensor): Ground truth 6D rotation vectors (batch_size, 6)

        Returns:
            torch.Tensor: Frobenius norm loss on 6D vectors
        """
        return torch.norm(pred_6d - target_6d, p="fro", dim=1).mean()


class Rotation6DMSELoss(nn.Module):
    """
    MSE loss computed directly on 6D rotation representation before matrix conversion.
    """

    def forward(self, pred_6d: torch.Tensor, target_6d: torch.Tensor) -> torch.Tensor:
        """
        Compute MSE loss directly on 6D rotation vectors

        Args:
            pred_6d (torch.Tensor): Predicted 6D rotation vectors (batch_size, 6)
            target_6d (torch.Tensor): Ground truth 6D rotation vectors (batch_size, 6)

        Returns:
            torch.Tensor: MSE loss on 6D vectors
        """
        return nn.functional.mse_loss(pred_6d, target_6d)


class Rotation6DLpNormLoss(nn.Module):
    """
    L_p norm loss computed directly on 6D rotation representation with configurable p value
    """

    def __init__(self, p: float = 2.0):
        super().__init__()
        self.p = p

    def forward(self, pred_6d: torch.Tensor, target_6d: torch.Tensor) -> torch.Tensor:
        """
        Compute L_p norm loss directly on 6D rotation vectors

        Args:
            pred_6d (torch.Tensor): Predicted 6D rotation vectors (batch_size, 6)
            target_6d (torch.Tensor): Ground truth 6D rotation vectors (batch_size, 6)

        Returns:
            torch.Tensor: L_p norm loss on 6D vectors
        """
        return torch.norm(pred_6d - target_6d, p=self.p, dim=1).mean()


class Rotation6DNuclearLoss(nn.Module):
    """
    Nuclear norm (trace norm) loss computed directly on 6D rotation representation
    """

    def forward(self, pred_6d: torch.Tensor, target_6d: torch.Tensor) -> torch.Tensor:
        """
        Compute nuclear norm loss directly on 6D rotation vectors

        Args:
            pred_6d (torch.Tensor): Predicted 6D rotation vectors (batch_size, 6)
            target_6d (torch.Tensor): Ground truth 6D rotation vectors (batch_size, 6)

        Returns:
            torch.Tensor: Nuclear norm loss on 6D vectors
        """
        # For 6D vectors, reshape to matrix form for nuclear norm
        diff = pred_6d - target_6d
        diff_matrix = diff.view(-1, 2, 3)  # Reshape to (batch_size, 2, 3) matrices
        return torch.norm(diff_matrix, p="nuc", dim=(1, 2)).mean()


class RotationMatrixGeodesicLoss(nn.Module):
    """
    Geodesic loss for rotation matrices
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the geodesic loss for rotation matrices

        Args:
            pred (torch.Tensor): Predicted rotation matrices
            target (torch.Tensor): Ground truth rotation matrices

        Returns:
            torch.Tensor: Geodesic loss
        """
        R_diff = torch.bmm(pred, target.transpose(1, 2))

        # Define an epsilon
        epsilon = 1e-7

        # Clamp the values to avoid numerical instability
        cos_theta = torch.clamp(
            (R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2] - 1) / 2,
            -1.0 + epsilon,
            1.0 - epsilon,
        )

        # Compute the angle (in radians)
        theta = torch.acos(cos_theta)

        return theta.mean()


class RotationMatrixGeodesicLossNoClip(nn.Module):
    """
    Geodesic loss for rotation matrices without clipping the cosine value. Used for debugging purposes
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the geodesic loss for rotation matrices

        Args:
            pred (torch.Tensor): Predicted rotation matrices
            target (torch.Tensor): Ground truth rotation matrices

        Returns:
            torch.Tensor: Geodesic loss
        """
        R_diff = torch.bmm(pred, target.transpose(1, 2))

        # Clamp the values to avoid numerical instability
        cos_theta = (R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2] - 1) / 2

        # Compute the angle (in radians)
        theta = torch.acos(cos_theta)

        return theta.mean()


class PenalizeBadBasisLoss(nn.Module):
    """
    Penalize bad basis vectors in the rotation matrix
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Penalize bad basis vectors in the rotation matrix
        We extract the two first basis vectors in the rotation matrix and penalize them for not having norm 1
        and for not being orthogonal to each other

        Args:
            x (torch.Tensor): Rotation matrix

        Returns:
            torch.Tensor: Penalty (norm - 1)^2 + dot_product^2
        """
        v1, v2 = torch.split(x, 3, dim=1)

        # v1 and v2 are penalized for not having norm 1
        penalty = (torch.norm(v1, p=2, dim=1) - 1).pow(2).mean() + (
            torch.norm(v2, p=2, dim=1) - 1
        ).pow(2).mean()

        # Penalize for not being orthogonal
        penalty += torch.sum(v1 * v2, dim=1).pow(2).mean()

        return penalty


class RelativeRotationFrobeniusLoss(nn.Module):
    """
    Relative Frobenius norm loss for rotation matrices: ||R_pred - R_target||_F / ||R_target||_F
    """

    def __init__(self, epsilon: float = 1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the relative Frobenius norm loss for rotation matrices

        Args:
            pred (torch.Tensor): Predicted rotation matrices (B, 3, 3)
            target (torch.Tensor): Ground truth rotation matrices (B, 3, 3)

        Returns:
            torch.Tensor: Relative Frobenius norm loss
        """
        diff_frobenius = torch.norm(pred - target, p="fro", dim=(1, 2))
        target_frobenius = torch.norm(target, p="fro", dim=(1, 2))
        relative_errors = diff_frobenius / (target_frobenius + self.epsilon)
        return relative_errors.mean()


class RelativeRotation6DFrobeniusLoss(nn.Module):
    """
    Relative Frobenius norm loss for 6D rotation representation: ||6D_pred - 6D_target||_F / ||6D_target||_F
    """

    def __init__(self, epsilon: float = 1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred_6d: torch.Tensor, target_6d: torch.Tensor) -> torch.Tensor:
        """
        Compute the relative Frobenius norm loss for 6D rotation vectors

        Args:
            pred_6d (torch.Tensor): Predicted 6D rotation vectors (batch_size, 6)
            target_6d (torch.Tensor): Ground truth 6D rotation vectors (batch_size, 6)

        Returns:
            torch.Tensor: Relative Frobenius norm loss
        """
        diff_frobenius = torch.norm(pred_6d - target_6d, p="fro", dim=1)
        target_frobenius = torch.norm(target_6d, p="fro", dim=1)
        relative_errors = diff_frobenius / (target_frobenius + self.epsilon)
        return relative_errors.mean()


class SymmetricRotation6DLoss(nn.Module):
    """
    Symmetry-aware relative 6D Frobenius loss (M5): for each sample, the smallest
    ||6D_pred - 6D(R_target·S_k)|| / ||6D(R_target·S_k)|| over its object's symmetries S_k.

    Symmetries multiply on the right (model frame) and the allocentric change on the left, so this
    works directly on allocentric targets. For an object with the identity only it equals
    RelativeRotation6DFrobeniusLoss.

    A symmetry can also shift the model (x -> S_k·x + s_k, e.g. ~19 mm for the YCB-V clamps): the
    equivalent pose is then (R·S_k, t + R·s_k), with its own translation label and allocentric
    change. pose_loss() handles that (M8): it builds every symmetric pose's labels from the
    original BOP pose, takes the rotation minimum, and returns the translation label of the chosen
    symmetry for the translation loss.
    """

    def __init__(
        self,
        sym_table: torch.Tensor,
        sym_offsets: torch.Tensor | None = None,
        depth_mode: str = "site",
        no_rotation_compensation: bool = False,
        epsilon: float = 1e-8,
    ):
        """
        Args:
            sym_table (torch.Tensor): (n_obj, K, 3, 3) symmetry rotations of every object, padded
                with the identity (src.bop_dataset.symmetry_table)
            sym_offsets (torch.Tensor): (n_obj, K, 3) their translation offsets in mm, or None
                (all zero)
            depth_mode, no_rotation_compensation: As the dataset's labels (src.bop_dataset)
            epsilon (float): Added to the target norm
        """
        super().__init__()
        sym_table = torch.as_tensor(sym_table, dtype=torch.float32)
        if sym_offsets is None:
            sym_offsets = torch.zeros(sym_table.shape[:-1])
        self.register_buffer("sym_table", sym_table)
        self.register_buffer("sym_offsets", torch.as_tensor(sym_offsets, dtype=torch.float32))
        self.has_offsets = bool((self.sym_offsets.abs() > 0).any())  # use pose_loss() then
        self.depth_mode = depth_mode
        self.no_rotation_compensation = no_rotation_compensation
        self.epsilon = epsilon

    def _min_over_targets(self, pred_6d: torch.Tensor, targets: torch.Tensor):
        """Per-sample min over K target rotations (B, K, 3, 3): (mean of the min, argmin (B,))."""
        targets_6d = torch.cat([targets[..., :, 0], targets[..., :, 1]], dim=-1)  # (B, K, 6)
        diff = torch.norm(pred_6d.float().unsqueeze(1) - targets_6d, dim=-1)
        relative = diff / (torch.norm(targets_6d, dim=-1) + self.epsilon)
        best = relative.min(dim=1)
        return best.values.mean(), best.indices

    def forward(
        self, pred_6d: torch.Tensor, target_R: torch.Tensor, obj_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Rotation part only (ignores the offsets).

        Args:
            pred_6d (torch.Tensor): Predicted 6D rotation vectors (B, 6)
            target_R (torch.Tensor): Ground truth rotation matrices (B, 3, 3)
            obj_idx (torch.Tensor): Object index of each sample (B,)

        Returns:
            torch.Tensor: Mean over the batch of the per-sample minimum
        """
        targets = target_R.float().unsqueeze(1) @ self.sym_table[obj_idx]  # (B, K, 3, 3)
        return self._min_over_targets(pred_6d, targets)[0]

    def symmetric_labels(
        self, obj_idx, R_m2c, t_m2c, K, bboxes, img_wh, diameter
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Labels of every symmetric pose (R·S_k, t + R·s_k), as BOPDataset.labels computes them.

        Args:
            obj_idx (B,); R_m2c (B, 3, 3), t_m2c (B, 3) mm: the original BOP pose; K (B, 3, 3);
            bboxes (B, 4) crop bbox; img_wh (B, 2); diameter (B,) mm

        Returns:
            U (B, K, 3) translation labels and allocentric rotations (B, K, 3, 3), in float64
        """
        B, n_sym = obj_idx.shape[0], self.sym_table.shape[1]
        f64 = dict(dtype=torch.float64, device=self.sym_table.device)
        R, t = R_m2c.to(**f64), t_m2c.to(**f64)
        R_k = R.unsqueeze(1) @ self.sym_table[obj_idx].double()  # (B, K, 3, 3)
        t_k = t.unsqueeze(1) + (R.unsqueeze(1) @ self.sym_offsets[obj_idx].double()[..., None])[
            ..., 0
        ]  # (B, K, 3)
        flat_t = t_k.reshape(-1, 3)

        def rep(x):
            x = x.to(**f64)
            return x.repeat_interleave(n_sym, dim=0)

        U = translation_to_bbox_relative_translation(
            flat_t,
            rep(bboxes),
            None,
            K=rep(K),
            img_wh=rep(img_wh),
            depth_mode=self.depth_mode,
            diameter=rep(diameter),
        ).reshape(B, n_sym, 3)
        # The allocentric change of a pose depends on its translation only: R_rel(t)^T = the
        # apparent orientation of the identity
        identity = torch.tensor([1.0, 0.0, 0.0, 0.0], **f64).expand(B * n_sym, 4)
        R_rel_T = quaternion_to_rotation_matrix(
            get_apparent_orientation(flat_t, identity, self.no_rotation_compensation)
        ).to(**f64)
        rotations = R_rel_T.reshape(B, n_sym, 3, 3) @ R_k
        return U, rotations

    def pose_loss(
        self, pred_6d, obj_idx, R_m2c, t_m2c, K, bboxes, img_wh, diameter
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Rotation loss over the full symmetric poses (offsets included).

        Args:
            pred_6d (torch.Tensor): Predicted 6D rotation vectors (B, 6)
            obj_idx, R_m2c, t_m2c, K, bboxes, img_wh, diameter: see symmetric_labels

        Returns:
            The rotation loss (mean of the per-sample min) and the translation label U (B, 3) of
            each sample's chosen symmetry, for the translation loss
        """
        device_type = self.sym_table.device.type
        with torch.no_grad(), torch.autocast(device_type, enabled=False):
            U, rotations = self.symmetric_labels(
                obj_idx, R_m2c, t_m2c, K, bboxes, img_wh, diameter
            )
        loss, k = self._min_over_targets(pred_6d, rotations.float())
        U_best = U[torch.arange(len(k), device=k.device), k].float()
        return loss, U_best


class Identity(nn.Module):
    """
    Identity loss

    This loss does nothing and is used for debugging purposes
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
