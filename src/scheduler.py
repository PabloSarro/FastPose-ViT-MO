import math
import torch
import numpy as np


class LogarithmicLR(torch.optim.lr_scheduler._LRScheduler):
    """Learning rate scheduler with logarithmic decay from max_lr to min_lr."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_lr: float,
        min_lr: float,
        T_max_epochs: int,
        steps_per_epoch: int = 1,
        last_epoch: int = -1,
    ) -> None:
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.T_max_epochs = T_max_epochs  # Total epochs (user-friendly)
        self.T_max = T_max_epochs * steps_per_epoch  # Convert to steps for internal computation
        self.steps_per_epoch = steps_per_epoch  # For converting epochs to steps
        # Precompute logarithmic values to avoid repeated expensive operations
        self.log_max = math.log(max_lr)
        self.log_min = math.log(min_lr)
        self.log_diff = self.log_min - self.log_max
        self.inv_T_max = 1.0 / self.T_max  # Precompute division
        # Cache for last computed values to avoid redundant calculations
        self._last_computed_step = -1
        self._cached_lr = max_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        """Compute current learning rate based on training progress.

        Returns:
            List of learning rates for each parameter group.
        """
        # Use cached value if step hasn't changed (called multiple times per step)
        if self.last_epoch == self._last_computed_step:
            return [self._cached_lr for _ in self.base_lrs]

        if self.last_epoch == 0:
            self._cached_lr = self.max_lr
        else:
            # Optimized computation with precomputed values
            progress = min(self.last_epoch * self.inv_T_max, 1.0)
            log_lr = self.log_max + self.log_diff * progress
            self._cached_lr = math.exp(log_lr)

        self._last_computed_step = self.last_epoch
        return [self._cached_lr for _ in self.base_lrs]

    @staticmethod
    def compute_all_lrs(
        max_lr: float, min_lr: float, T_max_epochs: int, steps_per_epoch: int = 1
    ) -> list[float]:
        """Pre-compute all learning rates for the entire training run.

        Args:
            max_lr: Maximum learning rate.
            min_lr: Minimum learning rate.
            T_max_epochs: Total number of epochs.
            steps_per_epoch: Number of optimizer steps per epoch.

        Returns:
            List of learning rates for each step.
        """
        T_max_steps = T_max_epochs * steps_per_epoch  # Convert epochs to steps

        log_max = math.log(max_lr)
        log_min = math.log(min_lr)
        log_diff = log_min - log_max
        inv_steps = 1.0 / T_max_steps

        # Vectorized computation
        steps_array = np.arange(T_max_steps)
        progress_array = np.minimum(steps_array * inv_steps, 1.0)
        log_lr_array = log_max + log_diff * progress_array
        lrs = np.exp(log_lr_array).tolist()

        return lrs


class CosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    """Learning rate scheduler with cosine annealing from max_lr to min_lr."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_lr: float,
        min_lr: float,
        T_max_epochs: int,
        steps_per_epoch: int = 1,
        last_epoch: int = -1,
    ) -> None:
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.T_max_epochs = T_max_epochs  # Total epochs (user-friendly)
        self.T_max = T_max_epochs * steps_per_epoch  # Convert to steps for internal computation
        self.steps_per_epoch = steps_per_epoch  # For converting epochs to steps
        # Precompute constants for cosine annealing
        self.lr_range = max_lr - min_lr
        self.inv_T_max = 1.0 / self.T_max  # Precompute division
        # Cache for last computed values to avoid redundant calculations
        self._last_computed_step = -1
        self._cached_lr = max_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        """Compute current learning rate based on training progress.

        Returns:
            List of learning rates for each parameter group.
        """
        # Use cached value if step hasn't changed (called multiple times per step)
        if self.last_epoch == self._last_computed_step:
            return [self._cached_lr for _ in self.base_lrs]

        if self.last_epoch == 0:
            self._cached_lr = self.max_lr
        else:
            # Optimized cosine computation
            progress = min(self.last_epoch * self.inv_T_max, 1.0)
            self._cached_lr = self.min_lr + self.lr_range * 0.5 * (
                1 + math.cos(math.pi * progress)
            )

        self._last_computed_step = self.last_epoch
        return [self._cached_lr for _ in self.base_lrs]

    @staticmethod
    def compute_all_lrs(
        max_lr: float, min_lr: float, T_max_epochs: int, steps_per_epoch: int = 1
    ) -> list[float]:
        """Pre-compute all learning rates for the entire training run.

        Args:
            max_lr: Maximum learning rate.
            min_lr: Minimum learning rate.
            T_max_epochs: Total number of epochs.
            steps_per_epoch: Number of optimizer steps per epoch.

        Returns:
            List of learning rates for each step.
        """
        T_max_steps = T_max_epochs * steps_per_epoch  # Convert epochs to steps

        steps_array = np.arange(T_max_steps)
        lr_range = max_lr - min_lr
        inv_steps = 1.0 / T_max_steps

        # Vectorized cosine computation
        progress_array = steps_array * inv_steps
        cos_values = np.cos(math.pi * progress_array)
        lrs = (min_lr + lr_range * 0.5 * (1 + cos_values)).tolist()

        return lrs


class CosineAnnealingWithWarmRestarts(torch.optim.lr_scheduler._LRScheduler):
    """Cosine annealing with warm restarts and exponentially increasing period."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_lr: float,
        min_lr: float,
        T_0: int = 10,
        T_mult: int = 2,
        steps_per_epoch: int = 1,
        last_epoch: int = -1,
    ) -> None:
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.T_0_epochs = T_0  # Initial restart period in epochs (user-friendly)
        self.T_0 = T_0 * steps_per_epoch  # Convert to steps for internal computation
        self.T_mult = T_mult  # Restart period multiplier
        self.steps_per_epoch = steps_per_epoch  # For converting epochs to steps
        # Cache for last computed values to avoid redundant calculations
        self._last_computed_step = -1
        self._cached_lr = max_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        """Compute current learning rate based on position within restart cycle.

        Returns:
            List of learning rates for each parameter group.
        """
        # Use cached value if step hasn't changed (called multiple times per step)
        if self.last_epoch == self._last_computed_step:
            return [self._cached_lr for _ in self.base_lrs]

        if self.last_epoch == 0:
            self._cached_lr = self.max_lr
        else:
            # Compute current position within the current restart cycle
            T_cur = self.last_epoch
            T_i = self.T_0
            while T_cur >= T_i:
                T_cur -= T_i
                T_i *= self.T_mult
            progress = T_cur / T_i
            self._cached_lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (
                1 + math.cos(math.pi * progress)
            )

        self._last_computed_step = self.last_epoch
        return [self._cached_lr for _ in self.base_lrs]

    @staticmethod
    def compute_all_lrs(
        max_lr: float,
        min_lr: float,
        T_max: int,
        T_0_epochs: int = 10,
        T_mult: int = 2,
        steps_per_epoch: int = 1,
    ) -> list[float]:
        """Pre-compute all learning rates for the entire training run.

        Args:
            max_lr: Maximum learning rate.
            min_lr: Minimum learning rate.
            T_max: Total number of steps.
            T_0_epochs: Initial restart period in epochs.
            T_mult: Restart period multiplier.
            steps_per_epoch: Number of optimizer steps per epoch.

        Returns:
            List of learning rates for each step.
        """
        T_0_steps = T_0_epochs * steps_per_epoch  # Convert epochs to steps

        lrs = []
        for step_idx in range(T_max):
            T_cur = step_idx
            T_i = T_0_steps
            while T_cur >= T_i:
                T_cur -= T_i
                T_i *= T_mult
            progress = T_cur / T_i
            lr = min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
            lrs.append(lr)
        return lrs
