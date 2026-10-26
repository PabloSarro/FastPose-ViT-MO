# This file contains the training checkpoint helpers used to resume interrupted runs

import os
import random

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel


def get_rng_state() -> dict:
    """Capture the Python, NumPy, torch and CUDA random number generator states.

    Returns:
        Dict with one entry per generator.
    """
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def set_rng_state(state: dict) -> None:
    """Restore the generator states captured by get_rng_state.

    Args:
        state: Dict returned by get_rng_state.
    """
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the plain model inside torch.compile and/or DistributedDataParallel wrappers."""
    model = getattr(model, "_orig_mod", model)
    return model.module if isinstance(model, DistributedDataParallel) else model


def save_checkpoint(path: str, checkpoint: dict) -> None:
    """Save a checkpoint atomically: write a temporary file, then rename it.

    A job killed while saving leaves the previous checkpoint intact.

    Args:
        path: Destination file (e.g. $RUN_DIR/last.pt).
        checkpoint: Dict to save with torch.save.
    """
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    tmp_path = f"{path}.tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, path)


def load_checkpoint(path: str) -> dict:
    """Load a checkpoint written by save_checkpoint, on the CPU.

    Args:
        path: Checkpoint file.

    Returns:
        The checkpoint dict.
    """
    # weights_only=False: the checkpoint also holds RNG states and arguments (our own file)
    return torch.load(path, map_location="cpu", weights_only=False)


def resolve_resume_path(resume: str | None, checkpoint_path: str) -> str | None:
    """Decide which checkpoint to resume from.

    Args:
        resume: None (start fresh), "auto" (checkpoint_path if it exists, else start fresh),
            or the path of a checkpoint that must exist.
        checkpoint_path: Where this run saves its checkpoint.

    Returns:
        Path of the checkpoint to load, or None to start fresh.
    """
    if resume is None:
        return None
    if resume == "auto":
        return checkpoint_path if os.path.exists(checkpoint_path) else None
    if not os.path.exists(resume):
        raise FileNotFoundError(f"Checkpoint to resume from not found: {resume}")
    return resume
