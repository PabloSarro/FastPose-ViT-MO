"""Safe loading of checkpoint files (audit W9, 2026-11-14).

torch.load(weights_only=False) runs arbitrary pickle code: acceptable for our own files, not for
files that come from elsewhere (the LW-DETR COCO weights are downloaded from a third-party
Hugging Face repository). torch.load(weights_only=True) only builds tensors and plain containers;
our training checkpoints also hold the run's arguments (argparse.Namespace) and numpy RNG states,
which are data-only types that we allowlist explicitly. Anything else is refused.
"""

import argparse

import numpy as np
import torch

SAFE_TYPES = [
    argparse.Namespace,
    np.ndarray,
    np.dtype,
    np._core.multiarray._reconstruct,
    *[getattr(np.dtypes, n) for n in dir(np.dtypes) if n.endswith("DType")],
]


def load_safe(path: str, map_location="cpu"):
    """torch.load with weights_only=True plus the allowlist above (no arbitrary code)."""
    with torch.serialization.safe_globals(SAFE_TYPES):
        return torch.load(path, map_location=map_location, weights_only=True)
