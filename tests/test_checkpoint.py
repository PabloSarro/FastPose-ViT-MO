"""Checkpoint/resume (src/checkpoint.py, used by src/train.py).

Interrupting a run, saving last.pt and resuming in a fresh process state must continue
exactly like the uninterrupted run: same weights bit for bit, for Muon (with its auxiliary
Adam) and for Adam with the warm-restart scheduler, with random numbers drawn every step.
"""

import os
import random

import numpy as np
import pytest
import torch

from src.checkpoint import (
    get_rng_state,
    load_checkpoint,
    resolve_resume_path,
    save_checkpoint,
    set_rng_state,
    unwrap_model,
)
from src.optimizer import SingleDeviceMuonWithAuxAdam
from src.scheduler import CosineAnnealingWithWarmRestarts

STEPS, SPLIT = 6, 3


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build(optimizer_name, seed):
    """Model, optimizer and scheduler as in train.py, from a given seed."""
    seed_all(seed)
    model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 9))
    if optimizer_name == "muon":
        params = list(model.parameters())
        optimizer = SingleDeviceMuonWithAuxAdam(
            [
                {"params": [p for p in params if p.ndim >= 2], "use_muon": True, "lr": 1e-2},
                {
                    "params": [p for p in params if p.ndim < 2],
                    "use_muon": False,
                    "lr": 1e-3,
                    "betas": (0.9, 0.95),
                },
            ]
        )
        scheduler = None
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = CosineAnnealingWithWarmRestarts(
            optimizer, 1e-3, 1e-5, T_0=1, T_mult=2, steps_per_epoch=2
        )
    return model, optimizer, scheduler


def step(model, optimizer, scheduler):
    """One training step that consumes Python, NumPy and torch random numbers."""
    x = torch.randn(4, 8) + float(np.random.rand()) + random.random()
    loss = torch.nn.functional.dropout(model(x), p=0.2).pow(2).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if scheduler is not None:
        scheduler.step()


def state(model, optimizer, scheduler, done):
    return {
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": done - 1,
        "global_iteration": done,
        "rng": get_rng_state(),
    }


@pytest.mark.parametrize("optimizer_name", ["muon", "adam"])
def test_resume_continues_bit_for_bit(tmp_path, optimizer_name):
    # Uninterrupted reference run.
    model, optimizer, scheduler = build(optimizer_name, seed=0)
    for _ in range(STEPS):
        step(model, optimizer, scheduler)
    reference = {k: v.clone() for k, v in model.state_dict().items()}

    # Interrupted run: SPLIT steps, checkpoint, then a fresh process state.
    model, optimizer, scheduler = build(optimizer_name, seed=0)
    for _ in range(SPLIT):
        step(model, optimizer, scheduler)
    path = str(tmp_path / "last.pt")
    save_checkpoint(path, state(model, optimizer, scheduler, SPLIT))

    model, optimizer, scheduler = build(optimizer_name, seed=123)  # different init and RNG
    ckpt = load_checkpoint(path)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    set_rng_state(ckpt["rng"])
    assert ckpt["global_iteration"] == SPLIT
    for _ in range(STEPS - SPLIT):
        step(model, optimizer, scheduler)

    for k, v in model.state_dict().items():
        assert torch.equal(v, reference[k]), k


def test_without_rng_restore_the_run_diverges(tmp_path):
    """Sanity check of the test above: skipping the RNG restore changes the result."""
    model, optimizer, _ = build("muon", seed=0)
    for _ in range(STEPS):
        step(model, optimizer, None)
    reference = model[0].weight.clone()
    model, optimizer, _ = build("muon", seed=0)
    for _ in range(SPLIT):
        step(model, optimizer, None)
    ckpt = state(model, optimizer, None, SPLIT)
    model, optimizer, _ = build("muon", seed=123)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    for _ in range(STEPS - SPLIT):
        step(model, optimizer, None)
    assert not torch.equal(model[0].weight, reference)


def test_save_is_atomic_and_replaces(tmp_path):
    path = str(tmp_path / "sub" / "last.pt")
    save_checkpoint(path, {"epoch": 0})
    save_checkpoint(path, {"epoch": 1})
    assert load_checkpoint(path)["epoch"] == 1
    assert sorted(os.listdir(tmp_path / "sub")) == ["last.pt"]  # no .tmp left behind


def test_unwrap_compiled_model(monkeypatch):
    # The container's uid has no passwd entry; torch.compile asks getpass for a user name.
    monkeypatch.setenv("USER", os.environ.get("USER") or "pytest")
    model = torch.nn.Linear(2, 2)
    compiled = torch.compile(model)
    assert unwrap_model(compiled) is model
    assert unwrap_model(model) is model
    assert list(unwrap_model(compiled).state_dict()) == ["weight", "bias"]  # no _orig_mod. prefix


def test_resolve_resume_path(tmp_path):
    ckpt = str(tmp_path / "last.pt")
    assert resolve_resume_path(None, ckpt) is None
    assert resolve_resume_path("auto", ckpt) is None  # nothing yet: start fresh
    save_checkpoint(ckpt, {})
    assert resolve_resume_path("auto", ckpt) == ckpt
    assert resolve_resume_path(ckpt, "elsewhere.pt") == ckpt
    with pytest.raises(FileNotFoundError):
        resolve_resume_path(str(tmp_path / "missing.pt"), ckpt)
