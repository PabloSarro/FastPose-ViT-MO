"""DDP (M6): sharded samplers, and a real 2-process run on the CPU (gloo backend).

- Each rank gets a disjoint, equal share of the epoch order; with world size 1 nothing changes.
- With DDP and the single-device Muon (+ aux Adam) used by train.py, both ranks end with identical
  weights, equal to one process training on the full batch (DDP averages the gradients).
"""

import os

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from src.bop_dataset import ObjectBalancedSampler
from src.checkpoint import unwrap_model
from src.distributed import DistInfo, all_gather_object, all_reduce_mean, shard
from src.optimizer import SingleDeviceMuonWithAuxAdam

WORLD = 2
STEPS = 3
BATCH = 8  # global batch: 4 samples per rank


def test_shard_is_disjoint_equal_and_complete():
    order = list(range(11))
    shards = [shard(order, r, 3) for r in range(3)]
    assert [len(s) for s in shards] == [3, 3, 3]
    assert sorted(sum(shards, [])) == order[:9]  # the last 11 % 3 = 2 samples are dropped
    assert shard(order, 0, 1) == order


def test_balanced_sampler_shards():
    obj_idx = np.repeat([0, 1, 2], [40, 50, 30])
    full = ObjectBalancedSampler(obj_idx, 60, seed=3)
    ranks = [ObjectBalancedSampler(obj_idx, 60, seed=3, rank=r, world_size=WORLD) for r in range(WORLD)]
    for epoch in (0, 5):
        for s in (full, *ranks):
            s.set_epoch(epoch)
        parts = [list(s) for s in ranks]
        assert [len(p) for p in parts] == [len(s) for s in ranks] == [30, 30]
        assert not set(parts[0]) & set(parts[1])
        assert sorted(parts[0] + parts[1]) == sorted(list(full))  # together: the 1-GPU epoch


def _model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(6, 16), nn.ReLU(), nn.Linear(16, 3))


def _optimizer(model):
    params = list(model.parameters())
    return SingleDeviceMuonWithAuxAdam(
        [
            {"params": [p for p in params if p.ndim >= 2], "use_muon": True, "lr": 0.02},
            {"params": [p for p in params if p.ndim < 2], "use_muon": False, "lr": 0.002},
        ]
    )


def _data():
    g = torch.Generator().manual_seed(1)
    return torch.randn(STEPS, BATCH, 6, generator=g), torch.randn(STEPS, BATCH, 3, generator=g)


def _train(model, xs, ys, rank=0, world=1):
    opt = _optimizer(model)
    per_rank = BATCH // world
    for x, y in zip(xs, ys):
        sl = slice(rank * per_rank, (rank + 1) * per_rank)
        loss = ((model(x[sl]) - y[sl]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.item()


def _worker(rank, init_file, out_dir):
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=WORLD)
    info = DistInfo(rank=rank, world_size=WORLD)
    model = DistributedDataParallel(_model())
    assert isinstance(unwrap_model(model), nn.Sequential)
    xs, ys = _data()
    loss = _train(model, xs, ys, rank, WORLD)
    mean_loss = all_reduce_mean([loss], info, torch.device("cpu"))[0]
    ranks = all_gather_object(rank, info)
    torch.save(
        {"state": unwrap_model(model).state_dict(), "mean_loss": mean_loss, "ranks": ranks},
        os.path.join(out_dir, f"rank{rank}.pt"),
    )
    dist.destroy_process_group()


def test_ddp_two_processes_match_one_process(tmp_path):
    mp.spawn(_worker, args=(str(tmp_path / "init"), str(tmp_path)), nprocs=WORLD, join=True)
    r0, r1 = (torch.load(tmp_path / f"rank{r}.pt") for r in range(WORLD))
    assert r0["ranks"] == r1["ranks"] == [0, 1]
    assert r0["mean_loss"] == pytest.approx(r1["mean_loss"])
    for k in r0["state"]:
        torch.testing.assert_close(r0["state"][k], r1["state"][k], rtol=0, atol=0)  # identical
    single = _model()
    xs, ys = _data()
    _train(single, xs, ys)
    for k, v in single.state_dict().items():
        # Same updates as one process on the full batch (measured gap ~3e-10: summation order)
        torch.testing.assert_close(r0["state"][k], v, rtol=0, atol=1e-6)


def test_world_size_one_helpers_are_no_ops():
    info = DistInfo()
    assert not info.enabled and info.is_main
    assert all_reduce_mean([1.5, 2.0], info, torch.device("cpu")) == [1.5, 2.0]
    assert all_gather_object("x", info) == ["x"]
