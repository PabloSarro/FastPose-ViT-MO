# This file contains the multi-GPU (DDP) helpers of src/train.py (M6)
#
# A run started with torchrun (python -m torch.distributed.run --nproc_per_node=N src/train.py ...)
# gets one process per GPU; a plain `python src/train.py` is a single process and none of this
# changes anything (world size 1).

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistInfo:
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def init_distributed() -> DistInfo:
    """Join the process group when started by torchrun (WORLD_SIZE > 1), else world size 1.

    NCCL on GPUs (one GPU per process, cuda:LOCAL_RANK), gloo on the CPU.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return DistInfo()
    info = DistInfo(int(os.environ["RANK"]), world_size, int(os.environ.get("LOCAL_RANK", "0")))
    if torch.cuda.is_available():
        torch.cuda.set_device(info.local_rank)
        dist.init_process_group("nccl", device_id=torch.device("cuda", info.local_rank))
    else:
        dist.init_process_group("gloo")
    return info


def cleanup_distributed(info: DistInfo) -> None:
    if info.enabled and dist.is_initialized():
        dist.destroy_process_group()


def barrier(info: DistInfo) -> None:
    if info.enabled:
        dist.barrier()


def all_reduce_mean(values: list[float], info: DistInfo, device: torch.device) -> list[float]:
    """Mean of each value over the ranks (the values themselves with world size 1)."""
    if not info.enabled:
        return values
    t = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(t)
    return (t / info.world_size).tolist()


def all_gather_object(obj, info: DistInfo) -> list:
    """[obj of rank 0, obj of rank 1, ...] on every rank ([obj] with world size 1)."""
    if not info.enabled:
        return [obj]
    out = [None] * info.world_size
    dist.all_gather_object(out, obj)
    return out


def shard(indices: list, rank: int, world_size: int) -> list:
    """This rank's share of an epoch's sample order: every world_size-th index from rank on.

    The order is cut to a multiple of world_size first, so every rank gets the same number of
    samples (and of batches) and the shards are disjoint.
    """
    n = len(indices) - len(indices) % world_size
    return indices[rank:n:world_size]
