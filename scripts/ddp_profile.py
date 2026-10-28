# This file runs src/train.py unchanged and reports, per epoch and per rank, where the training
# step's time goes (M6 multi-GPU speed checks): how long the step waited for the DataLoader, and
# the CPU time of the optimizer step. It patches only DataLoader.__iter__ and the optimizer's
# step from the outside, so the training itself is the same as with src/train.py.
#
#   python -m torch.distributed.run --nproc_per_node=2 scripts/ddp_profile.py <train.py args>
#   (used by scripts/ddp_bench.sh)

import os
import runpy
import sys
import time

import torch.utils.data as tud

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.optimizer as optimizer_module  # noqa: E402

RANK = int(os.environ.get("RANK", "0"))
stats = {"opt_seconds": 0.0, "opt_steps": 0}

_step = optimizer_module.SingleDeviceMuonWithAuxAdam.step


def _timed_step(self, *args, **kwargs):
    start = time.perf_counter()
    out = _step(self, *args, **kwargs)
    stats["opt_seconds"] += time.perf_counter() - start
    stats["opt_steps"] += 1
    return out


optimizer_module.SingleDeviceMuonWithAuxAdam.step = _timed_step

_iter = tud.DataLoader.__iter__


def _timed_iter(self):
    it = _iter(self)
    if len(self) < 100:  # validation loader
        return it

    def timed():
        wait, n, start = 0.0, 0, time.perf_counter()
        stats["opt_seconds"], stats["opt_steps"] = 0.0, 0
        while True:
            t0 = time.perf_counter()
            try:
                batch = next(it)
            except StopIteration:
                break
            if n >= 5:  # the first batches include the workers' start
                wait += time.perf_counter() - t0
            n += 1
            yield batch
        total = time.perf_counter() - start
        print(
            f"PROFILE rank {RANK}: {n} batches in {total:.1f} s ({1000 * total / max(n, 1):.1f} "
            f"ms/step), data wait {1000 * wait / max(n - 5, 1):.1f} ms/step, optimizer step "
            f"(CPU) {1000 * stats['opt_seconds'] / max(stats['opt_steps'], 1):.1f} ms/step",
            flush=True,
        )

    return timed()


tud.DataLoader.__iter__ = _timed_iter
sys.argv = ["src/train.py"] + sys.argv[1:]
runpy.run_path("src/train.py", run_name="__main__")
