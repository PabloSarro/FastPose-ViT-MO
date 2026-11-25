"""scripts/resume_run.sh: a new job continues an earlier job's variant folder and command."""

import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "resume_run.sh"
# Stands in for a training script: appends to the checkpoint it finds, or starts one
TRAIN = 'mkdir -p "$RUN_DIR/$1" && echo "$JOB SPE=$SPE" >> "$RUN_DIR/$1/last.pt"'


def make_job(runs: Path, name: str, command: str, checkpoint: str | None = None) -> Path:
    run = runs / name
    run.mkdir(parents=True)
    (run / "info.txt").write_text(f"job:      {name}\ncommand:  {command}\nsource:   x\n")
    if checkpoint is not None:
        (run / "v2").mkdir()
        (run / "v2" / "last.pt").write_text(checkpoint)
        (run / "v2" / "train_1.log").write_text("log\n")
    return run


def resume(runs: Path, new: str, old: str, cwd: Path) -> str:
    """Submit-and-run job `new`, as rcp/train.sh would; returns the output."""
    run = runs / new
    if not run.exists():
        make_job(runs, new, f"bash scripts/resume_run.sh {old}")
    env = dict(os.environ, RUN_DIR=str(run), JOB=new)
    done = subprocess.run(["bash", str(SCRIPT), old], cwd=cwd, env=env,
                          capture_output=True, text=True, check=True)
    return done.stdout


def original(tmp_path: Path, checkpoint: str | None) -> Path:
    (tmp_path / "train.sh").write_text(TRAIN + "\n")
    runs = tmp_path / "runs"
    make_job(runs, "a-001", "bash -c SPE=$((2 * 5)) bash train.sh v2", checkpoint)
    return runs


def test_copies_checkpoint_once_and_runs_the_same_command(tmp_path):
    runs = original(tmp_path, "epoch 3\n")
    resume(runs, "a-002", "a-001", tmp_path)
    new = runs / "a-002" / "v2"
    assert new.joinpath("last.pt").read_text() == "epoch 3\na-002 SPE=10\n"
    assert new.joinpath("train_1.log").exists()
    # The earlier run folder is untouched
    assert (runs / "a-001" / "v2" / "last.pt").read_text() == "epoch 3\n"
    # A restart of the new job keeps its own, newer checkpoint
    resume(runs, "a-002", "a-001", tmp_path)
    assert new.joinpath("last.pt").read_text() == "epoch 3\na-002 SPE=10\na-002 SPE=10\n"
    assert not (runs / "a-002" / "v2.part").exists()


def test_partial_copy_is_redone(tmp_path):
    runs = original(tmp_path, "epoch 3\n")
    make_job(runs, "a-002", "bash scripts/resume_run.sh a-001")
    (runs / "a-002" / "v2.part").mkdir()
    (runs / "a-002" / "v2.part" / "last.pt").write_text("trunc")
    resume(runs, "a-002", "a-001", tmp_path)
    assert (runs / "a-002" / "v2" / "last.pt").read_text() == "epoch 3\na-002 SPE=10\n"


def test_resume_of_a_resumed_job(tmp_path):
    runs = original(tmp_path, "epoch 3\n")
    resume(runs, "a-002", "a-001", tmp_path)
    resume(runs, "a-003", "a-002", tmp_path)
    assert (runs / "a-003" / "v2" / "last.pt").read_text() == (
        "epoch 3\na-002 SPE=10\na-003 SPE=10\n"
    )


def test_resumed_job_that_never_ran_falls_back_to_the_original(tmp_path):
    runs = original(tmp_path, "epoch 3\n")
    make_job(runs, "a-002", "bash scripts/resume_run.sh a-001")  # stopped while pending
    resume(runs, "a-003", "a-002", tmp_path)
    assert (runs / "a-003" / "v2" / "last.pt").read_text() == "epoch 3\na-003 SPE=10\n"


def test_no_checkpoint_starts_from_scratch(tmp_path):
    runs = original(tmp_path, None)
    out = resume(runs, "a-002", "a-001", tmp_path)
    assert "starting from scratch" in out
    assert (runs / "a-002" / "v2" / "last.pt").read_text() == "a-002 SPE=10\n"
