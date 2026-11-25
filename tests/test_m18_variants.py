"""scripts/m18_dense.sh: every variant name gives the intended training flags and seed.

The script runs in a temporary folder with stand-ins for `.venv/bin/python` (records the arguments
of src/train.py) and `nvidia-smi`, and with no pose modes to evaluate; nothing is trained.
"""

import os
import stat
import subprocess
from pathlib import Path

import pytest

from src.train_args import build_parser

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "m18_dense.sh"

# densex5 and, per variant, only the arguments that differ from it
BASE = {
    "xyz_loss_weight": 5.0,
    "mask_loss_weight": 1.0,
    "rotation_loss_weight": 1.0,
    "translation_loss_weight": 2.0,
    "bbox_crop_percent": 10.0,
    "bop_occlusion_aug": True,
    "bop_occlusion_p": 0.5,
    "bop_occlusion_max_holes": 4,
    "bop_occlusion_max_size": 0.3,
    "bop_cutout_p": 0.0,
    "bop_occlusion_targets": "keep",
    "bop_train_split": "train",
    "bop_real_fraction": None,
    "vit_model": "vit_b_16_384",
    "dense_symmetric_xyz": False,
}
CHANGES = {
    "dense": {"xyz_loss_weight": 1.0},
    "densex5": {},
    "densexl": {"xyz_loss_weight": 0.2},
    "densesym": {"xyz_loss_weight": 1.0, "dense_symmetric_xyz": True},
    "densesymx5": {"dense_symmetric_xyz": True},
    "densesymx5l": {"dense_symmetric_xyz": True, "vit_model": "vit_l_16_512"},
    "densex5l": {"vit_model": "vit_l_16_512"},
    "densex10": {"xyz_loss_weight": 10.0},
    "densex5m05": {"mask_loss_weight": 0.5},
    "densex5m20": {"mask_loss_weight": 2.0},
    "densex5m50": {"mask_loss_weight": 5.0},
    "densex5r05": {"rotation_loss_weight": 0.5},
    "densex5r20": {"rotation_loss_weight": 2.0},
    "densex5j15": {"bbox_crop_percent": 15.0},
    "densex5op8": {"bop_occlusion_p": 0.8},
    "densex5obig": {"bop_occlusion_max_holes": 6, "bop_occlusion_max_size": 0.4},
    "densex5ocut": {"bop_cutout_p": 0.5},
    "densex5real": {"bop_train_split": "train_mix", "bop_real_fraction": 0.25},
    "densex5ochide": {"bop_occlusion_targets": "hide"},
    "densex5ocmask": {"bop_occlusion_targets": "hide_mask"},
}


def executable(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def train_args(tmp_path: Path, variant: str):
    executable(
        tmp_path / ".venv" / "bin" / "python",
        '#!/usr/bin/env bash\n[[ $1 == src/train.py ]] && printf "%s\\n" "${@:2}" > args.txt\n'
        "exit 0\n",
    )
    executable(tmp_path / "bin" / "nvidia-smi", "#!/usr/bin/env bash\necho GPU 0\n")
    det = tmp_path / "datasets/bop_detections/default_task1/bop23_default_detections_for_task1"
    (det / "gdrnppdet-pbr").mkdir(parents=True, exist_ok=True)
    (det / "gdrnppdet-pbr" / "gdrnppdet-pbr_lmo-test_x.json").write_text("[]")
    env = {k: v for k, v in os.environ.items() if k not in ("SPE", "EPOCHS", "NUM_WORKERS")}
    env.update(
        RUN_DIR=str(tmp_path / "run"),
        NAS_HOME=str(tmp_path),
        MODES=" ",  # no evaluation (an empty string would mean the default modes)
        PATH=f"{tmp_path / 'bin'}:{os.environ['PATH']}",
    )
    done = subprocess.run(
        ["bash", str(SCRIPT), variant], cwd=tmp_path, env=env, capture_output=True, text=True
    )
    if done.returncode:
        return done, None
    return done, build_parser().parse_args((tmp_path / "args.txt").read_text().split("\n")[:-1])


@pytest.mark.parametrize("variant", sorted(CHANGES))
@pytest.mark.parametrize("suffix, seed", [("", 0), ("2", 1)])
def test_variant_flags_and_seed(tmp_path, variant, suffix, seed):
    done, args = train_args(tmp_path, variant + suffix)
    assert done.returncode == 0, done.stdout + done.stderr
    expected = {**BASE, **CHANGES[variant]}
    assert {k: getattr(args, k) for k in expected} == expected
    assert args.bop_sampler_seed == seed
    assert args.dense_heads and args.bop_epoch_samples == 2 * 42208


def test_unknown_variant_is_refused(tmp_path):
    done, _ = train_args(tmp_path, "densex7")
    assert done.returncode == 5 and "unknown variant" in done.stdout
