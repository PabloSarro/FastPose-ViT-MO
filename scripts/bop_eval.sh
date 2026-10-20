#!/usr/bin/env bash
# Run the official BOP19 pose evaluation (AR_VSD, AR_MSSD, AR_MSPD) on result CSVs.
#
# Usage: bash scripts/bop_eval.sh RESULTS_DIR CSV[,CSV...] [extra eval_bop19_pose.py args]
#   e.g. bash scripts/bop_eval.sh $NAS_HOME/results/bop-m0-gtpred gtpred_lmo-test.csv
# Scores go to RESULTS_DIR/eval/<method>_<dataset>-<split>/scores_bop19.json.
#
# Needs: the tree from scripts/make_bop_tree.sh, and bop_toolkit cloned next to this repo.
# The image (fastposevit:1.59) lacks libEGL.so.1 (Ubuntu package libegl1), which the VSD
# renderer needs. Until the image installs it, a copy extracted from the .deb lives in
# $NAS_HOME/.local/lib/egl; vispy finds it through EGL_LIBRARY (it ignores LD_LIBRARY_PATH).
set -euo pipefail
(( $# >= 2 )) || { sed -n '4,5p' "$0" >&2; exit 2; }
results_dir=$1 csvs=$2
shift 2

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
nas_home=${NAS_HOME:-$(dirname "$repo")}
bop_toolkit=${BOP_TOOLKIT:-$nas_home/bop_toolkit}

if ! ldconfig -p | grep -q 'libEGL.so.1 '; then
    egl_dir=$nas_home/.local/lib/egl
    [[ -e $egl_dir/libEGL.so.1 ]] || { echo "No libEGL.so.1 in the image or in $egl_dir" >&2; exit 1; }
    export EGL_LIBRARY=$egl_dir/libEGL.so.1
    export LD_LIBRARY_PATH=$egl_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
fi
export PYOPENGL_PLATFORM=egl
export BOP_PATH=${BOP_PATH:-$nas_home/datasets/bop}
export PYTHONWARNINGS=ignore
# eval_bop19_pose.py starts its sub-steps with a bare `python`: make it the venv's.
export PATH=$repo/.venv/bin:$PATH

python "$bop_toolkit/scripts/eval_bop19_pose.py" --renderer_type vispy \
    --num_workers "${BOP_NUM_WORKERS:-8}" --results_path "$results_dir" \
    --eval_path "$results_dir/eval" --result_filenames "$csvs" "$@"
