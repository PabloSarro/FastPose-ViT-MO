#!/usr/bin/env bash
# M11 C1: RGB-D variant (scripts/depth_correction.py, GPU point-splat method) of every RGB result,
# then the official BOP19 eval. One folder per input: $OUT/<name>/fastposevitmodepth_<ds>-test.csv.
# Usage (repo root, dev or a job with a GPU): bash scripts/m11_rgbd.sh NAME DATASET RGB_CSV [...]
#   (triples NAME DATASET RGB_CSV, repeated). OUT defaults to $NAS_HOME/results/m11/rgbd_splat.
set -euo pipefail
export PYTHONPATH=.
OUT=${OUT:-$NAS_HOME/results/m11/rgbd_splat}
(( $# >= 3 && $# % 3 == 0 )) || { sed -n '4,5p' "$0"; exit 5; }
while (( $# )); do
    name=$1 ds=$2 csv=$3; shift 3
    out=$OUT/$name
    [[ -e $out/DONE ]] && { echo "== $name: done"; continue; }
    mkdir -p "$out"
    echo "== $name: correct ($(date '+%T'))"
    .venv/bin/python scripts/depth_correction.py --dataset "$ds" --csv "$csv" \
        --out "$out/fastposevitmodepth_${ds}-test.csv" > "$out/correction.log" 2>&1
    echo "== $name: official eval ($(date '+%T'))"
    bash scripts/bop_eval.sh "$out" "fastposevitmodepth_${ds}-test.csv" > "$out/bop_eval.log" 2>&1
    touch "$out/DONE"
done
echo "== all done ($(date '+%T'))"
