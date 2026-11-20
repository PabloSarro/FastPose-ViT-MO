#!/usr/bin/env bash
# M19 R1 (upper-bound check): CosyPose's PBR-only refiner on our poses, 1-4 iterations, official
# BOP19 eval of every iteration count. Poses = a BOP results CSV of ours (test split, detections).
#
# Usage (dev job, GPU; the venv and models: see scripts/cosypose_refine.py):
#   bash scripts/m19_r1.sh lmo|ycbv|tless RESULTS_CSV
# Output: $NAS_HOME/results/m19_r1_<dataset>/ (refined CSVs, eval/, summary.txt with the 4 AR rows
# and the unrefined AR read from the CSV's own bop_eval log if it sits next to it).
set -euo pipefail
(( $# == 2 )) || { echo "usage: $0 lmo|ycbv|tless RESULTS_CSV"; exit 5; }
ds=$1 csv=$2
out=$NAS_HOME/results/m19_r1_$ds
mkdir -p "$out"
export HAPPYPOSE_DATA_DIR=$NAS_HOME/happypose_data PYTHONPATH=$NAS_HOME/bop_toolkit USER=${USER:-rcp}
method=$(basename "$csv" | cut -d_ -f1)
if [[ ! -e $out/refined.DONE ]]; then
    $NAS_HOME/venvs/happypose/bin/python -u scripts/cosypose_refine.py --dataset "$ds" --csv "$csv" \
        --out_dir "$out" --iterations 4
    touch "$out/refined.DONE"
fi
{
    echo "== M19 R1 $ds: poses from $csv"
    base=$(dirname "$csv")/bop_eval_test.log
    [[ -e $base ]] && grep -a "bop19_average_recall:" "$base" | sed 's/^/unrefined /' || true
} | tee "$out/summary.txt"
for n in 1 2 3 4; do
    f=${method}r${n}_$ds-test.csv
    [[ -e $out/eval/${f%.csv}/scores_bop19.json ]] || bash scripts/bop_eval.sh "$out" "$f" > "$out/bop_eval_r$n.log" 2>&1
    grep -a "bop19_average_recall:\|bop19_average_time_per_image" "$out/bop_eval_r$n.log" | sed "s/^/iterations=$n /" | tee -a "$out/summary.txt"
done
