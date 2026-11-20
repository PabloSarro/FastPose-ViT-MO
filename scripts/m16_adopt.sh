#!/usr/bin/env bash
# M16: pool the per-object models of one dataset and seed (scripts/m16_single.sh jobs) into the
# two numbers of the adoption rule: the validation score on noisy boxes (pooled over the objects,
# scripts/m16_merge_valn.py) and the official test AR with the BOP default detections (the
# per-object CSVs merged by scripts/m10_merge.py, then scripts/bop_eval.sh).
#
# Usage (from the project root, in the dev job; the official eval takes ~10-20 min):
#   bash scripts/m16_adopt.sh ycbv|tless SEED(0|1)
# Output: $NAS_HOME/results/m16_<dataset>_s<seed>/ (merged CSV, eval/, summary.txt).
set -euo pipefail
(( $# == 2 )) || { echo "usage: $0 ycbv|tless SEED"; exit 5; }
ds=$1 seed=$2
RUNS=$NAS_HOME/outputs/FastPose-ViT/runs
case $ds in
    ycbv)  n=21; no_val=" 1 5 9 10 15 17 18 19 21 " ;;  # absent from the held-out real videos
    tless) n=30; no_val=" " ;;
    *) echo "unknown dataset $ds"; exit 5 ;;
esac
suffix=""; tag="obj"
[[ $seed == 1 ]] && suffix="s1"
out=$NAS_HOME/results/m16_${ds}_s$seed
mkdir -p "$out"
val_dirs=() csvs=()
for o in $(seq 1 "$n"); do
    run=$RUNS/$ds-m16-obj$o$suffix-001
    sub=obj$o; [[ $seed == 1 ]] && sub=obj${o}_s1
    [[ -e $run/$sub/eval_det/DONE ]] || { echo "missing: $run/$sub/eval_det/DONE" >&2; exit 1; }
    csvs+=("$run/$sub/eval_det/fastposevitmo_$ds-test.csv")
    if [[ $no_val != *" $o "* ]]; then
        [[ -e $run/$sub/eval_valrealn/DONE ]] || { echo "missing valrealn of obj $o" >&2; exit 1; }
        val_dirs+=("$run/$sub/eval_valrealn")
    fi
done
{
    echo "== $ds seed $seed: validation, noisy boxes, pooled over ${#val_dirs[@]} objects"
    .venv/bin/python scripts/m16_merge_valn.py --split val_real "${val_dirs[@]}"
} | tee "$out/summary.txt"
PYTHONPATH=. .venv/bin/python scripts/m10_merge.py "$out/fastposevitmo_$ds-test.csv" "${csvs[@]}" | tee -a "$out/summary.txt"
bash scripts/bop_eval.sh "$out" "fastposevitmo_$ds-test.csv" > "$out/bop_eval_test.log" 2>&1
grep -a "bop19_average_recall:" "$out/bop_eval_test.log" | tee -a "$out/summary.txt"
