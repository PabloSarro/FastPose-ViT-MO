#!/usr/bin/env bash
# M18.7 (2026-11-25): seed ensemble of densex5 + PnP, evaluation only (no training). The poses of
# the two seed models are averaged per target (scripts/pose_ensemble.py), then scored like single
# runs: the noisy-box validation (own score) and the LM-O test with the official BOP19 eval.
# Usage (repo root, in dev):  bash scripts/m18_ensemble.sh RUN_A/densex5 RUN_B/densex52 OUT_DIR
set -euo pipefail
(( $# == 3 )) || { echo "usage: $0 VARIANT_DIR_SEED_A VARIANT_DIR_SEED_B OUT_DIR"; exit 5; }
a=$1 b=$2 out=$3
export PYTHONPATH=.
test_csv=fastposevitmo_lmo-test.csv val_csv=fastposevitmo_lmo-val_real.csv
mkdir -p "$out/eval_valn_pnp" "$out/eval_det_pnp"
.venv/bin/python scripts/pose_ensemble.py --dataset lmo --out "$out/eval_valn_pnp/$val_csv" \
    "$a/eval_valn_pnp/$val_csv" "$b/eval_valn_pnp/$val_csv"
# the two single seeds are scored too: they must reproduce their eval_valn_pnp/val_score.txt
.venv/bin/python scripts/val_score_csv.py --dataset lmo --split val_real \
    "$a/eval_valn_pnp/$val_csv" "$b/eval_valn_pnp/$val_csv" "$out/eval_valn_pnp/$val_csv" \
    | tee "$out/eval_valn_pnp/val_score.txt"
.venv/bin/python scripts/pose_ensemble.py --dataset lmo --out "$out/eval_det_pnp/$test_csv" \
    "$a/eval_det_pnp/$test_csv" "$b/eval_det_pnp/$test_csv"
bash scripts/bop_eval.sh "$out/eval_det_pnp" "$test_csv" > "$out/eval_det_pnp/bop_eval_test.log" 2>&1
grep -E 'bop19_average_recall' "$out/eval_det_pnp/bop_eval_test.log" | tail -3
