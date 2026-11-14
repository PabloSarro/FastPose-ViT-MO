#!/usr/bin/env bash
# Audit W2 (TASKS.md): validation evaluation (LM real val + PBR val, GT boxes, own recalls) of
# finished LM-O pose models, so that configurations are chosen on validation, not on LM-O test.
# Takes variant folders (RUN/<variant>, holding last.pt); the evaluation flags follow the variant
# name as in scripts/m10_ablations.sh (a trailing 2 = second seed). Output: <variant>/eval_val/
# (summary.json, table_<split>.md, val_score.txt; DONE file, so it can be restarted).
#
# Usage (repo root, dev or a job with a GPU):
#   bash scripts/m12_valeval.sh $NAS_HOME/outputs/FastPose-ViT/runs/lmo-m10-absrot-001/absrot ...
set -euo pipefail
export PYTHONPATH=.
(( $# )) || { sed -n '9,10p' "$0"; exit 5; }
for dir in "$@"; do
    v=$(basename "$dir"); base=${v%2}
    VIT=vit_b_16_384 DEPTH=site eval_extra=()
    case $base in
        absrot)  eval_extra=(--no_rotation_compensation) ;;
        scratch) VIT=vit_b_16_384_no_weights ;;
        legacy)  DEPTH=legacy ;;
        ctx)     eval_extra=(--crop_context) ;;
        ctx125)  eval_extra=(--crop_context --crop_scale 1.25) ;;
        ctx15)   eval_extra=(--crop_context --crop_scale 1.5) ;;
    esac
    out=$dir/eval_val
    [[ -e $out/DONE ]] && { echo "== $v: done"; continue; }
    [[ -e $dir/last.pt ]] || { echo "== $v: no last.pt, skipped"; continue; }
    echo "== $v: validation ($(date '+%T'))"
    .venv/bin/python src/bop_evaluate.py --dataset lmo --model_weights "$dir/last.pt" \
        --vit_model "$VIT" --splits val_real val_pbr --depth_mode "$DEPTH" \
        --object_conditioning token "${eval_extra[@]}" --out_dir "$out" > "$dir/eval_val.log" 2>&1
    .venv/bin/python scripts/val_score.py "$out" > "$out/val_score.txt"
    touch "$out/DONE"
    cat "$out/val_score.txt"
done
