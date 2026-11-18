#!/usr/bin/env bash
# M16: noisy-box validation (eval_valn, as scripts/m12_valeval.sh) of single-object LM-O models,
# so that one-model-per-object can be judged by the adoption rule (validation AND test). Each
# model is evaluated on its own object only; the noisy boxes are the same per instance as in the
# shared models' eval_valn (src/bop_evaluate.noisy_boxes_for_rows). Then pool the objects with
# scripts/m16_merge_valn.py.
#
# Usage (repo root, dev or a job with a GPU), one or more object folders (holding last.pt):
#   bash scripts/m16_single_valn.sh $NAS_HOME/outputs/FastPose-ViT/runs/lmo-m10-obj8-001/obj8 ...
set -euo pipefail
export PYTHONPATH=.
(( $# )) || { sed -n '8,9p' "$0"; exit 5; }
for dir in "$@"; do
    name=$(basename "$dir"); obj=${name#obj}; obj=${obj%_s*}
    out=$dir/eval_valn
    [[ -e $dir/last.pt ]] || { echo "== $name: no last.pt, skipped"; continue; }
    [[ -e $out/DONE ]] && { cat "$out/val_score.txt"; continue; }
    echo "== $name (object $obj): validation, noisy boxes ($(date '+%T'))"
    .venv/bin/python src/bop_evaluate.py --dataset lmo --obj_ids "$obj" \
        --model_weights "$dir/last.pt" --vit_model vit_b_16_384 --splits val_real \
        --depth_mode site --object_conditioning token --box_noise 0.1 --out_dir "$out" \
        > "$dir/eval_valn.log" 2>&1
    .venv/bin/python scripts/val_score.py "$out" > "$out/val_score.txt"
    touch "$out/DONE"
    cat "$out/val_score.txt"
done
