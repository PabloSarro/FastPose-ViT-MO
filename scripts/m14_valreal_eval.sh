#!/usr/bin/env bash
# M11 B3: the held-out real YCB-V videos (split val_real) with the PBR-only YCB-V models, as the
# baseline of scripts/m14_ycbv_real.sh. Takes variant folders (RUN/<variant>, holding last.pt);
# output <variant>/eval_valreal{,n}/ (GT boxes, boxes perturbed by up to 10 %), DONE files.
# Usage (repo root, GPU): bash scripts/m14_valreal_eval.sh $NAS_HOME/outputs/FastPose-ViT/runs/ycbv-m8-tok2x-001/tok2x ...
set -euo pipefail
export PYTHONPATH=.
for dir in "$@"; do
    for mode in "" n; do
        out=$dir/eval_valreal$mode
        [[ -e $out/DONE ]] && continue
        extra=(); [[ $mode == n ]] && extra=(--box_noise 0.1)
        echo "== $(basename "$dir"): val_real $mode ($(date '+%T'))"
        .venv/bin/python src/bop_evaluate.py --dataset ycbv --model_weights "$dir/last.pt" \
            --vit_model vit_b_16_384 --depth_mode site --object_conditioning token \
            --splits val_real "${extra[@]}" --out_dir "$out"
        .venv/bin/python scripts/val_score.py "$out" > "$out/val_score.txt"
        touch "$out/DONE"
    done
done
