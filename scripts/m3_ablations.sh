#!/usr/bin/env bash
# M3 short ablations at 224 on the driller (obj 8): train + evaluate several variants one after
# the other, inside one train job. Each variant gets $RUN_DIR/<name>/ (train log, last.pt,
# eval/). A finished variant has a DONE file and is skipped, so a restarted job continues.
#
# Usage (from the project root, inside a job): bash scripts/m3_ablations.sh [VARIANT...]
# Default variants, in priority order: base dec legacy base2 domgap (base2 = same flags as base,
# an independent run (training is not seeded): the run-to-run noise to judge the differences).
set -euo pipefail
: "${RUN_DIR:?run inside a train job (RUN_DIR not set)}"
export PYTHONPATH=.
EPOCHS=${EPOCHS:-30}  # 30 = end of the 2nd warm-restart cycle (10 + 20); ~1.1 h per variant

# Variant -> extra train.py flags. All: ViT-B/16 at 224, driller, standard aug unless stated.
declare -A FLAGS=(
    [base]="--depth_mode site --translation_loss relative_translation"
    [base2]="--depth_mode site --translation_loss relative_translation"
    [legacy]="--depth_mode legacy --translation_loss relative_translation"
    [dec]="--depth_mode site --translation_loss decoupled --decoupled_lambda 1.0"
    [domgap]="--depth_mode site --translation_loss relative_translation --bop_pixel_aug domain_gap"
)
variants=("$@")
(( ${#variants[@]} )) || variants=(base dec legacy base2 domgap)

for v in "${variants[@]}"; do
    out=$RUN_DIR/$v
    [[ -e $out/DONE ]] && { echo "== $v: already done"; continue; }
    mkdir -p "$out"
    echo "== $v: train ($(date '+%F %T'))"
    # shellcheck disable=SC2086
    .venv/bin/python src/train.py --dataset LMO --bop_obj_ids 8 --bop_val_split val \
        --max_val_samples 2000 --rotation_format matrix --rotation_loss 6d_relative_frobenius \
        --max_lr 1e-4 --min_lr 1e-6 --epochs "$EPOCHS" --num_hidden_layers 0 --hidden_layer_dim 0 \
        --nb_class_tokens 1 --batch_size 32 --vit_model vit_b_16 --optimizer muon \
        --scheduler cosineannealingwarmrestarts --num_workers 14 --merge_outputs --no_mlp \
        --bbox_crop_percent 10 --log_dir "$out/" --model_save_path "$out/model.pth" \
        --resume auto ${FLAGS[$v]}
    echo "== $v: eval ($(date '+%F %T'))"
    # last.pt (final epoch = end of a cycle) rather than the best-val model.pth
    .venv/bin/python src/bop_evaluate.py --model_weights "$out/last.pt" --obj_id 8 \
        --vit_model vit_b_16 --depth_mode "$(grep -o "depth_mode [a-z]*" <<<"${FLAGS[$v]}" | cut -d" " -f2)" \
        --out_dir "$out/eval"
    touch "$out/DONE"
done
echo "== all done ($(date '+%F %T'))"
