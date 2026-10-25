#!/usr/bin/env bash
# M4: one shared model with per-object heads on the 8 LM-O objects, object-balanced virtual epochs.
# Trains the given variants one after the other, then evaluates last.pt per object on LM real val
# (LM-O test only at the end of the milestone). Each variant gets $RUN_DIR/<name>/ (train log,
# last.pt, eval/obj_<id>/, eval/table_val_real.md). A finished variant has a DONE file and is
# skipped, so a restarted job continues.
#
# Usage (from the project root, inside a job), one variant per job:
#   bash rcp/train.sh lmo-m4-dg-001 bash scripts/m4_multi.sh dg
# Variants: std std2 dg dg2 dg2x dg4x (std2/dg2 = the same flags with another sampler seed, for
# the run-to-run noise). 1x = B1's compute: 42,208 samples per epoch (5,276 per object), 150
# epochs, ~7.5-8.7 h at 384 px; 2x/4x scale the epoch size (same schedule, more data per epoch).
set -euo pipefail
: "${RUN_DIR:?run inside a train job (RUN_DIR not set)}"
export PYTHONPATH=.
EPOCHS=${EPOCHS:-150}
VIT=${VIT:-vit_b_16_384}
SPE=${SPE:-42208}  # 1x: B1's 42,214 driller rows rounded down to a multiple of 8 objects x batch 32

# Variant -> extra train.py flags. All: $VIT, all 8 LM-O objects, site depth, relative loss.
declare -A FLAGS=(
    [std]="--bop_pixel_aug standard --bop_epoch_samples $SPE --bop_sampler_seed 0"
    [std2]="--bop_pixel_aug standard --bop_epoch_samples $SPE --bop_sampler_seed 1"
    [dg]="--bop_pixel_aug domain_gap --bop_epoch_samples $SPE --bop_sampler_seed 0"
    [dg2]="--bop_pixel_aug domain_gap --bop_epoch_samples $SPE --bop_sampler_seed 1"
    [dg2x]="--bop_pixel_aug domain_gap --bop_epoch_samples $((2 * SPE)) --bop_sampler_seed 0"
    [dg4x]="--bop_pixel_aug domain_gap --bop_epoch_samples $((4 * SPE)) --bop_sampler_seed 0"
)
(( $# )) || { echo "usage: $0 VARIANT... (${!FLAGS[*]})"; exit 5; }

for v in "$@"; do
    [[ -n ${FLAGS[$v]:-} ]] || { echo "unknown variant $v"; exit 5; }
    out=$RUN_DIR/$v
    [[ -e $out/DONE ]] && { echo "== $v: already done"; continue; }
    mkdir -p "$out"
    echo "== $v: train ($(date '+%F %T'))"
    # shellcheck disable=SC2086
    .venv/bin/python src/train.py --dataset LMO --bop_val_split val \
        --max_val_samples 2000 --rotation_format matrix --rotation_loss 6d_relative_frobenius \
        --max_lr 1e-4 --min_lr 1e-6 --epochs "$EPOCHS" --num_hidden_layers 0 --hidden_layer_dim 0 \
        --nb_class_tokens 1 --batch_size 32 --vit_model "$VIT" --optimizer muon \
        --scheduler cosineannealingwarmrestarts --num_workers 14 --merge_outputs --no_mlp \
        --bbox_crop_percent 10 --depth_mode site --translation_loss relative_translation \
        --log_dir "$out/" --model_save_path "$out/model.pth" --resume auto ${FLAGS[$v]}
    echo "== $v: eval ($(date '+%F %T'))"
    # last.pt (final epoch = end of a cycle) rather than the best-val model.pth; all 8 objects,
    # per-object results in eval/obj_<id>/ and the table in eval/table_val_real.md
    .venv/bin/python src/bop_evaluate.py --model_weights "$out/last.pt" --splits val_real \
        --vit_model "$VIT" --depth_mode site --out_dir "$out/eval"
    touch "$out/DONE"
done
echo "== all done ($(date '+%F %T'))"
