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
# Class-token ablation (domain_gap, 2x; a trailing "2" = sampler seed 1): dg2x2 (per-object heads,
# option a, repeat of dg2x), tok2x tok2x2 (per-object class token, option b), both2x both2x2 (a + b).
# Final M4 model (chosen 2026-10-26): tok4x tok4x2 (class token, domain_gap, 4x, 2 seeds).
# M5 exit: symtok2x symtok2x2 = tok2x/tok2x2 + --symmetric_rotation_loss (model.pth is then
# selected on (AR_MSSD + AR_MSPD) / 2; the evaluation uses last.pt as for every variant).
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
    [dg2x2]="--bop_pixel_aug domain_gap --bop_epoch_samples $((2 * SPE)) --bop_sampler_seed 1"
    [tok2x]="--bop_pixel_aug domain_gap --bop_epoch_samples $((2 * SPE)) --bop_sampler_seed 0"
    [tok2x2]="--bop_pixel_aug domain_gap --bop_epoch_samples $((2 * SPE)) --bop_sampler_seed 1"
    [both2x]="--bop_pixel_aug domain_gap --bop_epoch_samples $((2 * SPE)) --bop_sampler_seed 0"
    [both2x2]="--bop_pixel_aug domain_gap --bop_epoch_samples $((2 * SPE)) --bop_sampler_seed 1"
    [tok4x]="--bop_pixel_aug domain_gap --bop_epoch_samples $((4 * SPE)) --bop_sampler_seed 0"
    [tok4x2]="--bop_pixel_aug domain_gap --bop_epoch_samples $((4 * SPE)) --bop_sampler_seed 1"
    [symtok2x]="--bop_pixel_aug domain_gap --bop_epoch_samples $((2 * SPE)) --bop_sampler_seed 0 --symmetric_rotation_loss"
    [symtok2x2]="--bop_pixel_aug domain_gap --bop_epoch_samples $((2 * SPE)) --bop_sampler_seed 1 --symmetric_rotation_loss"
)
# Variant -> --object_conditioning (train and eval); default heads
declare -A COND=(
    [tok2x]=token [tok2x2]=token [both2x]=both [both2x2]=both [tok4x]=token [tok4x2]=token
    [symtok2x]=token [symtok2x2]=token
)
(( $# )) || { echo "usage: $0 VARIANT... (${!FLAGS[*]})"; exit 5; }
# One process per GPU (DDP, M6) when the job has several: GPUS=2 CPUS=32 MEMORY=64Gi bash rcp/train.sh
# ... (--batch_size stays the global batch; --num_workers is per process)
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
if (( NGPU > 1 )); then
    TRAIN=(.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node="$NGPU" src/train.py)
else
    TRAIN=(.venv/bin/python src/train.py)
fi

for v in "$@"; do
    [[ -n ${FLAGS[$v]:-} ]] || { echo "unknown variant $v"; exit 5; }
    cond=${COND[$v]:-heads}
    out=$RUN_DIR/$v
    [[ -e $out/DONE ]] && { echo "== $v: already done"; continue; }
    mkdir -p "$out"
    echo "== $v: train ($(date '+%F %T'))"
    # shellcheck disable=SC2086
    "${TRAIN[@]}" --dataset LMO --bop_val_split val \
        --max_val_samples 2000 --rotation_format matrix --rotation_loss 6d_relative_frobenius \
        --max_lr 1e-4 --min_lr 1e-6 --epochs "$EPOCHS" --num_hidden_layers 0 --hidden_layer_dim 0 \
        --nb_class_tokens 1 --batch_size 32 --vit_model "$VIT" --optimizer muon \
        --scheduler cosineannealingwarmrestarts --num_workers 14 --merge_outputs --no_mlp \
        --bbox_crop_percent 10 --depth_mode site --translation_loss relative_translation \
        --log_dir "$out/" --model_save_path "$out/model.pth" --resume auto \
        --object_conditioning "$cond" ${FLAGS[$v]}
    echo "== $v: eval ($(date '+%F %T'))"
    # last.pt (final epoch = end of a cycle) rather than the best-val model.pth; all 8 objects,
    # per-object results in eval/obj_<id>/ and the table in eval/table_val_real.md
    .venv/bin/python src/bop_evaluate.py --model_weights "$out/last.pt" --splits val_real \
        --vit_model "$VIT" --depth_mode site --object_conditioning "$cond" --out_dir "$out/eval"
    touch "$out/DONE"
done
echo "== all done ($(date '+%F %T'))"
