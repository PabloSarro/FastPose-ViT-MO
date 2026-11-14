#!/usr/bin/env bash
# Paper ablations (docs/PAPER_TODO.md §3): each variant = the 2x ablation baseline symtok2x
# (scripts/m4_multi.sh: token, domain_gap, symmetric loss, bbox_visib, 384 px, 150 epochs,
# 84,416 samples per epoch; LM-O test AR 0.655 / 0.655 for seeds 0 / 1) with ONE change, then
# LM-O test with the BOP default detections and the official BOP19 eval (last.pt), as
# scripts/paper_ablation_eval.sh. DONE files per step, so a restarted job continues.
#
# Usage (from the project root, inside a job), one variant per job; a trailing "2" = seed 1:
#   CPUS=16 bash rcp/train.sh lmo-m10-absrot-001 bash scripts/m10_ablations.sh absrot
# Variants:
#   absrot   absolute (egocentric) rotation target instead of the apparent one (WACV contribution)
#   scratch  ViT-B/16-384 from random initialisation instead of SWAG (WACV Table 5)
#   bbox0    no bounding-box jitter (WACV Table 8: 0 %)
#   bbox20   bounding-box jitter up to 20 % per side (default 10 %)
#   legacy   the WACV alpha-based depth target instead of `site`, with 8 objects
# Augmentation study (2026-11-08, user + supervisor; baseline pixel aug = domain_gap):
#   augnone   no pixel augmentation
#   augcolor  brightness / contrast, gamma, hue-saturation only     (src/bop_dataset.PIXEL_AUGS)
#   augblur   Gaussian, motion, defocus blur only
#   augnoise  Gaussian and ISO sensor noise only
#   augsimple color + blur + noise
#   augstrong domain_gap with every op at p = 0.8 instead of 0.5
#   bbox5 bbox15 bbox30   bounding-box jitter up to 5 / 15 / 30 % per side (with bbox0, bbox20)
# M11 "improve first" (2026-11-14, user; TASKS.md M11 phase A):
#   ctx      image around the box instead of black bands (square crop, no margin)
#   ctx125   context margin: square of 1.25 x max(w, h), filled with the image
#   ctx15    same, 1.5 x
#   occl     synthetic occlusion: 1-4 random rectangles erased per training crop (p = 0.5)
#   tlw      translation loss weighted x2 (total = 2 * translation + rotation)
set -euo pipefail
: "${RUN_DIR:?run inside a train job (RUN_DIR not set)}"
export PYTHONPATH=.
EPOCHS=${EPOCHS:-150}
SPE=${SPE:-$((2 * 42208))}
DET=$(ls "$NAS_HOME"/datasets/bop_detections/default_task1/bop23_default_detections_for_task1/gdrnppdet-pbr/gdrnppdet-pbr_lmo-test_*.json)

(( $# == 1 )) || { echo "usage: $0 VARIANT[2]"; exit 5; }
v=$1 base=${1%2} seed=0
[[ $v != "$base" ]] && seed=1
VIT=vit_b_16_384 DEPTH=site extra=() eval_extra=()
case $base in
    absrot)  extra=(--no_rotation_compensation); eval_extra=(--no_rotation_compensation) ;;
    scratch) VIT=vit_b_16_384_no_weights ;;
    bbox0)   extra=(--bbox_crop_percent 0) ;;
    bbox20)  extra=(--bbox_crop_percent 20) ;;
    legacy)  DEPTH=legacy ;;
    augnone)   extra=(--no_pixel_augmentation) ;;
    augcolor)  extra=(--bop_pixel_aug color) ;;
    augblur)   extra=(--bop_pixel_aug blur) ;;
    augnoise)  extra=(--bop_pixel_aug noise) ;;
    augsimple) extra=(--bop_pixel_aug simple) ;;
    augstrong) extra=(--bop_pixel_aug domain_gap_strong) ;;
    bbox5)   extra=(--bbox_crop_percent 5) ;;
    bbox15)  extra=(--bbox_crop_percent 15) ;;
    bbox30)  extra=(--bbox_crop_percent 30) ;;
    ctx)     extra=(--crop_context); eval_extra=(--crop_context) ;;
    ctx125)  extra=(--crop_context --crop_scale 1.25); eval_extra=(--crop_context --crop_scale 1.25) ;;
    ctx15)   extra=(--crop_context --crop_scale 1.5); eval_extra=(--crop_context --crop_scale 1.5) ;;
    occl)    extra=(--bop_occlusion_aug) ;;
    tlw)     extra=(--translation_loss_weight 2) ;;
    *) echo "unknown variant $v"; exit 5 ;;
esac
out=$RUN_DIR/$v
mkdir -p "$out"
if [[ ! -e $out/TRAINED ]]; then
    echo "== $v: train, seed $seed ($(date '+%F %T'))"
    .venv/bin/python src/train.py --dataset LMO --bop_val_split val \
        --max_val_samples 2000 --rotation_format matrix --rotation_loss 6d_relative_frobenius \
        --max_lr 1e-4 --min_lr 1e-6 --epochs "$EPOCHS" --num_hidden_layers 0 --hidden_layer_dim 0 \
        --nb_class_tokens 1 --batch_size 32 --vit_model "$VIT" --optimizer muon \
        --scheduler cosineannealingwarmrestarts --num_workers 14 --merge_outputs --no_mlp \
        --bbox_crop_percent 10 --depth_mode "$DEPTH" --translation_loss relative_translation \
        --log_dir "$out/" --model_save_path "$out/model.pth" --resume auto \
        --object_conditioning token --symmetric_rotation_loss --bop_pixel_aug domain_gap \
        --bop_epoch_samples "$SPE" --bop_sampler_seed "$seed" "${extra[@]}"
    touch "$out/TRAINED"
fi
if [[ ! -e $out/eval_det/DONE ]]; then
    echo "== $v: eval LM-O test, detections ($(date '+%F %T'))"
    .venv/bin/python src/bop_evaluate.py --dataset lmo --model_weights "$out/last.pt" \
        --vit_model "$VIT" --splits test --depth_mode "$DEPTH" --object_conditioning token \
        "${eval_extra[@]}" --detections "$DET" --out_dir "$out/eval_det"
    bash scripts/bop_eval.sh "$out/eval_det" fastposevitmo_lmo-test.csv \
        > "$out/eval_det/bop_eval_test.log" 2>&1
    touch "$out/eval_det/DONE"
fi
echo "== $v: all done ($(date '+%F %T'))"
