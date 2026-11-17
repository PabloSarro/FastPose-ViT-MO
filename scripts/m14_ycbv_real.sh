#!/usr/bin/env bash
# M11 B3 (user, 2026-11-17): YCB-V trained on PBR + the real train_real images (--bop_train_split
# train_mix: 626,520 PBR rows + 481,668 real rows of 76 videos; four uw videos held out as real
# val, --bop_val_split val_real), otherwise the YCB-V recipe of scripts/m8_ycbv.sh (token, 150
# epochs, domain_gap, symmetric loss). Then validation on the held-out real videos (GT boxes and
# boxes perturbed by up to 10 %), the YCB-V test with GT boxes and with the BOP default
# detections, and the official BOP19 eval. DONE files per step, so a restarted job continues.
#
# Usage (repo root, inside a job), one variant per job; a trailing "2" = seed 1:
#   CPUS=16 bash rcp/train.sh ycbv-m14-real2x-001 bash scripts/m14_ycbv_real.sh real2x
# Variants: real2x / real2x2 (2x budget, compare tok2x / tok2x2), real4x / real4x2 (4x, compare
# the final tok4x / tok4x2).
set -euo pipefail
: "${RUN_DIR:?run inside a train job (RUN_DIR not set)}"
export PYTHONPATH=.
EPOCHS=${EPOCHS:-150}
SPE4=443184  # 21 objects x 21,104 (LM-O 4x per object)
DET=$(ls "$NAS_HOME"/datasets/bop_detections/default_task1/bop23_default_detections_for_task1/gdrnppdet-pbr/gdrnppdet-pbr_ycbv-test_*.json)
CSV=fastposevitmo_ycbv-test.csv
declare -A SEED=([real2x]=0 [real2x2]=1 [real4x]=0 [real4x2]=1)
declare -A SPE=([real2x]=$((SPE4 / 2)) [real2x2]=$((SPE4 / 2)) [real4x]=$SPE4 [real4x2]=$SPE4)
(( $# == 1 )) && [[ -n ${SEED[$1]:-} ]] || { echo "usage: $0 (${!SEED[*]})"; exit 5; }
v=$1
out=$RUN_DIR/$v
mkdir -p "$out"
if [[ ! -e $out/TRAINED ]]; then
    echo "== $v: train ($(date '+%F %T'))"
    .venv/bin/python src/train.py --dataset YCBV --bop_train_split train_mix --bop_val_split val_real \
        --max_val_samples 2000 --rotation_format matrix --rotation_loss 6d_relative_frobenius \
        --max_lr 1e-4 --min_lr 1e-6 --epochs "$EPOCHS" --num_hidden_layers 0 --hidden_layer_dim 0 \
        --nb_class_tokens 1 --batch_size 32 --vit_model vit_b_16_384 --optimizer muon \
        --scheduler cosineannealingwarmrestarts --num_workers 14 --merge_outputs --no_mlp \
        --bbox_crop_percent 10 --depth_mode site --translation_loss relative_translation \
        --log_dir "$out/" --model_save_path "$out/model.pth" --resume auto \
        --object_conditioning token --symmetric_rotation_loss --bop_pixel_aug domain_gap \
        --bop_epoch_samples "${SPE[$v]}" --bop_sampler_seed "${SEED[$v]}"
    touch "$out/TRAINED"
fi
eval_args=(--dataset ycbv --model_weights "$out/last.pt" --vit_model vit_b_16_384 --depth_mode site
           --object_conditioning token)
if [[ ! -e $out/eval_valreal/DONE ]]; then
    echo "== $v: eval held-out real videos, GT boxes ($(date '+%F %T'))"
    .venv/bin/python src/bop_evaluate.py "${eval_args[@]}" --splits val_real --out_dir "$out/eval_valreal"
    .venv/bin/python scripts/val_score.py "$out/eval_valreal" > "$out/eval_valreal/val_score.txt"
    touch "$out/eval_valreal/DONE"
fi
if [[ ! -e $out/eval_valrealn/DONE ]]; then
    echo "== $v: eval held-out real videos, noisy boxes ($(date '+%F %T'))"
    .venv/bin/python src/bop_evaluate.py "${eval_args[@]}" --splits val_real --box_noise 0.1 \
        --out_dir "$out/eval_valrealn"
    .venv/bin/python scripts/val_score.py "$out/eval_valrealn" > "$out/eval_valrealn/val_score.txt"
    touch "$out/eval_valrealn/DONE"
fi
if [[ ! -e $out/eval_test/DONE ]]; then
    echo "== $v: eval test, GT boxes ($(date '+%F %T'))"
    .venv/bin/python src/bop_evaluate.py "${eval_args[@]}" --splits test --out_dir "$out/eval_test"
    bash scripts/bop_eval.sh "$out/eval_test" "$CSV" > "$out/eval_test/bop_eval_test.log" 2>&1
    touch "$out/eval_test/DONE"
fi
if [[ ! -e $out/eval_det/DONE ]]; then
    echo "== $v: eval test, detections ($(date '+%F %T'))"
    .venv/bin/python src/bop_evaluate.py "${eval_args[@]}" --splits test --detections "$DET" \
        --out_dir "$out/eval_det"
    bash scripts/bop_eval.sh "$out/eval_det" "$CSV" > "$out/eval_det/bop_eval_test.log" 2>&1
    touch "$out/eval_det/DONE"
fi
echo "== $v: all done ($(date '+%F %T'))"
