#!/usr/bin/env bash
# M10 T-LESS (user, 2026-11-08): the final LM-O / YCB-V recipe on all 30 T-LESS objects, as
# scripts/m8_ycbv.sh: token, domain_gap, symmetric loss (with translation offsets), bbox_visib,
# 384 px, 150 epochs. Budget 4x = LM-O 4x per object: 30 x 21,104 = 633,120 samples per epoch
# (~83 h on 1 GPU, from YCB-V's ~58 h for 443,184).
# Evaluation of last.pt: val_pbr (GT bbox_visib), test with GT boxes and test with the BOP default
# detections (GDRNPPDet-PBR), each test run followed by the official BOP19 eval. DONE files per
# step, so a restarted job continues (training resumes from last.pt).
#
# Usage (from the project root, inside a job), one variant per job:
#   CPUS=16 bash rcp/train.sh tless-m10-tok4x-001 bash scripts/m10_tless.sh tok4x
# Variants: tok4x tok4x2 (sampler seeds 0 / 1).
set -euo pipefail
: "${RUN_DIR:?run inside a train job (RUN_DIR not set)}"
export PYTHONPATH=.
EPOCHS=${EPOCHS:-150}
VIT=${VIT:-vit_b_16_384}
SPE4=${SPE4:-633120}  # 30 objects x 21,104
DET=$(ls "$NAS_HOME"/datasets/bop_detections/default_task1/bop23_default_detections_for_task1/gdrnppdet-pbr/gdrnppdet-pbr_tless-test_*.json)
CSV=fastposevitmo_tless-test.csv

declare -A SEED=([tok4x]=0 [tok4x2]=1)
(( $# )) || { echo "usage: $0 VARIANT... (${!SEED[*]})"; exit 5; }

for v in "$@"; do
    [[ -n ${SEED[$v]:-} ]] || { echo "unknown variant $v"; exit 5; }
    out=$RUN_DIR/$v
    mkdir -p "$out"
    if [[ ! -e $out/TRAINED ]]; then
        echo "== $v: train ($(date '+%F %T'))"
        .venv/bin/python src/train.py --dataset TLESS --bop_val_split val \
            --max_val_samples 2000 --rotation_format matrix --rotation_loss 6d_relative_frobenius \
            --max_lr 1e-4 --min_lr 1e-6 --epochs "$EPOCHS" --num_hidden_layers 0 --hidden_layer_dim 0 \
            --nb_class_tokens 1 --batch_size 32 --vit_model "$VIT" --optimizer muon \
            --scheduler cosineannealingwarmrestarts --num_workers 14 --merge_outputs --no_mlp \
            --bbox_crop_percent 10 --depth_mode site --translation_loss relative_translation \
            --log_dir "$out/" --model_save_path "$out/model.pth" --resume auto \
            --object_conditioning token --symmetric_rotation_loss --bop_pixel_aug domain_gap \
            --bop_epoch_samples "$SPE4" --bop_sampler_seed "${SEED[$v]}"
        touch "$out/TRAINED"
    fi
    eval_args=(--dataset tless --model_weights "$out/last.pt" --vit_model "$VIT" --depth_mode site
               --object_conditioning token)
    if [[ ! -e $out/eval_valpbr/DONE ]]; then
        echo "== $v: eval val_pbr ($(date '+%F %T'))"
        .venv/bin/python src/bop_evaluate.py "${eval_args[@]}" --splits val_pbr --out_dir "$out/eval_valpbr"
        touch "$out/eval_valpbr/DONE"
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
done
echo "== all done ($(date '+%F %T'))"
