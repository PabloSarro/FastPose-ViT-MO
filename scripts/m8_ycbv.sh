#!/usr/bin/env bash
# M8: the LM-O final recipe (tok4x: class token, domain_gap, symmetric loss with offsets) on all 21
# YCB-V objects, then the YCB-V test evaluation. Each variant gets $RUN_DIR/<name>/ (train log,
# last.pt, eval_*). A finished step has a DONE file and is skipped, so a restarted job continues.
#
# Usage (from the project root, inside a job), one variant per job:
#   CPUS=16 bash rcp/train.sh ycbv-m8-tok4x-001 bash scripts/m8_ycbv.sh tok4x
# Variants: tok4x tok4x2 (sampler seeds 0 / 1, for the run-to-run noise).
# Budget: LM-O 4x per object (21,104 samples per object per epoch, 150 epochs), i.e. 21 x 21,104
# = 443,184 samples per epoch, ~2.6x the LM-O epoch (~56 h at 332 samples/s).
# Evaluation of last.pt: val_pbr (GT bbox_visib), test with GT boxes and test with the BOP default
# detections (GDRNPPDet-PBR), each test run followed by the official BOP19 eval.
set -euo pipefail
: "${RUN_DIR:?run inside a train job (RUN_DIR not set)}"
export PYTHONPATH=.
EPOCHS=${EPOCHS:-150}
VIT=${VIT:-vit_b_16_384}
SPE=${SPE:-443184}
DET=$(ls "$NAS_HOME"/datasets/bop_detections/default_task1/bop23_default_detections_for_task1/gdrnppdet-pbr/gdrnppdet-pbr_ycbv-test_*.json)
CSV=fastposevitmo_ycbv-test.csv

declare -A SEED=([tok4x]=0 [tok4x2]=1)
(( $# )) || { echo "usage: $0 VARIANT... (${!SEED[*]})"; exit 5; }

for v in "$@"; do
    [[ -n ${SEED[$v]:-} ]] || { echo "unknown variant $v"; exit 5; }
    out=$RUN_DIR/$v
    mkdir -p "$out"
    if [[ ! -e $out/TRAINED ]]; then
        echo "== $v: train ($(date '+%F %T'))"
        .venv/bin/python src/train.py --dataset YCBV --bop_val_split val \
            --max_val_samples 2000 --rotation_format matrix --rotation_loss 6d_relative_frobenius \
            --max_lr 1e-4 --min_lr 1e-6 --epochs "$EPOCHS" --num_hidden_layers 0 --hidden_layer_dim 0 \
            --nb_class_tokens 1 --batch_size 32 --vit_model "$VIT" --optimizer muon \
            --scheduler cosineannealingwarmrestarts --num_workers 14 --merge_outputs --no_mlp \
            --bbox_crop_percent 10 --depth_mode site --translation_loss relative_translation \
            --log_dir "$out/" --model_save_path "$out/model.pth" --resume auto \
            --object_conditioning token --symmetric_rotation_loss --bop_pixel_aug domain_gap \
            --bop_epoch_samples "$SPE" --bop_sampler_seed "${SEED[$v]}"
        touch "$out/TRAINED"
    fi
    eval_args=(--dataset ycbv --model_weights "$out/last.pt" --vit_model "$VIT" --depth_mode site
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
