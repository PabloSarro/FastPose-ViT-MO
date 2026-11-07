#!/usr/bin/env bash
# M10: one model per object (as GDRNPP's "MModel" entries) vs the shared LM-O model tok4x. Trains a
# single-object model with tok4x's recipe and per-object budget, then evaluates last.pt on LM-O test
# for that object only: with GT boxes (eval_test/) and with the BOP default detections (eval_det/;
# its CSV holds only this object's poses). The 8 eval_det CSVs are merged afterwards for the
# official BOP19 AR, to compare with tok4x (0.668 with detections). DONE files per step, so a
# restarted job continues.
#
# Usage (from the project root, inside a job), one object per job:
#   CPUS=16 bash rcp/train.sh lmo-m10-obj8-001 bash scripts/m10_single.sh 8
# Recipe = tok4x (scripts/m4_multi.sh): token conditioning, domain_gap, no symmetric loss (as tok4x
# was trained), bbox_visib crops, 150 epochs, 21,104 samples per epoch (tok4x: 168,832 for 8
# objects = 21,104 per object), ~3 h at 384 px.
set -euo pipefail
: "${RUN_DIR:?run inside a train job (RUN_DIR not set)}"
export PYTHONPATH=.
EPOCHS=${EPOCHS:-150}
VIT=${VIT:-vit_b_16_384}
SPE=${SPE:-21104}
DET=$(ls "$NAS_HOME"/datasets/bop_detections/default_task1/bop23_default_detections_for_task1/gdrnppdet-pbr/gdrnppdet-pbr_lmo-test_*.json)

(( $# == 1 )) || { echo "usage: $0 OBJ_ID (one of 1 5 6 8 9 10 11 12)"; exit 5; }
obj=$1
[[ " 1 5 6 8 9 10 11 12 " == *" $obj "* ]] || { echo "not an LM-O object: $obj"; exit 5; }
out=$RUN_DIR/obj$obj
mkdir -p "$out"
if [[ ! -e $out/TRAINED ]]; then
    echo "== obj $obj: train ($(date '+%F %T'))"
    .venv/bin/python src/train.py --dataset LMO --bop_obj_ids "$obj" --bop_val_split val \
        --max_val_samples 2000 --rotation_format matrix --rotation_loss 6d_relative_frobenius \
        --max_lr 1e-4 --min_lr 1e-6 --epochs "$EPOCHS" --num_hidden_layers 0 --hidden_layer_dim 0 \
        --nb_class_tokens 1 --batch_size 32 --vit_model "$VIT" --optimizer muon \
        --scheduler cosineannealingwarmrestarts --num_workers 14 --merge_outputs --no_mlp \
        --bbox_crop_percent 10 --depth_mode site --translation_loss relative_translation \
        --log_dir "$out/" --model_save_path "$out/model.pth" --resume auto \
        --object_conditioning token --no-symmetric_rotation_loss --bop_pixel_aug domain_gap \
        --bop_epoch_samples "$SPE" --bop_sampler_seed 0
    touch "$out/TRAINED"
fi
eval_args=(--dataset lmo --obj_ids "$obj" --model_weights "$out/last.pt" --vit_model "$VIT"
           --depth_mode site --object_conditioning token --splits test)
if [[ ! -e $out/eval_test/DONE ]]; then
    echo "== obj $obj: eval test, GT boxes ($(date '+%F %T'))"
    .venv/bin/python src/bop_evaluate.py "${eval_args[@]}" --out_dir "$out/eval_test"
    touch "$out/eval_test/DONE"
fi
if [[ ! -e $out/eval_det/DONE ]]; then
    echo "== obj $obj: eval test, detections ($(date '+%F %T'))"
    .venv/bin/python src/bop_evaluate.py "${eval_args[@]}" --detections "$DET" --out_dir "$out/eval_det"
    touch "$out/eval_det/DONE"
fi
echo "== all done ($(date '+%F %T'))"
