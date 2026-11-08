#!/usr/bin/env bash
# M9 stretch: train our multi-class LW-DETR detector on LM-O PBR (object_detector/bop_detector.py),
# then use it instead of the BOP default detections: LM-O test detections (best.pth by val_pbr
# mAP), pose eval of the final pose model tok4x with them, official BOP19 eval. To compare with
# the default detections (GDRNPPDet): AR 0.668. DONE files per step, so a restarted job continues
# (training itself resumes from $RUN_DIR/<name>/last.pt).
#
# Usage (from the project root, inside a job), one variant per job:
#   CPUS=24 MEMORY=64Gi bash rcp/train.sh lmo-m9-dettiny-001 bash scripts/m9_det.sh tiny 0
#   VARIANT = tiny | small; SEED = training seed
set -euo pipefail
: "${RUN_DIR:?run inside a train job (RUN_DIR not set)}"
export PYTHONPATH=.
(( $# == 2 )) || { echo "usage: $0 VARIANT SEED"; exit 5; }
variant=$1 seed=$2
EPOCHS=${EPOCHS:-50}
WORKERS=${WORKERS:-22}
POSE=${POSE:-$NAS_HOME/outputs/FastPose-ViT/runs/lmo-m4-tok4x-001/tok4x/last.pt}
COCO=$NAS_HOME/weights/lwdetr/pretrain_weights/LWDETR_${variant}_60e_coco.pth
out=$RUN_DIR/$variant
mkdir -p "$out"

if [[ ! -e $out/TRAINED ]]; then
    echo "== $variant: train ($(date '+%F %T'))"
    .venv/bin/python object_detector/bop_detector.py train --dataset lmo --variant "$variant" \
        --pretrain "$COCO" --epochs "$EPOCHS" --seed "$seed" --num_workers "$WORKERS" \
        --out_dir "$out" --resume auto
    touch "$out/TRAINED"
fi
if [[ ! -e $out/det_test/DONE ]]; then
    echo "== $variant: detections on LM-O test ($(date '+%F %T'))"
    .venv/bin/python object_detector/bop_detector.py predict --dataset lmo --variant "$variant" \
        --weights "$out/best.pth" --out_dir "$out/det_test" --num_workers 8
    touch "$out/det_test/DONE"
fi
if [[ ! -e $out/eval_det/DONE ]]; then
    echo "== $variant: pose eval with our detections ($(date '+%F %T'))"
    .venv/bin/python src/bop_evaluate.py --dataset lmo --model_weights "$POSE" --splits test \
        --depth_mode site --object_conditioning token \
        --detections "$out/det_test/lwdetr_lmo-test.json" --out_dir "$out/eval_det"
    bash scripts/bop_eval.sh "$out/eval_det" fastposevitmo_lmo-test.csv \
        > "$out/eval_det/bop_eval_test.log" 2>&1
    touch "$out/eval_det/DONE"
fi
echo "== $variant: all done ($(date '+%F %T'))"
