#!/usr/bin/env bash
# M19 R2: train our render-and-compare refiner on LM-O PBR (src/train_refiner.py), then refine the
# poses of the adopted LM-O model (tlwocc 4x, seed a) with 0..4 iterations: noisy-box validation
# (LM real validation images) and the LM-O test with the BOP default detections + the official
# BOP19 eval. Iterations 0 is the same code path without the refiner (the baseline of the rule).
# DONE files per step, so a restarted job continues (training resumes from last.pt).
#
# Usage (from the project root, inside a job), one variant per job; a trailing "2" = seed 1:
#   CPUS=16 bash rcp/train.sh lmo-m19-cnxa-001 bash scripts/m19_refiner.sh cnxa
# Variants (the learning rate is untuned: run both, with seeds):
#   cnxa   ConvNeXt-B refiner, lr 5e-5 (head x3)
#   cnxb   same, lr 1e-4
#   cnxs   as cnxa, starting-pose noise weighted towards small errors (--noise dr: 60 / 35 / 5 %)
#   cnxp   as cnxa, smaller starting errors for PnP poses (--noise pnp); evaluated on the PnP poses
#          of the multi-head model lmo-m18-densesym-001 (adopted 2026-11-21) instead of tlwocc 4x
#   cnxq   as cnxp with a single small noise level (--noise small: 3 deg, 0.01, 0.015)
set -euo pipefail
: "${RUN_DIR:?run inside a train job (RUN_DIR not set)}"
export PYTHONPATH=.
EPOCHS=${EPOCHS:-100}
DET=$(ls "$NAS_HOME"/datasets/bop_detections/default_task1/bop23_default_detections_for_task1/gdrnppdet-pbr/gdrnppdet-pbr_lmo-test_*.json)
DR=$NAS_HOME/outputs/FastPose-ViT/runs/lmo-m13-tlwocc4x-001/tlwocc/last.pt  # the poses to refine

(( $# == 1 )) || { echo "usage: $0 VARIANT[2]"; exit 5; }
VARIANTS=" cnxa cnxb cnxs cnxp cnxq "
v=$1 base=$1 seed=0
if [[ $VARIANTS != *" $v "* && $v == *2 ]]; then
    base=${v%2} seed=1
fi
NOISE=mix
case $base in
    cnxa) LR=5e-5 ;;
    cnxb) LR=1e-4 ;;
    cnxs) LR=5e-5 NOISE=dr ;;
    cnxp) LR=5e-5 NOISE=pnp ;;
    cnxq) LR=5e-5 NOISE=small ;;
    *) echo "unknown variant $v"; exit 5 ;;
esac
out=$RUN_DIR/$v
mkdir -p "$out"
if [[ ! -e $out/TRAINED ]]; then
    echo "== $v: train refiner, seed $seed ($(date '+%F %T'))"
    .venv/bin/python src/train_refiner.py --dataset LMO --out_dir "$out" --resume auto \
        --vit_model convnext_base --epochs "$EPOCHS" --lr "$LR" --seed "$seed" \
        --num_workers "${NUM_WORKERS:-12}" --occlusion_aug --noise "$NOISE"
    touch "$out/TRAINED"
fi
POSE_ARGS=()
if [[ $base == cnxp || $base == cnxq ]]; then  # PnP poses of the multi-head model
    DR=$NAS_HOME/outputs/FastPose-ViT/runs/lmo-m18-densesym-001/densesym/last.pt
    POSE_ARGS=(--dense_heads --pose_mode pnp)
fi
EVAL=(.venv/bin/python src/bop_evaluate.py --dataset lmo --model_weights "$DR" --vit_model vit_b_16_384
      --depth_mode site --object_conditioning token --refine_weights "$out/last.pt"
      --refine_vit_model convnext_base --num_workers "${NUM_WORKERS:-12}" ${POSE_ARGS[@]+"${POSE_ARGS[@]}"})
for n in 0 1 2 3 4; do
    if [[ ! -e $out/eval_valn_r$n/DONE ]]; then  # noisy boxes: see scripts/m12_valeval.sh
        echo "== $v: validation, noisy boxes, $n refinement iterations ($(date '+%F %T'))"
        "${EVAL[@]}" --splits val_real --box_noise 0.1 --refine_iterations "$n" --out_dir "$out/eval_valn_r$n"
        .venv/bin/python scripts/val_score.py "$out/eval_valn_r$n" > "$out/eval_valn_r$n/val_score.txt"
        touch "$out/eval_valn_r$n/DONE"
    fi
done
for n in 0 1 2 3 4; do
    if [[ ! -e $out/eval_det_r$n/DONE ]]; then
        echo "== $v: LM-O test, detections, $n refinement iterations ($(date '+%F %T'))"
        "${EVAL[@]}" --splits test --detections "$DET" --refine_iterations "$n" --out_dir "$out/eval_det_r$n"
        bash scripts/bop_eval.sh "$out/eval_det_r$n" fastposevitmo_lmo-test.csv > "$out/eval_det_r$n/bop_eval_test.log" 2>&1
        touch "$out/eval_det_r$n/DONE"
    fi
done
echo "== $v: all done ($(date '+%F %T'))"
