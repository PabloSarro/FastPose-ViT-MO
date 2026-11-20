#!/usr/bin/env bash
# M18 multi-head network: the tlwocc recipe (LM-O, token conditioning, symmetric rotation loss,
# 384 px, ViT-B) + dense heads (visible mask + object coordinates, src/models.py DenseHead), then
# the four pose modes of src/pnp.py evaluated on the same weights, as scripts/m10_ablations.sh:
# validation (GT boxes and noisy boxes) first, the LM-O test with the BOP default detections and
# the official BOP19 eval (last.pt). DONE files per step, so a restarted job continues.
#
# Usage (from the project root, inside a job), one variant per job; a trailing "2" = seed 1:
#   CPUS=16 bash rcp/train.sh lmo-m18-dense-001 bash -c 'SPE=$((2 * 42208)) bash scripts/m18_dense.sh dense'
# Variants (loss weights are untuned, TASKS.md M18.6 (c): run them in parallel, with seeds):
#   dense     mask BCE weight 1, XYZ L1 weight 1
#   densex5   XYZ L1 weight 5  (the XYZ term is small next to the pose losses)
#   densexl   XYZ L1 weight 0.2
# Baseline for the DR head: tlwocc / tlwocc2 (same SPE); for the pose modes: DR of the same run.
# Compare per pose mode with `scripts/val_score.py` on eval_val_<mode>, eval_valn_<mode>.
set -euo pipefail
: "${RUN_DIR:?run inside a train job (RUN_DIR not set)}"
export PYTHONPATH=.
EPOCHS=${EPOCHS:-150}
SPE=${SPE:-$((2 * 42208))}
DET=$(ls "$NAS_HOME"/datasets/bop_detections/default_task1/bop23_default_detections_for_task1/gdrnppdet-pbr/gdrnppdet-pbr_lmo-test_*.json)
MODES=${MODES:-"dr pnp pnp_init dr_rot_pnp_t"}

(( $# == 1 )) || { echo "usage: $0 VARIANT[2]"; exit 5; }
VARIANTS=" dense densex5 densexl "
v=$1 base=$1 seed=0
if [[ $VARIANTS != *" $v "* && $v == *2 ]]; then
    base=${v%2} seed=1
fi
VIT=vit_b_16_384 DEPTH=site
MASK_W=1 XYZ_W=1
case $base in
    dense)   ;;
    densex5) XYZ_W=5 ;;
    densexl) XYZ_W=0.2 ;;
    *) echo "unknown variant $v"; exit 5 ;;
esac
out=$RUN_DIR/$v
mkdir -p "$out"
NGPU=$(nvidia-smi -L | wc -l)
PY=(.venv/bin/python)
(( NGPU > 1 )) && PY=(.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node="$NGPU")
if [[ ! -e $out/TRAINED ]]; then
    echo "== $v: train, seed $seed, $NGPU GPU(s) ($(date '+%F %T'))"
    "${PY[@]}" src/train.py --dataset LMO --bop_val_split val \
        --max_val_samples 2000 --rotation_format matrix --rotation_loss 6d_relative_frobenius \
        --max_lr 1e-4 --min_lr 1e-6 --epochs "$EPOCHS" --num_hidden_layers 0 --hidden_layer_dim 0 \
        --nb_class_tokens 1 --batch_size 32 --vit_model "$VIT" --optimizer muon \
        --scheduler cosineannealingwarmrestarts --num_workers "${NUM_WORKERS:-14}" --merge_outputs --no_mlp \
        --bbox_crop_percent 10 --depth_mode "$DEPTH" --translation_loss relative_translation \
        --log_dir "$out/" --model_save_path "$out/model.pth" --resume auto \
        --object_conditioning token --symmetric_rotation_loss --bop_pixel_aug domain_gap \
        --translation_loss_weight 2 --bop_occlusion_aug \
        --dense_heads --dense_size 64 --mask_loss_weight "$MASK_W" --xyz_loss_weight "$XYZ_W" \
        --bop_epoch_samples "$SPE" --bop_sampler_seed "$seed"
    touch "$out/TRAINED"
fi
EVAL=(.venv/bin/python src/bop_evaluate.py --dataset lmo --model_weights "$out/last.pt"
      --vit_model "$VIT" --depth_mode "$DEPTH" --object_conditioning token --dense_heads)
for mode in $MODES; do
    if [[ ! -e $out/eval_val_$mode/DONE ]]; then
        echo "== $v: eval validation, GT boxes, $mode ($(date '+%F %T'))"
        "${EVAL[@]}" --splits val_real val_pbr --pose_mode "$mode" --out_dir "$out/eval_val_$mode"
        .venv/bin/python scripts/val_score.py "$out/eval_val_$mode" > "$out/eval_val_$mode/val_score.txt"
        touch "$out/eval_val_$mode/DONE"
    fi
    if [[ ! -e $out/eval_valn_$mode/DONE ]]; then  # noisy boxes: see scripts/m12_valeval.sh
        "${EVAL[@]}" --splits val_real --pose_mode "$mode" --box_noise 0.1 --out_dir "$out/eval_valn_$mode"
        .venv/bin/python scripts/val_score.py "$out/eval_valn_$mode" > "$out/eval_valn_$mode/val_score.txt"
        touch "$out/eval_valn_$mode/DONE"
    fi
done
for mode in $MODES; do
    if [[ ! -e $out/eval_det_$mode/DONE ]]; then
        echo "== $v: eval LM-O test, detections, $mode ($(date '+%F %T'))"
        "${EVAL[@]}" --splits test --pose_mode "$mode" --detections "$DET" --out_dir "$out/eval_det_$mode"
        bash scripts/bop_eval.sh "$out/eval_det_$mode" fastposevitmo_lmo-test.csv \
            > "$out/eval_det_$mode/bop_eval_test.log" 2>&1
        touch "$out/eval_det_$mode/DONE"
    fi
done
echo "== $v: all done ($(date '+%F %T'))"
