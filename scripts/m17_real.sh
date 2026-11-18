#!/usr/bin/env bash
# M17 (2026-11-18, supervisor: train with PBR only AND with PBR + real, keep them apart): LM-O and
# T-LESS trained on PBR + their real images (--bop_train_split train_mix; YCB-V runs in
# scripts/m14_ycbv_real.sh). The recipe is the current one (token, domain_gap, symmetric loss,
# bbox_visib, 384 px, 150 epochs, + tlw + occl, adopted 2026-11-18), 2x budget. Each object's
# samples are 25 % real (--bop_real_fraction 0.25): by the row counts alone LM-O would see ~3 %
# real crops (about 1,000 real vs 40,000 PBR rows per object).
#   LM-O  real: LM test_all of the 8 objects without scene 2 (the LM-O test sequence) and without
#         the BOP19 images (= val_real). Baseline: the 2x tlwocc / tlwocc2 runs of
#         scripts/m10_ablations.sh (test 0.664 / 0.658, eval_valn 0.885 / 0.880).
#   T-LESS real: train_primesense (black background, so --bop_real_background pastes the object
#         on a random PBR image), without the validation rings 1, 4, 7. Baseline: base2x /
#         base2x2 here (PBR only, same recipe and budget; the 4x tok4x runs used older code).
# Then: validation (LM-O: LM real val, GT and noisy boxes; T-LESS: the held-out train_primesense
# rings, GT and noisy boxes), test with the BOP default detections and the official BOP19 eval.
# DONE files per step, so a restarted job continues.
#
# Usage (repo root, inside a job), one variant per job; a trailing "2" = seed 1:
#   CPUS=16 bash rcp/train.sh lmo-m17-real2x-001 bash scripts/m17_real.sh lmo real2x
#   CPUS=16 bash rcp/train.sh tless-m17-base2x-001 bash scripts/m17_real.sh tless base2x
set -euo pipefail
: "${RUN_DIR:?run inside a train job (RUN_DIR not set)}"
export PYTHONPATH=.
EPOCHS=${EPOCHS:-150}
(( $# == 2 )) || { echo "usage: $0 (lmo|tless) (real2x|base2x)[2]"; exit 5; }
ds=$1 v=$2 base=${2%2} seed=0
[[ $v != "$base" ]] && seed=1
case $ds in
    lmo)   DS=LMO;   SPE=${SPE:-$((2 * 42208))};  val=(eval_val val_real eval_valn) ;;
    tless) DS=TLESS; SPE=${SPE:-$((633120 / 2))}; val=(eval_valreal val_real eval_valrealn) ;;
    *) echo "unknown dataset $ds"; exit 5 ;;
esac
case $base in
    real2x) data=(--bop_train_split train_mix --bop_real_fraction 0.25)
            [[ $ds == tless ]] && data+=(--bop_real_background) ;;
    base2x) [[ $ds == tless ]] || { echo "base2x is for tless (LM-O baseline: tlwocc)"; exit 5; }
            data=() ;;
    *) echo "unknown variant $v"; exit 5 ;;
esac
DET=$(ls "$NAS_HOME"/datasets/bop_detections/default_task1/bop23_default_detections_for_task1/gdrnppdet-pbr/gdrnppdet-pbr_${ds}-test_*.json)
out=$RUN_DIR/$v
mkdir -p "$out"
if [[ ! -e $out/TRAINED ]]; then
    echo "== $ds $v: train, seed $seed ($(date '+%F %T'))"
    .venv/bin/python src/train.py --dataset "$DS" --bop_val_split val "${data[@]}" \
        --max_val_samples 2000 --rotation_format matrix --rotation_loss 6d_relative_frobenius \
        --max_lr 1e-4 --min_lr 1e-6 --epochs "$EPOCHS" --num_hidden_layers 0 --hidden_layer_dim 0 \
        --nb_class_tokens 1 --batch_size 32 --vit_model vit_b_16_384 --optimizer muon \
        --scheduler cosineannealingwarmrestarts --num_workers 14 --merge_outputs --no_mlp \
        --bbox_crop_percent 10 --depth_mode site --translation_loss relative_translation \
        --log_dir "$out/" --model_save_path "$out/model.pth" --resume auto \
        --object_conditioning token --symmetric_rotation_loss --bop_pixel_aug domain_gap \
        --translation_loss_weight 2 --bop_occlusion_aug \
        --bop_epoch_samples "$SPE" --bop_sampler_seed "$seed"
    touch "$out/TRAINED"
fi
eval_args=(--dataset "$ds" --model_weights "$out/last.pt" --vit_model vit_b_16_384
           --depth_mode site --object_conditioning token)
for step in "${val[0]}:" "${val[2]}:--box_noise 0.1"; do
    name=${step%%:*} extra=${step#*:}
    [[ -e $out/$name/DONE ]] && continue
    echo "== $ds $v: $name ($(date '+%F %T'))"
    # shellcheck disable=SC2086  # extra is empty or two words
    .venv/bin/python src/bop_evaluate.py "${eval_args[@]}" --splits "${val[1]}" $extra \
        --out_dir "$out/$name"
    .venv/bin/python scripts/val_score.py "$out/$name" > "$out/$name/val_score.txt"
    touch "$out/$name/DONE"
done
if [[ ! -e $out/eval_det/DONE ]]; then
    echo "== $ds $v: test, detections ($(date '+%F %T'))"
    .venv/bin/python src/bop_evaluate.py "${eval_args[@]}" --splits test --detections "$DET" \
        --out_dir "$out/eval_det"
    bash scripts/bop_eval.sh "$out/eval_det" "fastposevitmo_${ds}-test.csv" \
        > "$out/eval_det/bop_eval_test.log" 2>&1
    touch "$out/eval_det/DONE"
fi
echo "== $ds $v: all done ($(date '+%F %T'))"
