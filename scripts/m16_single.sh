#!/usr/bin/env bash
# M16 (2026-11-18, supervisor + user: re-run one network per object on all datasets): one
# single-object model, with the recipe of that dataset's shared tok4x and the same per-object
# budget (4x = 21,104 samples per epoch per object, 150 epochs, ~3 h), as scripts/m10_single.sh
# did for LM-O. Then, for that object only: validation and LM-O / YCB-V / T-LESS test with GT
# boxes and with the BOP default detections. The per-object test CSVs are merged afterwards
# (scripts/m10_merge.py + scripts/bop_eval.sh) and the validation runs pooled
# (scripts/m16_merge_valn.py), to compare with the shared tok4x / tok4x2 by the adoption rule.
# DONE files per step, so a restarted job continues (training resumes from last.pt).
#
# Usage (from the project root, inside a job), one object per job, SEED 0 (default) or 1:
#   CPUS=16 bash rcp/train.sh ycbv-m16-obj3-001 bash scripts/m16_single.sh ycbv 3
#   CPUS=16 bash rcp/train.sh ycbv-m16-obj3s1-001 bash scripts/m16_single.sh ycbv 3 1
# Recipe = the dataset's tok4x: token, domain_gap, bbox_visib, 384 px; symmetric loss off for
# LM-O (tok4x was trained without it), on for YCB-V and T-LESS (with translation offsets).
# Validation: LM-O LM real val (GT boxes and noisy boxes, eval_val / eval_valn); YCB-V the four
# held-out real videos (eval_valreal / eval_valrealn; objects absent from them are skipped);
# T-LESS the held-out train_primesense rings (M17; eval_valreal / eval_valrealn).
set -euo pipefail
: "${RUN_DIR:?run inside a train job (RUN_DIR not set)}"
export PYTHONPATH=.
EPOCHS=${EPOCHS:-150}
VIT=${VIT:-vit_b_16_384}
SPE=${SPE:-21104}
(( $# >= 2 && $# <= 3 )) || { echo "usage: $0 DATASET(lmo|ycbv|tless) OBJ_ID [SEED]"; exit 5; }
ds=$1 obj=$2 seed=${3:-0}
case $ds in
    lmo)   DS=LMO;   objs=" 1 5 6 8 9 10 11 12 "; sym=--no-symmetric_rotation_loss ;;
    ycbv)  DS=YCBV;  objs=" $(seq -s ' ' 1 21) "; sym=--symmetric_rotation_loss ;;
    tless) DS=TLESS; objs=" $(seq -s ' ' 1 30) "; sym=--symmetric_rotation_loss ;;
    *) echo "unknown dataset $ds"; exit 5 ;;
esac
[[ $objs == *" $obj "* ]] || { echo "not an object of $ds: $obj"; exit 5; }
DET=$(ls "$NAS_HOME"/datasets/bop_detections/default_task1/bop23_default_detections_for_task1/gdrnppdet-pbr/gdrnppdet-pbr_${ds}-test_*.json)
out=$RUN_DIR/obj$obj
[[ $seed != 0 ]] && out=$RUN_DIR/obj${obj}_s$seed
mkdir -p "$out"
if [[ ! -e $out/TRAINED ]]; then
    echo "== $ds obj $obj: train, seed $seed ($(date '+%F %T'))"
    .venv/bin/python src/train.py --dataset "$DS" --bop_obj_ids "$obj" --bop_val_split val \
        --max_val_samples 2000 --rotation_format matrix --rotation_loss 6d_relative_frobenius \
        --max_lr 1e-4 --min_lr 1e-6 --epochs "$EPOCHS" --num_hidden_layers 0 --hidden_layer_dim 0 \
        --nb_class_tokens 1 --batch_size 32 --vit_model "$VIT" --optimizer muon \
        --scheduler cosineannealingwarmrestarts --num_workers 14 --merge_outputs --no_mlp \
        --bbox_crop_percent 10 --depth_mode site --translation_loss relative_translation \
        --log_dir "$out/" --model_save_path "$out/model.pth" --resume auto \
        --object_conditioning token "$sym" --bop_pixel_aug domain_gap \
        --bop_epoch_samples "$SPE" --bop_sampler_seed "$seed"
    touch "$out/TRAINED"
fi
eval_args=(--dataset "$ds" --obj_ids "$obj" --model_weights "$out/last.pt" --vit_model "$VIT"
           --depth_mode site --object_conditioning token)
run_eval() {  # NAME SPLIT [extra args]: one evaluation step with its DONE file
    local name=$1 split=$2; shift 2
    [[ -e $out/$name/DONE ]] && return 0
    echo "== $ds obj $obj: $name ($(date '+%F %T'))"
    .venv/bin/python src/bop_evaluate.py "${eval_args[@]}" --splits "$split" "$@" \
        --out_dir "$out/$name"
    [[ $split == test ]] || .venv/bin/python scripts/val_score.py "$out/$name" > "$out/$name/val_score.txt"
    touch "$out/$name/DONE"
}
case $ds in
    lmo)
        run_eval eval_val val_real
        run_eval eval_valn val_real --box_noise 0.1 ;;
    ycbv)
        # objects without instances in the four held-out real videos (TASKS.md M11 B3)
        if [[ " 1 5 9 10 15 17 18 19 21 " != *" $obj "* ]]; then
            run_eval eval_valreal val_real
            run_eval eval_valrealn val_real --box_noise 0.1
        fi ;;
    tless)
        run_eval eval_valreal val_real
        run_eval eval_valrealn val_real --box_noise 0.1 ;;
esac
run_eval eval_test test
run_eval eval_det test --detections "$DET"
echo "== all done ($(date '+%F %T'))"
