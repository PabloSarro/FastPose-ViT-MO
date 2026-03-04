#!/bin/bash
set -e

# =============================================================================
# Object Detector Experiment
# Train LW-DETR with different model variants, augmentations, and grayscale
# =============================================================================

DATASET="SPEED_FIXED"

# Model variants → pretrained weights
declare -A model_configs=(
    ["small"]="lwdetr_weights/LWDETR_small_60e_coco.pth"
    ["tiny"]="lwdetr_weights/LWDETR_tiny_60e_coco.pth"
)

# Augmentation configurations (name:flags)
declare -a aug_configs=(
    "baseline:"
    "domain-gap:--domain_gap_pixel_augmentation"
    "domain-gap-style:--domain_gap_pixel_augmentation --do_style_aug"
)

# Grayscale options
declare -a grayscale_flags=("" "--convert_to_grayscale")
declare -a grayscale_suffixes=("rgb" "gray")

for model_variant in "${!model_configs[@]}"; do
    pretrain_weights="${model_configs[$model_variant]}"

    for gray_idx in "${!grayscale_flags[@]}"; do
        grayscale_flag="${grayscale_flags[$gray_idx]}"
        grayscale_suffix="${grayscale_suffixes[$gray_idx]}"

        for config in "${aug_configs[@]}"; do
            IFS=':' read -r NAME FLAGS <<< "$config"

            RES_DIR="results/optimal/object_detector/${model_variant}/${NAME}-${grayscale_suffix}"
            mkdir -p "$RES_DIR"
            MODEL_WEIGHTS="$RES_DIR/model.pth"
            LOG_DIR="$RES_DIR/"

            echo "Training detector $model_variant — $NAME ($grayscale_suffix)"

            python3 object_detector/train.py \
              --dataset_root_dir "$DATASET/" \
              --model_variant "$model_variant" \
              --batch_size 16 \
              --max_lr 5e-4 \
              --min_lr 5e-5 \
              --scheduler cosineannealingwarmrestarts \
              --epochs 70 \
              --pretrain_weights "$pretrain_weights" \
              --num_workers 8 \
              --model_save_path "$MODEL_WEIGHTS" \
              --log_dir "$LOG_DIR" \
              --static_training $FLAGS $grayscale_flag
        done
    done
done

echo "All object detector experiments completed"
