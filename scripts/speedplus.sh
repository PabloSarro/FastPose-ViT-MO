#!/bin/bash
set -e

# =============================================================================
# SPEED+ Augmentation Experiment
# Test augmentation strategies on vit_b_16 and vit_b_16_384
# =============================================================================

# Dataset configuration
DATASET="SPEED_PLUS_FIXED"
SCHEDULER="cosineannealingwarmrestarts"
BATCH_SIZE=8

# Rotation configuration
ROTATION_FORMAT="matrix"
ROTATION_LOSS="6d_relative_frobenius"

# MLP architecture
NUM_HIDDEN_LAYERS=0
HIDDEN_LAYER_DIM=0
NB_CLASS_TOKENS=1

# Experiment variables
models=("vit_b_16" "vit_b_16_384")
aug_flags=("--no_pixel_augmentation" "--domain_gap_pixel_augmentation" "--domain_gap_pixel_augmentation --do_style_aug")
aug_names=("no-aug" "domain-gap" "domain-gap-style")
grayscale_flags=("" "--convert_to_grayscale")
grayscale_suffixes=("rgb" "gray")

for model_idx in "${!models[@]}"; do
    VIT_MODEL="${models[$model_idx]}"
    m=$(echo "$VIT_MODEL" | sed 's/_/-/g')

    for gray_idx in "${!grayscale_flags[@]}"; do
        GRAYSCALE_FLAG="${grayscale_flags[$gray_idx]}"
        GRAYSCALE_SUFFIX="${grayscale_suffixes[$gray_idx]}"

        for aug_idx in "${!aug_flags[@]}"; do
            AUG_FLAGS="${aug_flags[$aug_idx]}"
            AUG_NAME="${aug_names[$aug_idx]}"

            RES_DIR="results/paper/speedplus_augmentation_experiment/${VIT_MODEL}/${AUG_NAME}-${GRAYSCALE_SUFFIX}"
            mkdir -p "$RES_DIR"
            MODEL_WEIGHTS="$RES_DIR/model.pth"
            LOG_DIR="$RES_DIR/"

            echo "Training $m — $AUG_NAME ($GRAYSCALE_SUFFIX)"

            python3 src/train.py \
              --model_save_path "$MODEL_WEIGHTS" \
              --log_dir "$LOG_DIR" \
              --rotation_format "$ROTATION_FORMAT" \
              --rotation_loss "$ROTATION_LOSS" \
              --translation_loss relative_translation \
              --max_lr 1e-4 \
              --min_lr 1e-6 \
              --epochs 150 \
              --num_hidden_layers "$NUM_HIDDEN_LAYERS" \
              --hidden_layer_dim "$HIDDEN_LAYER_DIM" \
              --nb_class_tokens "$NB_CLASS_TOKENS" \
              --batch_size "$BATCH_SIZE" \
              --vit_model "$VIT_MODEL" \
              --dataset SPEED_PLUS --dataset_root_dir "$DATASET/" \
              --optimizer muon \
              --scheduler "$SCHEDULER" \
              --num_workers 8 \
              --merge_outputs \
              --no_mlp \
              --bbox_crop_percent 30 $AUG_FLAGS $GRAYSCALE_FLAG
        done
    done
done

echo "All SPEED+ augmentation experiments completed"
