#!/bin/bash
set -e

# =============================================================================
# Class Tokens Experiment
# Test different numbers of class tokens: 1, 2, 3, 4, 5, 7, 10
# =============================================================================

# Model configuration
VIT_MODEL="vit_b_16"
DATASET="SPEED_FIXED"
SCHEDULER="cosineannealingwarmrestarts"
BATCH_SIZE=8

# Rotation configuration
ROTATION_FORMAT="matrix"
ROTATION_LOSS="6d_relative_frobenius"

# MLP architecture
NUM_HIDDEN_LAYERS=0
HIDDEN_LAYER_DIM=0

# Experiment variable
nb_class_tokens_array=(1 2 3 4 5 7 10)

m=$(echo "$VIT_MODEL" | sed 's/_/-/g')

for nb_tokens in "${nb_class_tokens_array[@]}"; do
    NAME="tokens-${nb_tokens}"

    RES_DIR="results/optimal/nb_class_tokens/${VIT_MODEL}/${NAME}"
    mkdir -p "$RES_DIR"
    MODEL_WEIGHTS="$RES_DIR/model.pth"
    LOG_DIR="$RES_DIR/"

    echo "Training $m with $nb_tokens class tokens"

    python3 src/train.py \
      --model_save_path "$MODEL_WEIGHTS" \
      --log_dir "$LOG_DIR" \
      --rotation_format "$ROTATION_FORMAT" \
      --rotation_loss "$ROTATION_LOSS" \
      --translation_loss relative_translation \
      --max_lr 1e-4 \
      --min_lr 1e-6 \
      --epochs 310 \
      --num_hidden_layers "$NUM_HIDDEN_LAYERS" \
      --hidden_layer_dim "$HIDDEN_LAYER_DIM" \
      --nb_class_tokens "$nb_tokens" \
      --batch_size "$BATCH_SIZE" \
      --vit_model "$VIT_MODEL" \
      --dataset SPEED --dataset_root_dir "$DATASET/" \
      --optimizer muon \
      --scheduler "$SCHEDULER" \
      --num_workers 8 \
      --merge_outputs \
      --no_mlp \
      --no_pixel_augmentation \
      --bbox_crop_percent 10 && \
    python3 src/evaluate.py \
      --model_weights "$MODEL_WEIGHTS" \
      --log_dir "$LOG_DIR" \
      --rotation_format "$ROTATION_FORMAT" \
      --num_hidden_layers "$NUM_HIDDEN_LAYERS" \
      --hidden_layer_dim "$HIDDEN_LAYER_DIM" \
      --nb_class_tokens "$nb_tokens" \
      --batch_size "$BATCH_SIZE" \
      --vit_model "$VIT_MODEL" \
      --dataset SPEED --dataset_root_dir "$DATASET/" \
      --output_json "$RES_DIR/results.json" \
      --num_workers 8 \
      --merge_outputs \
      --no_mlp
done

echo "All class tokens experiments completed"
