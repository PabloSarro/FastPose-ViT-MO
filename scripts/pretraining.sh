#!/bin/bash
set -e

# =============================================================================
# Pretraining Experiment
# Compare pretrained vs from-scratch ViT-B-16
# =============================================================================

# Configuration
DATASET="SPEED_FIXED"
SCHEDULER="cosineannealingwarmrestarts"
BATCH_SIZE=8

# Rotation configuration
ROTATION_FORMAT="matrix"
ROTATION_LOSS="6d_relative_frobenius"

# MLP architecture
NUM_HIDDEN_LAYERS=0
HIDDEN_LAYER_DIM=0
NB_CLASS_TOKENS=1

# Experiment variable
vit_models=("vit_b_16" "vit_b_16_no_weights")

for VIT_MODEL in "${vit_models[@]}"; do
    m=$(echo "$VIT_MODEL" | sed 's/_/-/g')

    RES_DIR="results/optimal/pretraining/${VIT_MODEL}"
    mkdir -p "$RES_DIR"
    MODEL_WEIGHTS="$RES_DIR/model.pth"
    LOG_DIR="$RES_DIR/"

    echo "Training $m (pretraining experiment)"

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
      --nb_class_tokens "$NB_CLASS_TOKENS" \
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
      --nb_class_tokens "$NB_CLASS_TOKENS" \
      --batch_size "$BATCH_SIZE" \
      --vit_model "$VIT_MODEL" \
      --dataset SPEED --dataset_root_dir "$DATASET/" \
      --output_json "$RES_DIR/results.json" \
      --num_workers 8 \
      --merge_outputs \
      --no_mlp
done

echo "All pretraining experiments completed"
