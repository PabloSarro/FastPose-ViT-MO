#!/usr/bin/env bash
# M6: multi-GPU (DDP) speed benchmark. Trains the M4 setting (ViT-B/16 384 px, 8 LM-O objects,
# token conditioning, domain_gap aug, global batch 32) for 3 short epochs of 16,896 samples and
# records GPU utilisation. Uses every GPU of the job (torchrun when there are several).
#
# Usage, one job per GPU count (scale CPUs and memory with the GPUs; workers are per process):
#   bash rcp/train.sh lmo-m6-bench1-001 bash scripts/ddp_bench.sh
#   GPUS=2 CPUS=32 MEMORY=64Gi bash rcp/train.sh lmo-m6-bench2-001 bash scripts/ddp_bench.sh
#   GPUS=4 CPUS=64 MEMORY=128Gi bash rcp/train.sh lmo-m6-bench4-001 bash scripts/ddp_bench.sh
#   GPUS=2 CPUS=24 MEMORY=64Gi bash rcp/train.sh NAME bash -c 'WORKERS=11 bash scripts/ddp_bench.sh'
#   (fewer CPUs: 2 x (11 workers + 1 main process) = 24; WORKERS must be set inside the job)
# Result: the last line, "BENCH gpus=N samples/s=... gpu_util=...%" (epochs 2-3: epoch 1 compiles).
set -euo pipefail
: "${RUN_DIR:?run inside a train job (RUN_DIR not set)}"
export PYTHONPATH=.
WORKERS=${WORKERS:-14}  # DataLoader workers per process (per GPU)
out=$RUN_DIR/bench
mkdir -p "$out"
NGPU=$(nvidia-smi -L | wc -l)
if (( NGPU > 1 )); then
    TRAIN=(.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node="$NGPU" src/train.py)
else
    TRAIN=(.venv/bin/python src/train.py)
fi

nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits -l 5 \
    > "$out/gpu_util.csv" &
smi=$!
trap 'kill $smi 2>/dev/null || true' EXIT

"${TRAIN[@]}" --dataset LMO --bop_val_split val --max_val_samples 256 \
    --rotation_format matrix --rotation_loss 6d_relative_frobenius --max_lr 1e-4 --min_lr 1e-6 \
    --epochs 3 --num_hidden_layers 0 --hidden_layer_dim 0 --nb_class_tokens 1 --batch_size 32 \
    --vit_model vit_b_16_384 --optimizer muon --scheduler cosineannealingwarmrestarts \
    --num_workers "$WORKERS" --merge_outputs --no_mlp --bbox_crop_percent 10 --depth_mode site \
    --translation_loss relative_translation --bop_pixel_aug domain_gap --bop_epoch_samples 16896 \
    --object_conditioning token --log_dir "$out/" --model_save_path "$out/model.pth" --resume auto \
    2>&1 | tee "$out/train.txt"
kill $smi 2>/dev/null || true

# Throughput of epochs 2-3; GPU utilisation over the samples with a busy GPU (> 0 %)
rate=$(grep -o "Train time: [0-9.]* s, [0-9.]* samples/s" "$out/train.txt" | tail -n 2 \
    | awk '{s += $5} END {printf "%.1f", s / NR}')
util=$(awk -F', ' '$2 > 0 {s += $2; n++} END {printf "%.0f", n ? s / n : 0}' "$out/gpu_util.csv")
echo "BENCH gpus=$NGPU workers/gpu=$WORKERS samples/s=$rate gpu_util=$util%"
