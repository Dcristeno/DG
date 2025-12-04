#!/bin/bash
# NCR + IRRA Training Script with W&B Integration
# Based on your original script, enhanced for reproducibility & experiment tracking
# Usage: ./run.sh [--noise_ratio 0.2] [--seed 123]

set -e  # Exit on error

# ============== User Configuration (Editable) ==============
root_dir="/root/autodl-tmp"
output_dir="output"
DATASET_NAME="CUHK-PEDES"   # Options: CUHK-PEDES, ICFG-PEDES, RSTPReid
loss="GCL"                 # Options: GCL, TAL, TRL, SDM
batch_size=128
num_epoch=60
pretrain_choice="ViT-B/16"
gpu_id=0
seed=42
noise_ratio=0.0  # Set to 0.2 or 0.5 for NCR noisy experiments

# Parse command-line arguments (e.g., --noise_ratio 0.2)
while [[ $# -gt 0 ]]; do
    case $1 in
        --noise_ratio) noise_ratio="$2"; shift 2;;
        --seed)        seed="$2"; shift 2;;
        --dataset)     DATASET_NAME="$2"; shift 2;;
        --gpu)         gpu_id="$1#2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

# ============== Validation & Setup ==============
if [ ! -d "$root_dir" ]; then
    echo "❌ Error: root_dir does not exist: $root_dir"
    exit 1
fi

# Auto-create output_dir
mkdir -p "$output_dir"

# W&B config
WANDB_PROJECT="NCR-IRRA"
if (( $(echo "$noise_ratio > 0" | bc -l) )); then
    WANDB_NAME="${DATASET_NAME}_noise${noise_ratio}_${loss}_ViT"
else
    WANDB_NAME="${DATASET_NAME}_clean_${loss}_ViT"
fi

echo "🚀 Starting experiment:"
echo "   Dataset: $DATASET_NAME"
echo "   Loss: $loss"
echo "   Noise Ratio: $noise_ratio"
echo "   Pretrain: $pretrain_choice"
echo "   Output: $output_dir"
echo "   W&B: $WANDB_PROJECT / $WANDB_NAME"
echo "──────────────────────────────────────"

# ============== Launch Training ==============
CUDA_VISIBLE_DEVICES=$gpu_id \
WANDB_PROJECT="$WANDB_PROJECT" \
WANDB_NAME="$WANDB_NAME" \
    python demo.py \
    --name DG \
    --batch_size $batch_size \
    --root_dir "$root_dir" \
    --output_dir "$output_dir" \
    --dataset_name $DATASET_NAME \
    --loss_names $loss \
    --num_epoch $num_epoch \
    --pretrain_choice "$pretrain_choice" \
    --noise_ratio $noise_ratio \
    --seed $seed

echo "✅ Training completed. Check W&B dashboard for results."