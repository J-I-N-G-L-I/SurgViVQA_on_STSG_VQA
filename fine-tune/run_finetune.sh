#!/bin/bash
#
# SSGVQA Fine-tuning script for SurgViVQA on Aire HPC
#
# This script fine-tunes SurgViVQA on the SSGVQA dataset using LoRA (default)
# for efficient single-GPU training.
#
# Usage:
#   sbatch run_finetune.sh
#   # or directly:
#   bash run_finetune.sh
#
# Test split (NEVER for training): VID02, VID22, VID43, VID60, VID74
#
# After training completes, evaluate with:
#   bash run_eval.sh
#

# ============================================================================
# SLURM Configuration (for Aire HPC)
# ============================================================================
#SBATCH --job-name=surgvivqa_ssgvqa_finetune
#SBATCH --output=logs/finetune_%j.out
#SBATCH --error=logs/finetune_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# ============================================================================
# Environment Setup
# ============================================================================
echo "========================================"
echo "SurgViVQA Fine-tuning on SSGVQA"
echo "========================================"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "========================================"

# ============================================================================
# Resolve project root directory
# When submitted via sbatch, use SLURM_SUBMIT_DIR; otherwise use script location
# ============================================================================
if [ -n "${SLURM_SUBMIT_DIR}" ]; then
    # Running under SLURM - use the directory where sbatch was called
    PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
    # Running directly - resolve from script location
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
fi

cd "${PROJECT_ROOT}" || { echo "ERROR: Cannot cd to ${PROJECT_ROOT}"; exit 1; }
echo "Project root: ${PROJECT_ROOT}"
echo "Working directory: $(pwd)"

# Create logs directory
mkdir -p logs

# Activate conda environment (adjust as needed for your HPC)
# source ~/.bashrc
# conda activate surgvivqa

# ============================================================================
# Data Paths (adjust for your HPC environment)
# ============================================================================
SSGVQA_ROOT="/mnt/scratch/sc232jl/datasets/SSGVQA/ssg-qa/"
IMAGE_ROOT="/mnt/scratch/sc232jl/datasets/CholecT45/data"

# ============================================================================
# Training Configuration
# ============================================================================
# Default: LoRA fine-tuning (lower VRAM, ~8-12GB)
# For full fine-tuning, add --full-finetune and reduce batch size

EPOCHS=30
BATCH_SIZE=4
LR=2e-5
NUM_FRAMES=16
MAX_LENGTH=128
PROMPT_MODE="simple"  # or "choices" for longer prompts with all labels

# LoRA configuration
LORA_R=8
LORA_ALPHA=32
LORA_DROPOUT=0.1

# ============================================================================
# Output Configuration
# ============================================================================
CHECKPOINT_DIR="${PROJECT_ROOT}/fine-tune/ckpt"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="ssgvqa_lora_${PROMPT_MODE}_${TIMESTAMP}"
OUTPUT_DIR="${CHECKPOINT_DIR}/${RUN_NAME}"

echo ""
echo "Configuration:"
echo "  SSGVQA Root: ${SSGVQA_ROOT}"
echo "  Image Root: ${IMAGE_ROOT}"
echo "  Epochs: ${EPOCHS}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  Learning Rate: ${LR}"
echo "  Num Frames: ${NUM_FRAMES}"
echo "  Prompt Mode: ${PROMPT_MODE}"
echo "  Output Dir: ${OUTPUT_DIR}"
echo ""

# ============================================================================
# Run Training
# ============================================================================
python "${PROJECT_ROOT}/fine-tune/train_ssgvqa.py" \
    --ssgvqa-root "${SSGVQA_ROOT}" \
    --image-root "${IMAGE_ROOT}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --num-frames "${NUM_FRAMES}" \
    --max-length "${MAX_LENGTH}" \
    --prompt-mode "${PROMPT_MODE}" \
    --lora-r "${LORA_R}" \
    --lora-alpha "${LORA_ALPHA}" \
    --lora-dropout "${LORA_DROPOUT}" \
    --checkpoint-dir "${OUTPUT_DIR}" \
    --workers 4 \
    --seed 42 \
    --log-interval 50

# ============================================================================
# Check Exit Status
# ============================================================================
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "Training completed successfully!"
    echo "========================================"
    echo "Checkpoint saved to: ${OUTPUT_DIR}"
    echo ""
    echo "To evaluate on test set, run:"
    echo "  python fine-tune/eval_finetuned.py --checkpoint ${OUTPUT_DIR}/best_model.pth --prompt-mode ${PROMPT_MODE}"
else
    echo ""
    echo "========================================"
    echo "Training FAILED!"
    echo "========================================"
    exit 1
fi
