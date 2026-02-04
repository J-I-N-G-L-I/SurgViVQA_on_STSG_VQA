#!/bin/bash
#
# Evaluation script for fine-tuned SurgViVQA on SSGVQA test set
#
# This script evaluates a fine-tuned checkpoint on the FIXED test split:
# VID02, VID22, VID43, VID60, VID74
#
# Metrics: Acc, mAP, mAR, mAF1, wF1 (same as SSGVQA-Net evaluation)
#
# Usage:
#   bash run_eval.sh [checkpoint_path] [prompt_mode]
#
# Examples:
#   bash run_eval.sh fine-tune/ckpt/best_model.pth simple
#   bash run_eval.sh fine-tune/ckpt/ssgvqa_lora_simple_20240101/best_model.pth simple
#

# ============================================================================
# SLURM Configuration (for Aire HPC)
# ============================================================================
#SBATCH --job-name=surgvivqa_ssgvqa_eval
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# ============================================================================
# Configuration
# ============================================================================
# Default checkpoint path (override with command line argument)
CHECKPOINT="${1:-fine-tune/ckpt/best_model.pth}"
PROMPT_MODE="${2:-simple}"

# Data paths (adjust for your HPC environment)
SSGVQA_ROOT="/mnt/scratch/sc232jl/datasets/SSGVQA/ssg-qa/"
IMAGE_ROOT="/mnt/scratch/sc232jl/datasets/CholecT45/data"

# Create logs directory
mkdir -p logs

# ============================================================================
# Environment Setup
# ============================================================================
echo "========================================"
echo "SurgViVQA Evaluation on SSGVQA Test Set"
echo "========================================"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "========================================"
echo ""
echo "Configuration:"
echo "  Checkpoint: ${CHECKPOINT}"
echo "  Prompt Mode: ${PROMPT_MODE}"
echo "  SSGVQA Root: ${SSGVQA_ROOT}"
echo "  Image Root: ${IMAGE_ROOT}"
echo "  Test Videos: VID02, VID22, VID43, VID60, VID74"
echo ""

# Check if checkpoint exists
if [ ! -f "${CHECKPOINT}" ]; then
    echo "ERROR: Checkpoint not found: ${CHECKPOINT}"
    exit 1
fi

# Change to project root
cd "$(dirname "$0")/.." || exit 1
echo "Working directory: $(pwd)"

# ============================================================================
# Run Evaluation
# ============================================================================
python fine-tune/eval_finetuned.py \
    --checkpoint "${CHECKPOINT}" \
    --ssgvqa-root "${SSGVQA_ROOT}" \
    --image-root "${IMAGE_ROOT}" \
    --prompt-mode "${PROMPT_MODE}" \
    --batch-size 4 \
    --num-frames 16 \
    --max-input-tokens 256 \
    --workers 4 \
    --seed 42

# ============================================================================
# Check Exit Status
# ============================================================================
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "Evaluation completed successfully!"
    echo "========================================"
    
    # Print summary from log if available
    OUTPUT_DIR=$(dirname "${CHECKPOINT}")
    LOG_FILE="${OUTPUT_DIR}/eval_test.log"
    if [ -f "${LOG_FILE}" ]; then
        echo ""
        echo "Summary (from log):"
        grep -E "(acc|mAP|mAR|mAF1|wF1|Summary)" "${LOG_FILE}" | tail -5
    fi
else
    echo ""
    echo "========================================"
    echo "Evaluation FAILED!"
    echo "========================================"
    exit 1
fi
