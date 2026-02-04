#!/bin/bash
#
# SSGVQA Fine-tuning script for SurgViVQA on Aire HPC (FIXED VERSION)
#
# Key fixes:
#   - LoRA-only by default: Only LoRA adapters trainable (~0.15% params)
#   - Step-based training: --max-steps for predictable HPC job times
#   - AMP support: --fp16 or --bf16 for faster training
#   - Realistic defaults for 12-hour SLURM job
#
# Usage:
#   cd /path/to/SurgViVQA_on_STSG_VQA  # IMPORTANT: must be in project root
#   sbatch fine-tune/run_finetune.sh
#
# Test split (NEVER for training): VID02, VID22, VID43, VID60, VID74
#
# After training completes, evaluate with:
#   bash fine-tune/run_eval.sh
#

# ============================================================================
# SLURM Configuration (for Aire HPC)
# ============================================================================
#SBATCH --job-name=ssgvqa_lora_ft
#SBATCH --output=logs/finetune_%j.out
#SBATCH --error=logs/finetune_%j.err
#SBATCH --time=12:00:00
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
echo "SurgViVQA LoRA Fine-tuning on SSGVQA (FIXED)"
echo "========================================"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "========================================"

# ============================================================================
# Resolve project root directory
# IMPORTANT: Run `sbatch fine-tune/run_finetune.sh` from project root!
# ============================================================================
if [ -n "${SLURM_SUBMIT_DIR}" ]; then
    PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
fi

cd "${PROJECT_ROOT}" || { echo "ERROR: Cannot cd to ${PROJECT_ROOT}"; exit 1; }
echo "Project root: ${PROJECT_ROOT}"
echo "Working directory: $(pwd)"

# Create logs directory
mkdir -p logs

# Activate conda environment
module load miniforge 2>/dev/null || true
conda activate SurgViVQAEnv

# ============================================================================
# Data Paths
# ============================================================================
SSGVQA_ROOT="/mnt/scratch/sc232jl/datasets/SSGVQA/ssg-qa/"
IMAGE_ROOT="/mnt/scratch/sc232jl/datasets/CholecT45/data"

# ============================================================================
# Training Configuration (OPTIMIZED for 12-hour SLURM job)
# ============================================================================
# LoRA-only fine-tuning (default): ~0.15% params trainable
# With 723K samples and BS=8, 1 epoch = ~90K steps
# For 12h job: ~30K steps achievable (about 1/3 epoch)

MAX_STEPS=30000           # Step-based training for predictable time
EVAL_EVERY=2000           # Validate every N steps
SAVE_EVERY=5000           # Checkpoint every N steps
VAL_MAX_SAMPLES=20000     # Limit val set for faster eval

BATCH_SIZE=8              # Higher batch size for better throughput
LR=5e-4                   # Higher LR for LoRA-only (fewer params to update)
NUM_FRAMES=16
MAX_LENGTH=128
PROMPT_MODE="simple"

# LoRA configuration (only LoRA weights trainable by default)
LORA_R=8
LORA_ALPHA=32
LORA_DROPOUT=0.1

# AMP for faster training (use --bf16 if on Ampere GPU, else --fp16)
AMP_FLAG="--fp16"

# ============================================================================
# Freeze Control (CRITICAL for LoRA-only)
# ============================================================================
# Default: VideoMAE frozen, BLIP frozen, GPT-2 base frozen, only LoRA trainable
# To train BLIP: add --train-blip
# To train GPT-2 base: add --train-gpt2-full
# To train last N layers of BLIP: add --unfreeze-blip-last-n N

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
echo "  Max Steps: ${MAX_STEPS}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  Learning Rate: ${LR}"
echo "  Num Frames: ${NUM_FRAMES}"
echo "  Prompt Mode: ${PROMPT_MODE}"
echo "  AMP: ${AMP_FLAG}"
echo "  Eval Every: ${EVAL_EVERY} steps"
echo "  Save Every: ${SAVE_EVERY} steps"
echo "  Output Dir: ${OUTPUT_DIR}"
echo ""
echo "Freeze settings (LoRA-only by default):"
echo "  - VideoMAE: frozen"
echo "  - BLIP text encoder: frozen (add --train-blip to unfreeze)"
echo "  - GPT-2 base: frozen (add --train-gpt2-full to unfreeze)"
echo "  - LoRA adapters: TRAINABLE"
echo ""

# ============================================================================
# Run Training
# ============================================================================
python "${PROJECT_ROOT}/fine-tune/train_ssgvqa.py" \
    --ssgvqa-root "${SSGVQA_ROOT}" \
    --image-root "${IMAGE_ROOT}" \
    --max-steps "${MAX_STEPS}" \
    --eval-every-steps "${EVAL_EVERY}" \
    --save-every-steps "${SAVE_EVERY}" \
    --val-max-samples "${VAL_MAX_SAMPLES}" \
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
    --log-interval 100 \
    ${AMP_FLAG}

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
