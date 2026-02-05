#!/bin/bash
#SBATCH --job-name=deepseek_judge_surgvivqa
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6G
#SBATCH --open-mode=append
# NOTE: CPU-only job (no GPU needed for DeepSeek API calls)

# =============================================================================
# DeepSeek LLM-as-a-judge evaluation for SurgViVQA on SSGVQA
#
# This script runs the DeepSeek judge on model predictions to compute
# semantic correctness metrics for paper reporting.
#
# Prerequisites:
#   1. Set DEEPSEEK_API_KEY environment variable
#   2. Ensure predictions JSONL exists (from utils/eval_surgvivqa_ssgvqa.py)
#
# Usage:
#   # Local execution
#   export DEEPSEEK_API_KEY="your-api-key"
#   bash run_deepseek_judge.sh
#
#   # SLURM submission
#   sbatch run_deepseek_judge.sh
# =============================================================================

set -e

# -----------------------------------------------------------------------------
# Environment setup
# -----------------------------------------------------------------------------

# Load conda if available (cluster-specific)
if command -v module &> /dev/null; then
    module load miniforge 2>/dev/null || module load anaconda 2>/dev/null || true
fi

# Activate conda environment
CONDA_ENV="${CONDA_ENV:-surgvivqa}"
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "[INFO] Conda environment already active: $CONDA_DEFAULT_ENV"
elif command -v conda &> /dev/null; then
    echo "[INFO] Activating conda environment: $CONDA_ENV"
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV" || echo "[WARN] Could not activate conda env: $CONDA_ENV"
fi

# Locale settings
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export PYTHONIOENCODING=UTF-8
export PYTHONUTF8=1
export PYTHONUNBUFFERED=1

# -----------------------------------------------------------------------------
# DeepSeek API configuration
# -----------------------------------------------------------------------------

# API key (REQUIRED - set this before running)
if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
    echo "[ERROR] DEEPSEEK_API_KEY is not set."
    echo "Please export it before running:"
    echo "  export DEEPSEEK_API_KEY='your-api-key'"
    exit 1
fi

# Optional: Override base URL if using a different endpoint
export DEEPSEEK_BASE_URL="${DEEPSEEK_BASE_URL:-https://api.deepseek.com/v1}"
export DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-deepseek-chat}"

# -----------------------------------------------------------------------------
# Path configuration
# -----------------------------------------------------------------------------

# Project root (adjust if needed)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Input predictions file (output from utils/eval_surgvivqa_ssgvqa.py)
PRED_JSONL="${PRED_JSONL:-${PROJECT_ROOT}/outputs/surgvivqa_ssgvqa_predictions.jsonl}"

# Output paths
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/deepseek_eval}"
OUT_JUDGED="${OUT_JUDGED:-${OUTPUT_DIR}/surgvivqa_ssgvqa_judged.jsonl}"
OUT_METRICS="${OUT_METRICS:-${OUTPUT_DIR}/surgvivqa_ssgvqa_metrics.json}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# -----------------------------------------------------------------------------
# Execution settings
# -----------------------------------------------------------------------------

# Rate limiting (seconds between requests)
SLEEP_SECONDS="${SLEEP_SECONDS:-0.1}"

# Number of parallel workers (increase with caution to avoid rate limits)
NUM_WORKERS="${NUM_WORKERS:-1}"

# Resume from previous run (set to 1 to enable)
RESUME="${RESUME:-1}"

# Maximum samples to process (for debugging; leave empty for all)
MAX_SAMPLES="${MAX_SAMPLES:-}"

# Save raw judge responses (set to 1 to enable)
SAVE_RAW="${SAVE_RAW:-0}"

# -----------------------------------------------------------------------------
# Build command
# -----------------------------------------------------------------------------

CMD="python -u ${PROJECT_ROOT}/fine-tune/deepseek_eval/evaluate.py"
CMD+=" --predictions-jsonl \"$PRED_JSONL\""
CMD+=" --output-judged-jsonl \"$OUT_JUDGED\""
CMD+=" --output-metrics-json \"$OUT_METRICS\""
CMD+=" --base-url \"$DEEPSEEK_BASE_URL\""
CMD+=" --judge-model \"$DEEPSEEK_MODEL\""
CMD+=" --sleep-seconds $SLEEP_SECONDS"
CMD+=" --num-workers $NUM_WORKERS"
CMD+=" --save-every 100"

if [ "$RESUME" = "1" ]; then
    CMD+=" --resume"
fi

if [ -n "$MAX_SAMPLES" ]; then
    CMD+=" --max-samples $MAX_SAMPLES"
fi

if [ "$SAVE_RAW" = "1" ]; then
    CMD+=" --save-raw"
fi

# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------

echo "============================================================"
echo "DeepSeek Judge Evaluation for SurgViVQA SSGVQA"
echo "============================================================"
echo "Project root:     $PROJECT_ROOT"
echo "Input predictions: $PRED_JSONL"
echo "Output judged:     $OUT_JUDGED"
echo "Output metrics:    $OUT_METRICS"
echo "Judge model:       $DEEPSEEK_MODEL"
echo "API base URL:      $DEEPSEEK_BASE_URL"
echo "Sleep seconds:     $SLEEP_SECONDS"
echo "Num workers:       $NUM_WORKERS"
echo "Resume mode:       $RESUME"
echo "============================================================"

# Check input file exists
if [ ! -f "$PRED_JSONL" ]; then
    echo "[ERROR] Input predictions file not found: $PRED_JSONL"
    echo "Please run utils/eval_surgvivqa_ssgvqa.py first."
    exit 1
fi

# Execute (use srun if running under SLURM)
if [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "[INFO] Running under SLURM (job $SLURM_JOB_ID)"
    srun -u $CMD
else
    echo "[INFO] Running locally"
    eval $CMD
fi

echo ""
echo "[Done] Evaluation complete."
echo "  Judged JSONL: $OUT_JUDGED"
echo "  Metrics JSON: $OUT_METRICS"
