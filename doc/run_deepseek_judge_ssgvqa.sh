#!/bin/bash
#SBATCH --job-name=deepseek_judge_ssgvqa
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6G
#SBATCH --open-mode=append
# NOTE: CPU-only job. If your cluster requires a GPU partition, change --partition accordingly.

module load miniforge

CONDA_ENV=${CONDA_ENV:-llava_med}
conda activate "${CONDA_ENV}"

export LANG=en_GB.UTF-8
export LC_ALL=en_GB.UTF-8
export PYTHONIOENCODING=UTF-8
export PYTHONUTF8=1
export PYTHONUNBUFFERED=1

# Cache to scratch
SCRATCH_DIR="${SCRATCH_DIR:-/mnt/scratch/${USER}}"
export HF_HOME="${HF_HOME:-${SCRATCH_DIR}/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"

export DEEPSEEK_API_KEY="sk-c259af861c7949009559d53f10bc9f2e"

if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
  echo "DEEPSEEK_API_KEY is not set. Please export it before submitting the job."
  exit 1
fi

PROJECT_DIR="${PROJECT_DIR:-/mnt/scratch/sc232jl/LLaVA-Med}"
PRED_JSONL="${PRED_JSONL:-/mnt/scratch/sc232jl/LLaVA-Med/ssgvqa_llavamed_predictions.jsonl}"
OUT_JUDGED="${OUT_JUDGED:-/mnt/scratch/sc232jl/LLaVA-Med/ssgvqa_llavamed_predictions_deepseek_judged.jsonl}"
OUT_METRICS="${OUT_METRICS:-/mnt/scratch/sc232jl/LLaVA-Med/ssgvqa_llavamed_metrics_deepseek.json}"

cd "${PROJECT_DIR}" || { echo "Cannot cd to ${PROJECT_DIR}"; exit 1; }

srun -u python utils/eval_ssgvqa_deepseek_judge.py \
  --predictions-jsonl "${PRED_JSONL}" \
  --output-judged-jsonl "${OUT_JUDGED}" \
  --output-metrics-json "${OUT_METRICS}" \
  --judge-model "deepseek-chat" \
  --base-url "https://api.deepseek.com/v1" \
  --sleep-seconds 0.1 \
  --num-workers 1 \
  --resume
