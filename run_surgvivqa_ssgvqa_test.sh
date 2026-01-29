#!/bin/bash
#SBATCH --job-name=surgvivqa_ssgvqa_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e
mkdir -p logs

source ~/.bashrc
module load miniforge
conda activate SurgViVQAEnv

# ---- Cache locations (HuggingFace/torch) ----
export HF_HOME=/mnt/scratch/sc232jl/hf_home
export TRANSFORMERS_CACHE=/mnt/scratch/sc232jl/hf_cache
export TORCH_HOME=/mnt/scratch/sc232jl/torch_cache
export TOKENIZERS_PARALLELISM=false

# ---- Project directory ----
REPO_DIR=/mnt/scratch/sc232jl/SurgViVQA_on_STSG_VQA
cd ${REPO_DIR}

# ---- Paths (edit as needed) ----
CKPT_PATH=/mnt/scratch/sc232jl/SurgViVQA/checkpoints/surgvivqa_ssgvqa/best_model.pth
SSGVQA_ROOT=/mnt/scratch/sc232jl/datasets/SSGVQA/ssg-qa
IMAGE_ROOT=/mnt/scratch/sc232jl/datasets/CholecT45/data

LOG_FILE=logs/ssgvqa_eval_${SLURM_JOB_ID}.log
PRED_FILE=logs/ssgvqa_predictions_${SLURM_JOB_ID}.jsonl
METRICS_FILE=logs/ssgvqa_metrics_${SLURM_JOB_ID}.json

python utils/test_surgvivqa_ssgvqa.py \
  --model-path ${CKPT_PATH} \
  --ssgvqa-root ${SSGVQA_ROOT} \
  --image-root ${IMAGE_ROOT} \
  --videos VID02 VID22 VID43 VID60 VID74 \
  --log-file ${LOG_FILE} \
  --predictions-file ${PRED_FILE} \
  --save-metrics ${METRICS_FILE} \
  --batch-size 4 \
  --workers 4 \
  --num-frames 1 \
  --max-prompt-len 128 \
  --max-new-tokens 8 \
  --decode-mode closed \
  --strict-answer-format