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
set -x
mkdir -p logs

source ~/.bashrc
module load miniforge
conda activate SurgViVQAEnv

export HF_HOME=/mnt/scratch/sc232jl/hf_home
export TORCH_HOME=/mnt/scratch/sc232jl/torch_cache
export TOKENIZERS_PARALLELISM=false

REPO_DIR=/mnt/scratch/sc232jl/SurgViVQA
cd ${REPO_DIR}

CKPT_PATH=/mnt/scratch/sc232jl/SurgViVQA/checkpoints/surgvivqa_gpt2_endovis_ckpt/best_model.pth
SSGVQA_ROOT=/mnt/scratch/sc232jl/datasets/SSGVQA/ssg-qa
IMAGE_ROOT=/mnt/scratch/sc232jl/datasets/CholecT45/data

LOG_FILE_SIMPLE=logs/ssgvqa_eval_${SLURM_JOB_ID}_simple.log
PRED_FILE_SIMPLE=logs/ssgvqa_predictions_${SLURM_JOB_ID}_simple.jsonl
LOG_FILE_CHOICES=logs/ssgvqa_eval_${SLURM_JOB_ID}_choices.log
PRED_FILE_CHOICES=logs/ssgvqa_predictions_${SLURM_JOB_ID}_choices.jsonl

python utils/eval_surgvivqa_ssgvqa.py \
  --model-path ${CKPT_PATH} \
  --ssgvqa-root ${SSGVQA_ROOT} \
  --image-root ${IMAGE_ROOT} \
  --videos VID02 VID22 VID43 VID60 VID74 \
  --log-file ${LOG_FILE_SIMPLE} \
  --predictions-file ${PRED_FILE_SIMPLE} \
  --batch-size 4 \
  --workers 4 \
  --num-frames 16 \
  --prompt-mode simple \
  --max-input-tokens 128 \
  --label-chunk-size 10 \
  --log-every-n 200

python utils/eval_surgvivqa_ssgvqa.py \
  --model-path ${CKPT_PATH} \
  --ssgvqa-root ${SSGVQA_ROOT} \
  --image-root ${IMAGE_ROOT} \
  --videos VID02 VID22 VID43 VID60 VID74 \
  --log-file ${LOG_FILE_CHOICES} \
  --predictions-file ${PRED_FILE_CHOICES} \
  --batch-size 4 \
  --workers 4 \
  --num-frames 16 \
  --prompt-mode choices \
  --max-input-tokens 1024 \
  --label-chunk-size 10 \
  --log-every-n 200
