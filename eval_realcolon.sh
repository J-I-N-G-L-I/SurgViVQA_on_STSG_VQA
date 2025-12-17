#!/bin/bash
#SBATCH --job-name=surgvivqa_eval_realcolon
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/scratch/sc232jl/SurgViVQA/logs/%x-%j.out
#SBATCH --error=/mnt/scratch/sc232jl/SurgViVQA/logs/%x-%j.err

set -eo pipefail

REPO_DIR=/scratch/sc232jl/SurgViVQA

# export HF_HOME=/scratch/sc232jl/.cache/huggingface
# export TRANSFORMERS_CACHE=/scratch/sc232jl/.cache/huggingface
# export TORCH_HOME=/scratch/sc232jl/.cache/torch
# export PIP_CACHE_DIR=/scratch/sc232jl/.cache/pip
# export TMPDIR=/scratch/sc232jl/.cache/tmp

# mkdir -p "$HF_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR" "$TMPDIR" /scratch/sc232jl/logs

source ~/.bashrc
conda activate SurgViVQAEnv

cd "$REPO_DIR"

echo "=== Job info ==="
date
hostname
nvidia-smi

#（可选）限制线程，避免 CPU 过度抢占
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# ===== 4) Run training =====
python -m train \
  --dataset realcolon \
  --epochs 60 \
  --batch_size 16 \
  --lr 0.0000002 \
  --seq_length 64 \
  --workers 8
