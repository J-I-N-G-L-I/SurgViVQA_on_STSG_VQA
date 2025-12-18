#!/bin/bash
#SBATCH --job-name=surgvivqa_stsg_eval
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
conda activate SurgViVQAEnv

cd /mnt/scratch/sc232jl/SurgViVQA

export HF_HOME=/scratch/sc232jl/hf_home
export TRANSFORMERS_CACHE=/scratch/sc232jl/hf_cache
export TORCH_HOME=/scratch/sc232jl/torch_cache
export TOKENIZERS_PARALLELISM=false

# >>> IMPORTANT <<<
# Save your uploaded vocab JSON (pasted.txt content) to a real .json file, e.g.:
# /mnt/scratch/sc232jl/SurgViVQA/vocabs/stsg_text_vocab.json
VOCAB_JSON=/users/sc232jl/SSGVQANet_hybrid_training_pro_with_rationale/utils/vocabs_train.json

python evaluation_stsg.py \
  --video_ids VID22 VID74 VID60 VID02 VID43 \
  --stsg_qa_root /mnt/scratch/sc232jl/datasets/SSGVQA/STSG_QA_Pro_8_Classes \
  --frame_root /mnt/scratch/sc232jl/datasets/CholecT45/data \
  --checkpoint /mnt/scratch/sc232jl/SurgViVQA/checkpoints/surgvivqa_gpt2_endovis_ckpt/best_model.pth \
  --batch_size 2 --workers 6 \
  --num_frames 16 \
  --max_prompt_len 512 --max_new_tokens 16 \
  --decode_mode closed \
  --text_vocab_json ${VOCAB_JSON} \
  --count_max 20 \
  --closed_text_topk 100 \
  --save_dir stsg_eval_outputs_full

# -----------------------------
# When you want to test WITH rationale later, uncomment:
#   --include_rationale --rationale_key auto \
# and increase max_prompt_len / max_new_tokens, e.g.:
#   --max_prompt_len 512 --max_new_tokens 24
# -----------------------------
