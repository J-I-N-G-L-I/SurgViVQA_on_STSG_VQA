#!/bin/bash
#SBATCH --job-name=stsg_score_jsonl
#SBATCH --output=/mnt/scratch/sc232jl/SurgViVQA/stsg_eval_outputs_full/score_%j.out
#SBATCH --error=/mnt/scratch/sc232jl/SurgViVQA/stsg_eval_outputs_full/score_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -eo pipefail


cd /mnt/scratch/sc232jl/SurgViVQA

export PYTHONPATH="$PWD:$PYTHONPATH"

conda activate SurgViVQAEnv

PRED_JSONL="/mnt/scratch/sc232jl/SurgViVQA/stsg_eval_outputs_full/predictions.jsonl"
VOCABS_JSON="/users/sc232jl/SSGVQANet_hybrid_training_pro_with_rationale/utils/vocabs_train.json"
OUT_JSON="/mnt/scratch/sc232jl/SurgViVQA/stsg_eval_outputs_full/report.json"

python utils/metrics_from_jsonl_result.py \
  --pred_jsonl "$PRED_JSONL" \
  --vocabs_json "$VOCABS_JSON" \
  --out_json "$OUT_JSON"

echo "[Done] Report saved to: $OUT_JSON"
