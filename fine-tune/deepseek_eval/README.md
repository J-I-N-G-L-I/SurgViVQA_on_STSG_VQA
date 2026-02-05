# DeepSeek LLM-as-a-Judge Evaluation for SurgViVQA SSGVQA

This module provides a DeepSeek-based evaluation pipeline for SurgViVQA predictions on SSGVQA, designed for reliable metrics reporting in academic papers.

## Why LLM-as-a-Judge?

- **Multi-label GT handling**: SSGVQA ground-truth answers can be comma-separated (e.g., `"gallbladder, cystic_plate"`). The judge determines if the prediction matches ANY valid GT label.
- **Semantic equivalence**: Rule-based matching may miss synonyms or semantic equivalents (e.g., `specimen_bag` ≈ `specimenbag`). The LLM judge handles these cases.
- **Reproducible metrics**: Provides consistent evaluation across different prediction formats.

## File Structure

```
fine-tune/deepseek_eval/
├── __init__.py           # Package init
├── judge.py              # DeepSeek API client with retry/rate-limiting
├── metrics.py            # Closed-set metric computation (Acc, mAP, mAR, mAF1, wF1)
├── evaluate.py           # Main evaluation script
├── run_deepseek_judge.sh # Shell script for local/SLURM execution
└── README.md             # This file
```

## Prerequisites

1. **DeepSeek API Key**: Obtain from [DeepSeek Platform](https://platform.deepseek.com/)
2. **Predictions JSONL**: Output from `utils/eval_surgvivqa_ssgvqa.py`

## Quick Start

### 1. Set API Key

```bash
export DEEPSEEK_API_KEY="your-api-key-here"
```

### 2. Run Evaluation

```bash
# Basic usage
python fine-tune/deepseek_eval/evaluate.py \
  --predictions-jsonl outputs/surgvivqa_ssgvqa_predictions.jsonl \
  --output-judged-jsonl outputs/deepseek_eval/judged.jsonl \
  --output-metrics-json outputs/deepseek_eval/metrics.json

# With resume support (for long runs)
python fine-tune/deepseek_eval/evaluate.py \
  --predictions-jsonl outputs/surgvivqa_ssgvqa_predictions.jsonl \
  --output-judged-jsonl outputs/deepseek_eval/judged.jsonl \
  --output-metrics-json outputs/deepseek_eval/metrics.json \
  --resume
```

### 3. Using Shell Script

```bash
# Set paths and run
export DEEPSEEK_API_KEY="your-api-key"
export PRED_JSONL="outputs/surgvivqa_ssgvqa_predictions.jsonl"
bash fine-tune/deepseek_eval/run_deepseek_judge.sh
```

## Input Format

Each line in the predictions JSONL (from `utils/eval_surgvivqa_ssgvqa.py`):

```json
{
  "video": "VID01",
  "frame_id": "000102",
  "image_path": "/path/to/frame.png",
  "question": "What is the grasper doing?",
  "gt_answer": "retract, grasp",
  "pred_label": "retract",
  "top5": [{"label": "retract", "score": -2.5}, ...],
  "prompt_mode": "simple"
}
```

## Output Format

### Judged JSONL

Each line includes original fields plus judge verdicts:

```json
{
  "video": "VID01",
  "frame_id": "000102",
  "question": "What is the grasper doing?",
  "gt_answer": "retract, grasp",
  "pred_label": "retract",
  "mapped_label": "retract",
  "is_correct": true,
  "confidence": 0.95,
  "note": "Prediction matches first GT label exactly."
}
```

### Metrics JSON

```json
{
  "overall": {
    "acc": 0.75,
    "mAP": 0.68,
    "mAR": 0.65,
    "mAF1": 0.66,
    "wF1": 0.72,
    "anymatch_acc": 0.78,
    "anymatch_correct": 1560,
    "anymatch_total": 2000,
    "judge_acc": 0.80,
    "judge_correct_count": 1600,
    "judge_total": 2000,
    "num_samples": 2000,
    "num_eval_samples": 1950
  },
  "per_video": {
    "VID01": {"n_samples": 200, "acc": 0.76, ...},
    "VID02": {"n_samples": 180, "acc": 0.74, ...}
  },
  "label_vocab": ["0", "1", ..., "yellow"]
}
```

## Metrics Explanation

| Metric | Description |
|--------|-------------|
| `acc` | Standard accuracy (first GT label vs prediction) |
| `mAP` | Mean Average Precision across all 51 classes |
| `mAR` | Mean Average Recall across all 51 classes |
| `mAF1` | Mean Average F1 across all 51 classes |
| `wF1` | Weighted F1 (weighted by class support) |
| `anymatch_acc` | Accuracy considering multi-label GT (prediction matches ANY GT label) |
| `judge_acc` | LLM judge semantic correctness rate |

## Command-Line Options

```
--predictions-jsonl     Input predictions JSONL (required)
--output-judged-jsonl   Output judged JSONL (required)
--output-metrics-json   Output metrics JSON (required)
--base-url              DeepSeek API base URL (default: https://api.deepseek.com/v1)
--judge-model           DeepSeek model name (default: deepseek-chat)
--request-timeout       API timeout in seconds (default: 60)
--max-retries           Max retry attempts (default: 5)
--sleep-seconds         Min seconds between requests (default: 0.1)
--max-samples           Process only first N samples (for debugging)
--num-workers           Number of parallel workers (default: 1)
--save-every            Flush output every N records (default: 100)
--resume                Resume from existing judged JSONL
--metrics-only          Skip judging, only compute metrics
--save-raw              Save raw judge responses
```

## SLURM Submission

```bash
# Edit run_deepseek_judge.sh to set paths
sbatch fine-tune/deepseek_eval/run_deepseek_judge.sh
```

## Metrics-Only Mode

If you already have a judged JSONL and just want to recompute metrics:

```bash
python fine-tune/deepseek_eval/evaluate.py \
  --predictions-jsonl outputs/predictions.jsonl \
  --output-judged-jsonl outputs/judged.jsonl \
  --output-metrics-json outputs/metrics_recomputed.json \
  --metrics-only
```

## Troubleshooting

### API Rate Limits

If you encounter rate limit errors:
1. Increase `--sleep-seconds` (e.g., 0.5 or 1.0)
2. Keep `--num-workers 1`
3. Use `--resume` to continue after interruption

### Missing DEEPSEEK_API_KEY

```bash
export DEEPSEEK_API_KEY="sk-your-key-here"
```

### Large Prediction Files

For files with 10k+ samples:
1. Use `--resume` to enable checkpointing
2. Set `--save-every 50` for frequent saves
3. Run with `--max-samples 100` first to test

## Citation

If you use this evaluation pipeline, please cite SurgViVQA and acknowledge the DeepSeek API.
