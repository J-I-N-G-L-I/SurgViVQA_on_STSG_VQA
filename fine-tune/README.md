# SurgViVQA Fine-tuning on SSGVQA

This folder contains the complete fine-tuning pipeline for training SurgViVQA on the SSGVQA dataset.

## Overview

The pipeline enables supervised instruction fine-tuning (SFT) of SurgViVQA on SSGVQA's 52-class closed-set VQA task. It supports both LoRA (PEFT) fine-tuning (default, lower VRAM) and full fine-tuning.

## Directory Structure

```
fine-tune/
├── README.md                 # This file
├── labels.py                 # 52-class label vocabulary (matches evaluation)
├── dataset.py                # SSGVQA dataset loader for training
├── trainer.py                # Training utilities (SFT loss, checkpoint saving)
├── train_ssgvqa.py           # Main fine-tuning script
├── eval_finetuned.py         # Evaluation wrapper script
├── run_finetune.sh           # HPC training script (SLURM compatible)
├── run_eval.sh               # HPC evaluation script (SLURM compatible)
├── splits/
│   ├── train_videos.json     # 35 training video IDs
│   ├── val_videos.json       # 5 validation video IDs
│   └── test_videos.json      # 5 test video IDs (FIXED, never for training)
└── ckpt/                     # Checkpoints saved here (created during training)
```

## Data Splits

Following SSGVQA-Net conventions:
- **Train**: 35 videos (defined in `splits/train_videos.json`)
- **Val**: 5 videos (defined in `splits/val_videos.json`)
- **Test**: 5 videos (FIXED): `VID02, VID22, VID43, VID60, VID74`

**IMPORTANT**: The test split must NEVER be used for training. This ensures fair comparison with SSGVQA-Net.

## Label Vocabulary

The 52-class label space includes:
- Numeric (0-10): counting answers
- Boolean (False, True): existence/yes-no answers
- Anatomical structures: cystic_artery, cystic_duct, gallbladder, liver, etc.
- Actions: aspirate, clip, coagulate, cut, dissect, grasp, etc.
- Instruments: bipolar, clipper, grasper, hook, irrigator, scissors, etc.
- Colors: blue, brown, red, silver, white, yellow

See `labels.py` for the complete ordered list.

## Quick Start

### 1. Fine-tuning (LoRA, recommended)

```bash
# On Aire HPC with SLURM
sbatch fine-tune/run_finetune.sh

# Or run directly
python fine-tune/train_ssgvqa.py \
    --ssgvqa-root /mnt/scratch/sc232jl/datasets/SSGVQA/ssg-qa/ \
    --image-root /mnt/scratch/sc232jl/datasets/CholecT45/data \
    --epochs 30 \
    --batch-size 4 \
    --lr 2e-5 \
    --prompt-mode simple
```

### 2. Full Fine-tuning (requires more VRAM)

```bash
python fine-tune/train_ssgvqa.py \
    --ssgvqa-root /mnt/scratch/sc232jl/datasets/SSGVQA/ssg-qa/ \
    --image-root /mnt/scratch/sc232jl/datasets/CholecT45/data \
    --full-finetune \
    --epochs 30 \
    --batch-size 2 \
    --lr 1e-6 \
    --prompt-mode simple
```

### 3. Evaluation on Test Set

```bash
# After training
python fine-tune/eval_finetuned.py \
    --checkpoint fine-tune/ckpt/best_model.pth \
    --prompt-mode simple

# Or use the shell script
bash fine-tune/run_eval.sh fine-tune/ckpt/best_model.pth simple
```

## Training Details

### Prompt Format

The training uses prompts compatible with the existing evaluation script:

**Simple mode** (`--prompt-mode simple`):
```
Question: What instrument is grasping the cystic duct?
Answer:
```

**Choices mode** (`--prompt-mode choices`):
```
Question: What instrument is grasping the cystic duct?
Candidates: 0, 1, 10, 2, 3, 4, 5, 6, 7, 8, 9, False, True, abdominal_wall_cavity, adhesion, anatomy, aspirate, bipolar, blood_vessel, blue, brown, clip, clipper, coagulate, cut, cystic_artery, cystic_duct, cystic_pedicle, cystic_plate, dissect, fluid, gallbladder, grasp, grasper, gut, hook, instrument, irrigate, irrigator, liver, omentum, pack, peritoneum, red, retract, scissors, silver, specimen_bag, specimenbag, white, yellow
Answer:
```

### Loss Computation

- Training uses SFT with prompt masking
- Loss is computed ONLY on answer tokens (prompt tokens masked with -100)
- Target format: ` LABEL` (leading space for GPT-2 tokenization compatibility)
- This matches the evaluation script's teacher-forcing log-prob scoring

### LoRA Configuration (default)

- Rank (r): 8
- Alpha: 32
- Dropout: 0.1
- Target modules: `c_attn`, `c_proj` (GPT-2 attention)
- Trainable parameters: ~0.3% of total

## Metrics

Evaluation uses the same metrics as SSGVQA-Net:
- **Acc**: Overall accuracy
- **mAP**: Mean Average Precision (macro)
- **mAR**: Mean Average Recall (macro)
- **mAF1**: Mean Average F1 (macro)
- **wF1**: Weighted F1

## Hyperparameter Recommendations

| Setting | LoRA (Default) | Full Fine-tune |
|---------|----------------|----------------|
| Batch Size | 4-8 | 2-4 |
| Learning Rate | 2e-5 | 1e-6 |
| Epochs | 30-50 | 20-30 |
| VRAM | ~8-12 GB | ~20-24 GB |

## Troubleshooting

### Out of Memory
- Reduce `--batch-size`
- Use LoRA (default) instead of `--full-finetune`
- Reduce `--num-frames`
- Enable gradient checkpointing (if implemented)

### Poor Convergence
- Try different learning rates (1e-5 to 5e-5 for LoRA)
- Increase `--lora-r` (16 or 32)
- Try `--prompt-mode choices` for more context
- Check that prompt mode matches between training and evaluation

### Label Mismatch
- Ensure the 52-class label vocabulary in `labels.py` matches the evaluation script
- Labels must be in the EXACT same order

## Files Reference

| File | Description |
|------|-------------|
| `labels.py` | 52-class label vocabulary and mapping utilities |
| `dataset.py` | `SSGVQADataset` class for loading training data |
| `trainer.py` | `train_epoch`, `validate_epoch`, checkpoint utilities |
| `train_ssgvqa.py` | Main training script with CLI arguments |
| `eval_finetuned.py` | Wrapper for existing evaluation script |
| `run_finetune.sh` | SLURM-compatible training script |
| `run_eval.sh` | SLURM-compatible evaluation script |
