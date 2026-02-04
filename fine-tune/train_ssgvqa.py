"""
SurgViVQA Fine-tuning on SSGVQA Dataset

This script fine-tunes the SurgViVQA model on the SSGVQA dataset for closed-set VQA.
It supports both LoRA (PEFT) fine-tuning (default, lower VRAM) and full fine-tuning.

Usage:
    # LoRA fine-tuning (default, recommended for single GPU)
    python train_ssgvqa.py \
        --ssgvqa-root /mnt/scratch/sc232jl/datasets/SSGVQA/ssg-qa/ \
        --image-root /mnt/scratch/sc232jl/datasets/CholecT45/data \
        --epochs 30 --batch-size 4 --lr 2e-5

    # Full fine-tuning (requires more VRAM)
    python train_ssgvqa.py \
        --ssgvqa-root /mnt/scratch/sc232jl/datasets/SSGVQA/ssg-qa/ \
        --image-root /mnt/scratch/sc232jl/datasets/CholecT45/data \
        --full-finetune --epochs 30 --batch-size 2 --lr 1e-6

Key Features:
- Closed-set 52-class label vocabulary (matches SSGVQA-Net evaluation)
- SFT training with prompt masking (loss only on answer tokens)
- LoRA/PEFT fine-tuning by default (lower VRAM usage)
- Optional full fine-tuning via --full-finetune flag
- Prompt modes: simple (default) or choices (includes all labels)
- Checkpoint saving to fine-tune/ckpt/
- Compatible with existing evaluation script (utils/eval_surgvivqa_ssgvqa.py)

Dataset:
- Train split: 35 videos (defined in fine-tune/splits/train_videos.json)
- Val split: 5 videos (defined in fine-tune/splits/val_videos.json)
- Test split (NEVER for training): VID02, VID22, VID43, VID60, VID74

HPC Reproducibility:
- Run on Aire with 1 GPU
- Set --seed for reproducibility
- All hyperparameters configurable via CLI

Copyright (c) 2024. All Rights Reserved.
"""

import os
import sys
import argparse
import logging
import random
import json
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType

# ============================================================================
# Add project root to path for imports
# ============================================================================
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Import from fine-tune folder
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from labels import SSGVQA_LABELS, LABEL2IDX, NORM_LABEL_MAP, NUM_CLASSES
from dataset import SSGVQADataset, collate_ssgvqa_train, load_split_videos
from trainer import (
    train_epoch, validate_epoch, save_checkpoint, load_checkpoint,
    adjust_learning_rate
)

# Import model from project
from models.model import SurgViVQA

import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Configuration
# ============================================================================

# HF cache directories (adjust for your HPC environment)
BASE_CACHE = "/scratch/sc232jl/.cache/huggingface"
os.environ["HF_HOME"] = BASE_CACHE
os.environ["TRANSFORMERS_CACHE"] = os.path.join(BASE_CACHE, "transformers")
os.environ["TORCH_HOME"] = os.path.join(BASE_CACHE, "torch")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def seed_everything(seed: int = 42) -> None:
    """Set random seed for reproducibility across all libraries."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    logging.info(f"Random seed set to {seed}")


def setup_logging(log_file: str, debug: bool = False) -> None:
    """Setup logging to file and console."""
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    
    level = logging.DEBUG if debug else logging.INFO
    
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def get_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune SurgViVQA on SSGVQA dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # -------------------------------------------------------------------------
    # Data paths
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--ssgvqa-root", type=str,
        default="/mnt/scratch/sc232jl/datasets/SSGVQA/ssg-qa/",
        help="Root directory of SSGVQA QA txt files"
    )
    parser.add_argument(
        "--image-root", type=str,
        default="/mnt/scratch/sc232jl/datasets/CholecT45/data",
        help="Root directory of CholecT45 frame images"
    )
    parser.add_argument(
        "--splits-dir", type=str,
        default=None,
        help="Directory containing split JSON files (default: fine-tune/splits/)"
    )
    
    # -------------------------------------------------------------------------
    # Model configuration
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--pretrained-ckpt", type=str,
        default=None,
        help="Path to pretrained SurgViVQA checkpoint to initialize from"
    )
    parser.add_argument(
        "--full-finetune", action="store_true",
        help="Full fine-tuning instead of LoRA (requires more VRAM)"
    )
    parser.add_argument(
        "--lora-r", type=int, default=8,
        help="LoRA rank (r parameter)"
    )
    parser.add_argument(
        "--lora-alpha", type=int, default=32,
        help="LoRA alpha parameter"
    )
    parser.add_argument(
        "--lora-dropout", type=float, default=0.1,
        help="LoRA dropout rate"
    )
    parser.add_argument(
        "--num-frames", type=int, default=16,
        help="Number of frames for video input (frames repeated for static images)"
    )
    
    # -------------------------------------------------------------------------
    # Training configuration
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--epochs", type=int, default=30,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Batch size per GPU"
    )
    parser.add_argument(
        "--lr", type=float, default=2e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--max-length", type=int, default=128,
        help="Max sequence length for tokenization"
    )
    parser.add_argument(
        "--gradient-accumulation-steps", type=int, default=1,
        help="Number of gradient accumulation steps"
    )
    parser.add_argument(
        "--max-grad-norm", type=float, default=1.0,
        help="Max gradient norm for clipping"
    )
    parser.add_argument(
        "--lr-decay-patience", type=int, default=5,
        help="Epochs without improvement before LR decay"
    )
    parser.add_argument(
        "--lr-decay-factor", type=float, default=0.8,
        help="LR decay factor"
    )
    parser.add_argument(
        "--early-stop-patience", type=int, default=15,
        help="Epochs without improvement before early stopping"
    )
    
    # -------------------------------------------------------------------------
    # Prompt configuration
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--prompt-mode", type=str, default="simple",
        choices=["simple", "choices"],
        help="Prompt mode: 'simple' (short prefix) or 'choices' (includes all 52 labels)"
    )
    
    # -------------------------------------------------------------------------
    # Output configuration
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--checkpoint-dir", type=str,
        default=None,
        help="Directory to save checkpoints (default: fine-tune/ckpt/)"
    )
    parser.add_argument(
        "--log-file", type=str,
        default=None,
        help="Log file path (default: checkpoint_dir/train.log)"
    )
    
    # -------------------------------------------------------------------------
    # Other options
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of dataloader workers"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--log-interval", type=int, default=50,
        help="Log every N batches"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Debug mode (use fewer samples)"
    )
    parser.add_argument(
        "--max-train-samples", type=int, default=None,
        help="Max training samples (for debugging)"
    )
    parser.add_argument(
        "--max-val-samples", type=int, default=None,
        help="Max validation samples (for debugging)"
    )
    
    args = parser.parse_args()
    
    # Set default paths
    fine_tune_dir = os.path.dirname(os.path.abspath(__file__))
    
    if args.splits_dir is None:
        args.splits_dir = os.path.join(fine_tune_dir, "splits")
    
    if args.checkpoint_dir is None:
        args.checkpoint_dir = os.path.join(fine_tune_dir, "ckpt")
    
    if args.log_file is None:
        args.log_file = os.path.join(args.checkpoint_dir, "train.log")
    
    # Debug mode defaults
    if args.debug:
        args.max_train_samples = args.max_train_samples or 100
        args.max_val_samples = args.max_val_samples or 50
    
    return args


def create_model(
    tokenizer,
    device: torch.device,
    num_frames: int,
    full_finetune: bool = False,
    lora_r: int = 8,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
) -> SurgViVQA:
    """
    Create SurgViVQA model with optional LoRA configuration.
    
    Args:
        tokenizer: GPT-2 tokenizer
        device: Target device
        num_frames: Number of video frames
        full_finetune: If True, no LoRA (full fine-tuning)
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
    
    Returns:
        SurgViVQA model
    """
    # Load base decoder model
    decoder_model = AutoModelForCausalLM.from_pretrained("gpt2")
    
    # LoRA configuration (only applied if not full fine-tuning)
    peft_config = None
    if not full_finetune:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,  # Training mode
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["c_attn", "c_proj"],  # GPT-2 attention modules
        )
        logging.info(f"Using LoRA fine-tuning: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
    else:
        logging.info("Using full fine-tuning (no LoRA)")
    
    # Create model
    model = SurgViVQA(
        device=device,
        tokenizer=tokenizer,
        decoder_model=decoder_model,
        peft_config=peft_config,
        num_frames=num_frames,
    )
    
    return model


def main():
    """Main training function."""
    args = get_args()
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Setup logging
    setup_logging(args.log_file, debug=args.debug)
    
    # Log configuration
    logging.info("=" * 80)
    logging.info("SurgViVQA Fine-tuning on SSGVQA")
    logging.info("=" * 80)
    logging.info(f"Configuration:")
    for key, value in vars(args).items():
        logging.info(f"  {key}: {value}")
    logging.info("=" * 80)
    
    # Save config
    config_path = os.path.join(args.checkpoint_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    logging.info(f"Config saved to {config_path}")
    
    # Set seed
    seed_everything(args.seed)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logging.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # =========================================================================
    # Tokenizer
    # =========================================================================
    logging.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logging.info(f"Tokenizer vocab size: {len(tokenizer)}")
    
    # =========================================================================
    # Model
    # =========================================================================
    logging.info("Creating model...")
    model = create_model(
        tokenizer=tokenizer,
        device=device,
        num_frames=args.num_frames,
        full_finetune=args.full_finetune,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    model = model.to(device)
    
    # Load pretrained checkpoint if provided
    if args.pretrained_ckpt:
        logging.info(f"Loading pretrained checkpoint: {args.pretrained_ckpt}")
        load_checkpoint(model, args.pretrained_ckpt, device, strict=False)
    
    # Model stats
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total parameters: {total_params:,}")
    logging.info(f"Trainable parameters: {trainable_params:,}")
    logging.info(f"Trainable ratio: {100 * trainable_params / total_params:.2f}%")
    
    # =========================================================================
    # Dataset and DataLoaders
    # =========================================================================
    logging.info("Loading datasets...")
    
    # Load split video IDs
    train_videos = load_split_videos(os.path.join(args.splits_dir, "train_videos.json"))
    val_videos = load_split_videos(os.path.join(args.splits_dir, "val_videos.json"))
    test_videos = load_split_videos(os.path.join(args.splits_dir, "test_videos.json"))
    
    logging.info(f"Train videos: {len(train_videos)}")
    logging.info(f"Val videos: {len(val_videos)}")
    logging.info(f"Test videos (NEVER for training): {test_videos}")
    
    # Verify test split is not in train/val
    train_val_set = set(train_videos + val_videos)
    test_set = set(test_videos)
    overlap = train_val_set & test_set
    if overlap:
        raise ValueError(f"CRITICAL: Test videos found in train/val: {overlap}")
    
    # Create datasets
    train_dataset = SSGVQADataset(
        ssgvqa_root=args.ssgvqa_root,
        image_root=args.image_root,
        video_ids=train_videos,
        processor=model.processor,
        label2idx=LABEL2IDX,
        norm_label_map=NORM_LABEL_MAP,
        labels_list=SSGVQA_LABELS,
        num_frames=args.num_frames,
        prompt_mode=args.prompt_mode,
        max_samples=args.max_train_samples,
    )
    
    val_dataset = SSGVQADataset(
        ssgvqa_root=args.ssgvqa_root,
        image_root=args.image_root,
        video_ids=val_videos,
        processor=model.processor,
        label2idx=LABEL2IDX,
        norm_label_map=NORM_LABEL_MAP,
        labels_list=SSGVQA_LABELS,
        num_frames=args.num_frames,
        prompt_mode=args.prompt_mode,
        max_samples=args.max_val_samples,
    )
    
    logging.info(f"Train samples: {len(train_dataset)}")
    logging.info(f"Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_ssgvqa_train,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_ssgvqa_train,
        pin_memory=True,
    )
    
    logging.info(f"Train batches: {len(train_loader)}")
    logging.info(f"Val batches: {len(val_loader)}")
    
    # =========================================================================
    # Optimizer
    # =========================================================================
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )
    logging.info(f"Optimizer: AdamW, LR={args.lr}")
    
    # =========================================================================
    # Training Loop
    # =========================================================================
    logging.info("Starting training...")
    
    best_val_loss = float('inf')
    best_epoch = 0
    epochs_since_improvement = 0
    
    for epoch in range(1, args.epochs + 1):
        logging.info(f"\n{'='*80}")
        logging.info(f"Epoch {epoch}/{args.epochs}")
        logging.info(f"{'='*80}")
        
        # Learning rate decay
        if epochs_since_improvement > 0 and epochs_since_improvement % args.lr_decay_patience == 0:
            adjust_learning_rate(optimizer, args.lr_decay_factor)
        
        # Train
        train_loss = train_epoch(
            model=model,
            tokenizer=tokenizer,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            max_length=args.max_length,
            log_interval=args.log_interval,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_grad_norm=args.max_grad_norm,
        )
        
        # Validate
        val_metrics = validate_epoch(
            model=model,
            tokenizer=tokenizer,
            val_loader=val_loader,
            device=device,
            epoch=epoch,
            max_length=args.max_length,
        )
        
        # Check for improvement
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_epoch = epoch
            epochs_since_improvement = 0
            
            # Save best checkpoint
            save_checkpoint(
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                epoch=epoch,
                val_metrics=val_metrics,
                checkpoint_dir=args.checkpoint_dir,
                filename="best_model.pth",
            )
            logging.info(f"New best model saved! Val loss: {best_val_loss:.6f}")
        else:
            epochs_since_improvement += 1
            logging.info(f"No improvement. Epochs since best: {epochs_since_improvement}")
        
        # Early stopping
        if epochs_since_improvement >= args.early_stop_patience:
            logging.info(f"Early stopping triggered after {epoch} epochs")
            break
        
        # Save periodic checkpoint
        if epoch % 5 == 0:
            save_checkpoint(
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                epoch=epoch,
                val_metrics=val_metrics,
                checkpoint_dir=args.checkpoint_dir,
                filename=f"checkpoint_epoch{epoch}.pth",
            )
    
    # =========================================================================
    # Final Summary
    # =========================================================================
    logging.info("\n" + "="*80)
    logging.info("Training Complete!")
    logging.info("="*80)
    logging.info(f"Best epoch: {best_epoch}")
    logging.info(f"Best val loss: {best_val_loss:.6f}")
    logging.info(f"Checkpoint saved to: {args.checkpoint_dir}")
    logging.info("")
    logging.info("To evaluate on test set, run:")
    logging.info(f"  python utils/eval_surgvivqa_ssgvqa.py \\")
    logging.info(f"      --model-path {os.path.join(args.checkpoint_dir, 'best_model.pth')} \\")
    logging.info(f"      --ssgvqa-root {args.ssgvqa_root} \\")
    logging.info(f"      --image-root {args.image_root} \\")
    logging.info(f"      --videos VID02 VID22 VID43 VID60 VID74 \\")
    logging.info(f"      --prompt-mode {args.prompt_mode} \\")
    logging.info(f"      --log-file {os.path.join(args.checkpoint_dir, 'eval.log')} \\")
    logging.info(f"      --predictions-file {os.path.join(args.checkpoint_dir, 'predictions.jsonl')}")


if __name__ == "__main__":
    main()
