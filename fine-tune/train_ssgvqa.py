"""
SurgViVQA Fine-tuning on SSGVQA Dataset (FIXED VERSION)

Key fixes from original:
- LoRA-only by default: BLIP text encoder and GPT-2 base weights frozen
- Proper loss alignment with diagnostics (supervised token counts)
- Data split sanity checks (no train/val/test overlap)
- HPC optimizations: max-steps, AMP, step-based eval/save
- Module-wise trainable param breakdown

Usage:
    # LoRA fine-tuning (default, recommended for single GPU)
    python train_ssgvqa.py \
        --ssgvqa-root /mnt/scratch/sc232jl/datasets/SSGVQA/ssg-qa/ \
        --image-root /mnt/scratch/sc232jl/datasets/CholecT45/data \
        --max-steps 30000 --batch-size 8 --lr 5e-4

Copyright (c) 2024. All Rights Reserved.
"""

import os
import sys
import argparse
import logging
import random
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

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
    """Set random seed for reproducibility."""
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
        help="Full fine-tuning of GPT-2 instead of LoRA (requires more VRAM)"
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
        help="Number of frames for video input"
    )
    
    # -------------------------------------------------------------------------
    # Freeze control (NEW - critical for LoRA-only)
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--train-blip", action="store_true",
        help="Unfreeze entire BLIP text encoder (default: frozen)"
    )
    parser.add_argument(
        "--train-gpt2-full", action="store_true",
        help="Unfreeze GPT-2 base weights (default: only LoRA adapters trainable)"
    )
    parser.add_argument(
        "--unfreeze-blip-last-n", type=int, default=0,
        help="Unfreeze last N layers of BLIP text encoder (default: 0 = all frozen)"
    )
    
    # -------------------------------------------------------------------------
    # Training configuration
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Max number of training epochs (use --max-steps for time-limited jobs)"
    )
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help="Max training steps (overrides epochs if set)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Batch size per GPU"
    )
    parser.add_argument(
        "--lr", type=float, default=5e-4,
        help="Learning rate (higher for LoRA-only, e.g., 1e-4 to 5e-4)"
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
        "--warmup-steps", type=int, default=500,
        help="Linear warmup steps"
    )
    
    # -------------------------------------------------------------------------
    # AMP (Mixed Precision)
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--fp16", action="store_true",
        help="Use FP16 mixed precision training"
    )
    parser.add_argument(
        "--bf16", action="store_true",
        help="Use BF16 mixed precision training (requires Ampere+ GPU)"
    )
    
    # -------------------------------------------------------------------------
    # Evaluation and checkpointing (step-based)
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--eval-every-steps", type=int, default=2000,
        help="Evaluate every N steps"
    )
    parser.add_argument(
        "--save-every-steps", type=int, default=5000,
        help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--val-max-samples", type=int, default=20000,
        help="Max validation samples for quick eval (None for full val set)"
    )
    
    # -------------------------------------------------------------------------
    # Prompt configuration
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--prompt-mode", type=str, default="simple",
        choices=["simple", "choices"],
        help="Prompt mode: 'simple' (short) or 'choices' (includes all labels)"
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
        "--log-interval", type=int, default=100,
        help="Log every N steps"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Debug mode (use fewer samples)"
    )
    parser.add_argument(
        "--max-train-samples", type=int, default=None,
        help="Max training samples (for debugging)"
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
        args.max_train_samples = args.max_train_samples or 500
        args.val_max_samples = 200
        args.max_steps = 100
        args.eval_every_steps = 20
        args.save_every_steps = 50
        args.log_interval = 10
    
    return args


def verify_splits_no_overlap(train_videos: List[str], val_videos: List[str], test_videos: List[str]) -> None:
    """Verify no overlap between train/val/test splits. Abort if overlap found."""
    train_set = set(train_videos)
    val_set = set(val_videos)
    test_set = set(test_videos)
    
    train_val_overlap = train_set & val_set
    train_test_overlap = train_set & test_set
    val_test_overlap = val_set & test_set
    
    has_overlap = False
    
    if train_val_overlap:
        logging.error(f"CRITICAL: Train-Val overlap: {train_val_overlap}")
        has_overlap = True
    if train_test_overlap:
        logging.error(f"CRITICAL: Train-Test overlap: {train_test_overlap}")
        has_overlap = True
    if val_test_overlap:
        logging.error(f"CRITICAL: Val-Test overlap: {val_test_overlap}")
        has_overlap = True
    
    if has_overlap:
        raise ValueError("Data leakage detected! Aborting training.")
    
    logging.info("Split verification passed: No overlap between train/val/test")


def freeze_model_components(
    model: SurgViVQA,
    train_blip: bool = False,
    train_gpt2_full: bool = False,
    unfreeze_blip_last_n: int = 0,
) -> None:
    """
    Freeze model components for LoRA-only training by default.
    
    Default behavior (all False, n=0):
    - VideoMAE: frozen (already done in model.__init__)
    - BLIP text encoder: frozen
    - GPT-2 base weights: frozen (only LoRA adapters trainable via PEFT)
    """
    # 1) VideoMAE is already frozen in model.__init__
    
    # 2) Freeze BLIP text encoder entirely first
    for name, param in model.text_encoder.named_parameters():
        param.requires_grad = False
    
    # Optionally unfreeze BLIP
    if train_blip:
        logging.info("Unfreezing entire BLIP text encoder")
        for param in model.text_encoder.parameters():
            param.requires_grad = True
    elif unfreeze_blip_last_n > 0:
        # Unfreeze last N layers
        logging.info(f"Unfreezing last {unfreeze_blip_last_n} layers of BLIP text encoder")
        encoder_layers = model.text_encoder.encoder.layer
        num_layers = len(encoder_layers)
        for i, layer in enumerate(encoder_layers):
            if i >= num_layers - unfreeze_blip_last_n:
                for param in layer.parameters():
                    param.requires_grad = True
    
    # 3) For GPT-2 with PEFT/LoRA, base weights are frozen by default
    # Only unfreeze if explicitly requested
    if train_gpt2_full:
        logging.info("Unfreezing GPT-2 base weights (full fine-tuning)")
        # Access base model through PEFT wrapper
        if hasattr(model.llm, 'base_model'):
            for name, param in model.llm.base_model.named_parameters():
                if 'lora' not in name.lower():
                    param.requires_grad = True
        else:
            for param in model.llm.parameters():
                param.requires_grad = True


def print_trainable_params_breakdown(model: SurgViVQA) -> Dict[str, int]:
    """Print module-wise breakdown of trainable parameters."""
    breakdown = {
        "visual_encoder": 0,
        "text_encoder": 0,
        "llm_base": 0,
        "llm_lora": 0,
        "other": 0,
    }
    
    trainable_names = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_names.append(name)
            num_params = param.numel()
            
            if name.startswith("visual_encoder"):
                breakdown["visual_encoder"] += num_params
            elif name.startswith("text_encoder"):
                breakdown["text_encoder"] += num_params
            elif name.startswith("llm"):
                if "lora" in name.lower():
                    breakdown["llm_lora"] += num_params
                else:
                    breakdown["llm_base"] += num_params
            else:
                breakdown["other"] += num_params
    
    total_trainable = sum(breakdown.values())
    total_params = sum(p.numel() for p in model.parameters())
    
    logging.info("=" * 60)
    logging.info("TRAINABLE PARAMETERS BREAKDOWN")
    logging.info("=" * 60)
    logging.info(f"  visual_encoder: {breakdown['visual_encoder']:,}")
    logging.info(f"  text_encoder (BLIP): {breakdown['text_encoder']:,}")
    logging.info(f"  llm_base (GPT-2 base): {breakdown['llm_base']:,}")
    logging.info(f"  llm_lora (LoRA adapters): {breakdown['llm_lora']:,}")
    logging.info(f"  other: {breakdown['other']:,}")
    logging.info("-" * 60)
    logging.info(f"  TOTAL TRAINABLE: {total_trainable:,}")
    logging.info(f"  TOTAL PARAMS: {total_params:,}")
    logging.info(f"  TRAINABLE RATIO: {100 * total_trainable / total_params:.4f}%")
    logging.info("=" * 60)
    
    # Print first 30 trainable param names for sanity check
    logging.info("First 30 trainable parameter names:")
    for i, name in enumerate(trainable_names[:30]):
        logging.info(f"  [{i+1}] {name}")
    if len(trainable_names) > 30:
        logging.info(f"  ... and {len(trainable_names) - 30} more")
    logging.info("=" * 60)
    
    # Sanity check for LoRA-only
    if breakdown["visual_encoder"] > 0:
        logging.warning("WARNING: VideoMAE has trainable params (should be frozen)")
    if breakdown["text_encoder"] > 0:
        logging.warning("WARNING: BLIP has trainable params (expected frozen by default)")
    if breakdown["llm_base"] > 0:
        logging.warning("WARNING: GPT-2 base has trainable params (expected frozen for LoRA-only)")
    if breakdown["llm_lora"] == 0 and total_trainable == 0:
        logging.error("ERROR: No trainable parameters! Check model configuration.")
    
    return breakdown


def compute_sft_loss_with_diagnostics(
    model: nn.Module,
    tokenizer,
    videos: torch.Tensor,
    prompt_prefixes: List[str],
    target_texts: List[str],
    device: torch.device,
    max_length: int = 128,
    return_diagnostics: bool = False,
) -> Tuple[torch.Tensor, Optional[Dict]]:
    """
    Compute SFT loss with correct causal LM shift and optional diagnostics.
    
    Critical alignment:
    - shift_logits = logits[:, :-1, :]
    - shift_labels = labels[:, 1:]
    - Labels: prompt tokens masked to -100, only answer tokens supervised
    
    Returns:
        loss: Scalar tensor
        diagnostics: Dict with supervised token counts (if return_diagnostics=True)
    """
    batch_size = len(prompt_prefixes)
    
    # Build full sequences: prompt_prefix + target_text
    full_sequences = [p + t for p, t in zip(prompt_prefixes, target_texts)]
    
    # Tokenize full sequences
    full_tok = tokenizer(
        full_sequences,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = full_tok["input_ids"].to(device)  # [B, L]
    attention_mask = full_tok["attention_mask"].to(device)  # [B, L]
    
    # Create labels (copy of input_ids, will mask prompt tokens)
    labels = input_ids.clone()
    
    # Mask prompt tokens in labels (set to -100)
    for idx, prompt_prefix in enumerate(prompt_prefixes):
        # Tokenize prompt prefix alone to find its length
        prompt_tok = tokenizer(prompt_prefix, add_special_tokens=False)
        prompt_len = len(prompt_tok["input_ids"])
        
        # Mask ALL prompt tokens (indices 0 to prompt_len-1)
        # After shift, logits[prompt_len-1] predicts labels[prompt_len] = first answer token
        labels[idx, :prompt_len] = -100
        
        # Also mask padding tokens
        pad_positions = (input_ids[idx] == tokenizer.pad_token_id)
        labels[idx][pad_positions] = -100
        
        # Keep exactly one EOS as the end signal if present
        eos_mask = (input_ids[idx] == tokenizer.eos_token_id)
        if eos_mask.sum() > 1:
            first_eos_pos = eos_mask.nonzero()[0].item()
            labels[idx, (first_eos_pos + 1):] = -100
    
    # Forward pass through model
    logits = model(
        video=videos.to(device),
        qa_inputs_ids=input_ids,
        qa_att_mask=attention_mask,
    )  # [B, L, vocab_size]
    
    # Shift logits and labels for causal LM
    # logits[:, i, :] predicts token at position i+1
    # So logits[:, :-1, :] predicts labels[:, 1:]
    shift_logits = logits[:, :-1, :].contiguous()  # [B, L-1, vocab_size]
    shift_labels = labels[:, 1:].contiguous()  # [B, L-1]
    
    # Compute cross entropy loss (ignoring -100)
    vocab_size = shift_logits.size(-1)
    loss = nn.functional.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        ignore_index=-100,
    )
    
    diagnostics = None
    if return_diagnostics:
        # Count supervised tokens
        supervised_mask = (shift_labels != -100)
        total_supervised = supervised_mask.sum().item()
        per_sample_supervised = supervised_mask.sum(dim=1)  # [B]
        samples_with_zero = (per_sample_supervised == 0).sum().item()
        
        diagnostics = {
            "total_supervised_tokens": total_supervised,
            "avg_supervised_per_sample": total_supervised / batch_size,
            "samples_with_zero_supervised": samples_with_zero,
            "batch_size": batch_size,
        }
    
    return loss, diagnostics


def save_checkpoint(
    model: nn.Module,
    tokenizer,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[GradScaler],
    global_step: int,
    val_metrics: Dict[str, float],
    checkpoint_dir: str,
    filename: str = "best_model.pth",
) -> str:
    """Save model checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    
    # Save model state dict
    torch.save(model.state_dict(), checkpoint_path)
    
    # Save tokenizer
    tokenizer.save_pretrained(checkpoint_dir)
    
    # Save training state separately
    state_path = os.path.join(checkpoint_dir, f"training_state_{filename.replace('.pth', '')}.pth")
    state = {
        "global_step": global_step,
        "optimizer_state_dict": optimizer.state_dict(),
        "val_metrics": val_metrics,
    }
    if scaler is not None:
        state["scaler_state_dict"] = scaler.state_dict()
    torch.save(state, state_path)
    
    logging.info(f"Checkpoint saved: {checkpoint_path}")
    
    return checkpoint_path


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device,
    strict: bool = False,
) -> None:
    """Load model checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt
    
    missing, unexpected = model.load_state_dict(state, strict=strict)
    
    logging.info(f"Loaded checkpoint: {checkpoint_path}")
    logging.info(f"Missing keys: {len(missing)}")
    logging.info(f"Unexpected keys: {len(unexpected)}")


@torch.no_grad()
def validate(
    model: nn.Module,
    tokenizer,
    val_loader,
    device: torch.device,
    max_length: int,
    max_batches: Optional[int] = None,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
) -> Dict[str, float]:
    """Quick validation for monitoring during training."""
    model.eval()
    total_loss = 0.0
    total_batches = 0
    total_supervised = 0
    total_samples = 0
    
    for batch_idx, batch in enumerate(val_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        
        videos, prompt_prefixes, target_texts, label_indices, _, _ = batch
        
        # Skip batches with invalid labels
        if (label_indices < 0).any():
            continue
        
        with autocast(enabled=use_amp, dtype=amp_dtype):
            loss, diag = compute_sft_loss_with_diagnostics(
                model=model,
                tokenizer=tokenizer,
                videos=videos,
                prompt_prefixes=prompt_prefixes,
                target_texts=target_texts,
                device=device,
                max_length=max_length,
                return_diagnostics=True,
            )
        
        total_loss += loss.item()
        total_batches += 1
        total_supervised += diag["total_supervised_tokens"]
        total_samples += diag["batch_size"]
    
    avg_loss = total_loss / max(1, total_batches)
    avg_supervised = total_supervised / max(1, total_samples)
    
    model.train()
    
    return {
        "loss": avg_loss,
        "avg_supervised_tokens": avg_supervised,
        "num_batches": total_batches,
        "num_samples": total_samples,
    }


def create_model(
    tokenizer,
    device: torch.device,
    num_frames: int,
    full_finetune: bool = False,
    lora_r: int = 8,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
) -> SurgViVQA:
    """Create SurgViVQA model with optional LoRA configuration."""
    # Load base decoder model
    decoder_model = AutoModelForCausalLM.from_pretrained("gpt2")
    
    # LoRA configuration (only applied if not full fine-tuning)
    peft_config = None
    if not full_finetune:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["c_attn", "c_proj"],
        )
        logging.info(f"Using LoRA: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
    else:
        logging.info("Using full fine-tuning (no LoRA)")
    
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
    logging.info("SurgViVQA Fine-tuning on SSGVQA (FIXED VERSION)")
    logging.info("=" * 80)
    logging.info("Configuration:")
    for key, value in vars(args).items():
        logging.info(f"  {key}: {value}")
    logging.info("=" * 80)
    
    # Save config
    config_path = os.path.join(args.checkpoint_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    
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
    
    # CRITICAL: Freeze components for LoRA-only training
    freeze_model_components(
        model=model,
        train_blip=args.train_blip,
        train_gpt2_full=args.train_gpt2_full,
        unfreeze_blip_last_n=args.unfreeze_blip_last_n,
    )
    
    model = model.to(device)
    
    # Load pretrained checkpoint if provided
    if args.pretrained_ckpt:
        logging.info(f"Loading pretrained checkpoint: {args.pretrained_ckpt}")
        load_checkpoint(model, args.pretrained_ckpt, device, strict=False)
    
    # Print trainable parameters breakdown (CRITICAL for debugging)
    print_trainable_params_breakdown(model)
    
    # =========================================================================
    # Dataset and DataLoaders
    # =========================================================================
    logging.info("Loading datasets...")
    
    # Load split video IDs
    train_videos = load_split_videos(os.path.join(args.splits_dir, "train_videos.json"))
    val_videos = load_split_videos(os.path.join(args.splits_dir, "val_videos.json"))
    test_videos = load_split_videos(os.path.join(args.splits_dir, "test_videos.json"))
    
    # CRITICAL: Verify no overlap (data leakage prevention)
    verify_splits_no_overlap(train_videos, val_videos, test_videos)
    
    logging.info(f"Train videos: {len(train_videos)} - {train_videos[:5]}...")
    logging.info(f"Val videos: {len(val_videos)} - {val_videos}")
    logging.info(f"Test videos (NEVER for training): {test_videos}")
    
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
        max_samples=args.val_max_samples,
    )
    
    logging.info(f"Train samples: {len(train_dataset)}")
    logging.info(f"Val samples: {len(val_dataset)}")
    
    # Dataloader kwargs for efficiency
    loader_kwargs = dict(
        pin_memory=True,
        collate_fn=collate_ssgvqa_train,
    )
    if args.workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
        **loader_kwargs,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        **loader_kwargs,
    )
    
    steps_per_epoch = len(train_loader)
    logging.info(f"Train batches per epoch: {steps_per_epoch}")
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
    
    # AMP setup
    use_amp = args.fp16 or args.bf16
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16
    scaler = GradScaler() if args.fp16 else None
    logging.info(f"AMP: {use_amp}, dtype={amp_dtype if use_amp else 'fp32'}")
    
    # =========================================================================
    # Training Loop (step-based)
    # =========================================================================
    logging.info("Starting training...")
    
    global_step = 0
    best_val_loss = float('inf')
    best_step = 0
    total_train_loss = 0.0
    accumulated_steps = 0
    train_start_time = time.time()
    
    max_steps = args.max_steps if args.max_steps else args.epochs * steps_per_epoch
    logging.info(f"Max steps: {max_steps}")
    
    model.train()
    optimizer.zero_grad()
    
    epoch = 0
    data_iter = iter(train_loader)
    
    while global_step < max_steps:
        # Get next batch (handle epoch boundaries)
        try:
            batch = next(data_iter)
        except StopIteration:
            epoch += 1
            data_iter = iter(train_loader)
            batch = next(data_iter)
            logging.info(f"Starting epoch {epoch}")
        
        videos, prompt_prefixes, target_texts, label_indices, _, _ = batch
        
        # Skip batches with invalid labels
        if (label_indices < 0).any():
            continue
        
        # Forward pass with AMP
        with autocast(enabled=use_amp, dtype=amp_dtype):
            loss, diag = compute_sft_loss_with_diagnostics(
                model=model,
                tokenizer=tokenizer,
                videos=videos,
                prompt_prefixes=prompt_prefixes,
                target_texts=target_texts,
                device=device,
                max_length=args.max_length,
                return_diagnostics=(global_step % args.log_interval == 0),
            )
        
        # CRITICAL: Check for zero supervised tokens (indicates bug)
        if diag is not None and diag["samples_with_zero_supervised"] == diag["batch_size"]:
            logging.error(f"CRITICAL: Step {global_step}: ALL samples have 0 supervised tokens!")
            logging.error(f"  Batch prompts[0]: {prompt_prefixes[0][:100]}...")
            logging.error(f"  Batch targets[0]: {target_texts[0]}")
            raise RuntimeError("Zero supervised tokens in entire batch - training is meaningless!")
        
        # Scale loss for gradient accumulation
        scaled_loss = loss / args.gradient_accumulation_steps
        
        # Backward pass
        if scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        
        total_train_loss += loss.item()
        accumulated_steps += 1
        
        # Update weights
        if accumulated_steps >= args.gradient_accumulation_steps:
            if args.max_grad_norm > 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            optimizer.zero_grad()
            global_step += 1
            
            # Warmup LR
            if global_step <= args.warmup_steps:
                warmup_lr = args.lr * global_step / args.warmup_steps
                for pg in optimizer.param_groups:
                    pg['lr'] = warmup_lr
            
            # Logging
            if global_step % args.log_interval == 0:
                avg_loss = total_train_loss / accumulated_steps
                elapsed = time.time() - train_start_time
                samples_per_sec = global_step * args.batch_size / elapsed
                current_lr = optimizer.param_groups[0]['lr']
                
                log_msg = (
                    f"Step {global_step}/{max_steps} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"LR: {current_lr:.2e} | "
                    f"Speed: {samples_per_sec:.1f} samples/sec"
                )
                if diag is not None:
                    log_msg += f" | SupToks/sample: {diag['avg_supervised_per_sample']:.1f}"
                logging.info(log_msg)
                
                total_train_loss = 0.0
            
            # Evaluation
            if global_step % args.eval_every_steps == 0:
                logging.info(f"Evaluating at step {global_step}...")
                val_metrics = validate(
                    model=model,
                    tokenizer=tokenizer,
                    val_loader=val_loader,
                    device=device,
                    max_length=args.max_length,
                    max_batches=len(val_loader),
                    use_amp=use_amp,
                    amp_dtype=amp_dtype,
                )
                logging.info(
                    f"Val Loss: {val_metrics['loss']:.4f} | "
                    f"Avg SupToks: {val_metrics['avg_supervised_tokens']:.1f}"
                )
                
                # Save best model
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    best_step = global_step
                    save_checkpoint(
                        model=model,
                        tokenizer=tokenizer,
                        optimizer=optimizer,
                        scaler=scaler,
                        global_step=global_step,
                        val_metrics=val_metrics,
                        checkpoint_dir=args.checkpoint_dir,
                        filename="best_model.pth",
                    )
                    logging.info(f"New best model! Val loss: {best_val_loss:.4f}")
                
                model.train()
            
            # Periodic checkpoint
            if global_step % args.save_every_steps == 0:
                save_checkpoint(
                    model=model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    scaler=scaler,
                    global_step=global_step,
                    val_metrics={"loss": total_train_loss / max(1, accumulated_steps)},
                    checkpoint_dir=args.checkpoint_dir,
                    filename=f"checkpoint_step{global_step}.pth",
                )
            
            accumulated_steps = 0
    
    # =========================================================================
    # Final Summary
    # =========================================================================
    total_time = time.time() - train_start_time
    logging.info("\n" + "=" * 80)
    logging.info("Training Complete!")
    logging.info("=" * 80)
    logging.info(f"Total steps: {global_step}")
    logging.info(f"Total time: {total_time / 3600:.2f} hours")
    logging.info(f"Best step: {best_step}")
    logging.info(f"Best val loss: {best_val_loss:.6f}")
    logging.info(f"Checkpoint saved to: {args.checkpoint_dir}")
    logging.info("")
    logging.info("To evaluate on test set, run:")
    logging.info(f"  python {_PROJECT_ROOT}/utils/eval_surgvivqa_ssgvqa.py \\")
    logging.info(f"      --model-path {os.path.join(args.checkpoint_dir, 'best_model.pth')} \\")
    logging.info(f"      --ssgvqa-root {args.ssgvqa_root} \\")
    logging.info(f"      --image-root {args.image_root} \\")
    logging.info(f"      --videos VID02 VID22 VID43 VID60 VID74 \\")
    logging.info(f"      --prompt-mode {args.prompt_mode}")


if __name__ == "__main__":
    main()
