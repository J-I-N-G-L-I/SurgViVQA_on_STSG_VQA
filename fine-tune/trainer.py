"""
Training utilities for SSGVQA fine-tuning of SurgViVQA.

Implements:
- Supervised Instruction Fine-tuning (SFT) with prompt masking
- Loss computation only on answer tokens (label)
- Learning rate scheduling
- Checkpoint saving

Key requirement: Training prompt + target format MUST be compatible with the 
existing evaluation script (utils/eval_surgvivqa_ssgvqa.py), which uses 
teacher-forcing log-prob scoring.

Training target format:
- prompt_prefix: "Question: ...\nAnswer:" (matches eval exactly)
- target_text: " " + LABEL (leading space for GPT-2 tokenization)
- Loss is computed ONLY on target tokens (prompt tokens are masked with -100)
"""

import os
import logging
import time
from typing import Dict, Any, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def adjust_learning_rate(optimizer: torch.optim.Optimizer, shrink_factor: float = 0.8) -> None:
    """Decay learning rate by shrink_factor."""
    logging.info("Decaying learning rate...")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    logging.info(f"New learning rate: {optimizer.param_groups[0]['lr']:.8f}")


def compute_sft_loss(
    model: nn.Module,
    tokenizer,
    videos: torch.Tensor,
    prompt_prefixes: List[str],
    target_texts: List[str],
    device: torch.device,
    max_length: int = 128,
) -> torch.Tensor:
    """
    Compute SFT loss with prompt masking.
    
    This implements causal LM training where:
    - Full sequence = prompt_prefix + target_text
    - Loss is only computed on target tokens (prompt tokens masked with -100)
    
    This matches the evaluation script's teacher-forcing scoring approach.
    
    Args:
        model: SurgViVQA model
        tokenizer: GPT-2 tokenizer
        videos: [B, T, 3, H, W] video tensor
        prompt_prefixes: List of prompt strings (length B)
        target_texts: List of target strings (length B), each " " + LABEL
        device: CUDA/CPU device
        max_length: Max sequence length for tokenization
    
    Returns:
        loss: Scalar tensor
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
    # For each sample, find where the answer starts and mask everything before
    for idx, prompt_prefix in enumerate(prompt_prefixes):
        # Tokenize prompt prefix alone to find its length
        prompt_tok = tokenizer(prompt_prefix, add_special_tokens=False)
        prompt_len = len(prompt_tok["input_ids"])
        
        # Mask prompt tokens (indices 0 to prompt_len-1)
        # Note: we keep the last prompt token in labels because LM predicts next token
        # So for position i, we predict token i+1. Mask positions 0 to prompt_len-2.
        # Actually, for causal LM: logits[i] predicts labels[i+1] after shifting.
        # We want to mask the prompt portion of the labels.
        mask_end = max(prompt_len - 1, 0)  # Keep one overlap for prediction boundary
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
    # logits[:, :-1, :] predicts labels[:, 1:]
    shift_logits = logits[:, :-1, :].contiguous()  # [B, L-1, vocab_size]
    shift_labels = labels[:, 1:].contiguous()  # [B, L-1]
    
    # Compute cross entropy loss (ignoring -100)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fn(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )
    
    return loss


def train_epoch(
    model: nn.Module,
    tokenizer,
    train_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_length: int = 128,
    log_interval: int = 50,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
) -> float:
    """
    Train for one epoch.
    
    Args:
        model: SurgViVQA model
        tokenizer: GPT-2 tokenizer
        train_loader: Training dataloader
        optimizer: Optimizer
        device: CUDA/CPU device
        epoch: Current epoch number
        max_length: Max sequence length
        log_interval: Log every N batches
        gradient_accumulation_steps: Number of steps to accumulate gradients
        max_grad_norm: Max gradient norm for clipping
    
    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0
    total_batches = 0
    
    optimizer.zero_grad()
    start_time = time.time()
    
    for batch_idx, batch in enumerate(train_loader):
        videos, prompt_prefixes, target_texts, label_indices, video_ids, frame_ids = batch
        
        # Skip batches with invalid labels
        if (label_indices < 0).any():
            continue
        
        # Compute loss
        loss = compute_sft_loss(
            model=model,
            tokenizer=tokenizer,
            videos=videos,
            prompt_prefixes=prompt_prefixes,
            target_texts=target_texts,
            device=device,
            max_length=max_length,
        )
        
        # Scale loss for gradient accumulation
        loss = loss / gradient_accumulation_steps
        loss.backward()
        
        # Update weights
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # Gradient clipping
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * gradient_accumulation_steps
        total_batches += 1
        
        # Logging
        if (batch_idx + 1) % log_interval == 0:
            elapsed = time.time() - start_time
            samples_per_sec = (batch_idx + 1) * len(videos) / elapsed
            logging.info(
                f"Epoch {epoch} | Batch {batch_idx + 1}/{len(train_loader)} | "
                f"Loss: {loss.item() * gradient_accumulation_steps:.4f} | "
                f"Speed: {samples_per_sec:.1f} samples/sec"
            )
    
    avg_loss = total_loss / max(1, total_batches)
    elapsed = time.time() - start_time
    logging.info(
        f"Epoch {epoch} Training complete | Avg Loss: {avg_loss:.6f} | Time: {elapsed:.1f}s"
    )
    
    return avg_loss


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    tokenizer,
    val_loader,
    device: torch.device,
    epoch: int,
    max_length: int = 128,
) -> Dict[str, float]:
    """
    Validate for one epoch.
    
    Computes:
    - Average validation loss (SFT loss)
    - Accuracy (greedy decoding matches ground truth label)
    
    Args:
        model: SurgViVQA model
        tokenizer: GPT-2 tokenizer
        val_loader: Validation dataloader
        device: CUDA/CPU device
        epoch: Current epoch number
        max_length: Max sequence length
    
    Returns:
        Dictionary with 'loss' and 'accuracy'
    """
    model.eval()
    total_loss = 0.0
    total_batches = 0
    correct = 0
    total = 0
    
    for batch_idx, batch in enumerate(val_loader):
        videos, prompt_prefixes, target_texts, label_indices, video_ids, frame_ids = batch
        
        # Skip batches with invalid labels
        valid_mask = label_indices >= 0
        if not valid_mask.any():
            continue
        
        # Compute loss
        loss = compute_sft_loss(
            model=model,
            tokenizer=tokenizer,
            videos=videos,
            prompt_prefixes=prompt_prefixes,
            target_texts=target_texts,
            device=device,
            max_length=max_length,
        )
        
        total_loss += loss.item()
        total_batches += 1
        
        # Compute accuracy via greedy decoding comparison
        # (Simplified: we check if the predicted first token matches target)
        # For true accuracy, use the full evaluation script after training
        
        # Tokenize prompts only
        prompt_tok = tokenizer(
            prompt_prefixes,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        prompt_ids = prompt_tok["input_ids"].to(device)
        prompt_mask = prompt_tok["attention_mask"].to(device)
        
        # Forward to get logits
        logits = model(
            video=videos.to(device),
            qa_inputs_ids=prompt_ids,
            qa_att_mask=prompt_mask,
        )
        
        # Get predicted next token at the end of prompt
        # For each sample, find the last non-pad position
        prompt_lens = prompt_mask.sum(dim=1)  # [B]
        
        for i, (target, label_idx) in enumerate(zip(target_texts, label_indices)):
            if label_idx < 0:
                continue
            
            pos = min(int(prompt_lens[i].item()) - 1, logits.size(1) - 1)
            pred_token = logits[i, pos, :].argmax().item()
            
            # Tokenize target (just the label with leading space)
            target_tok = tokenizer(target, add_special_tokens=False)
            if target_tok["input_ids"]:
                target_first_token = target_tok["input_ids"][0]
                if pred_token == target_first_token:
                    correct += 1
            total += 1
    
    avg_loss = total_loss / max(1, total_batches)
    accuracy = correct / max(1, total)
    
    logging.info(
        f"Epoch {epoch} Validation | Loss: {avg_loss:.6f} | "
        f"Token Acc: {accuracy:.4f} ({correct}/{total})"
    )
    
    return {"loss": avg_loss, "accuracy": accuracy}


def save_checkpoint(
    model: nn.Module,
    tokenizer,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_metrics: Dict[str, float],
    checkpoint_dir: str,
    filename: str = "best_model.pth",
) -> str:
    """
    Save model checkpoint.
    
    Args:
        model: SurgViVQA model
        tokenizer: GPT-2 tokenizer
        optimizer: Optimizer
        epoch: Current epoch
        val_metrics: Validation metrics dictionary
        checkpoint_dir: Directory to save checkpoint
        filename: Checkpoint filename
    
    Returns:
        Path to saved checkpoint
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    
    # Save model state dict (handles PEFT models correctly)
    torch.save(model.state_dict(), checkpoint_path)
    
    # Save tokenizer
    tokenizer.save_pretrained(checkpoint_dir)
    
    # Save training state separately
    state_path = os.path.join(checkpoint_dir, "training_state.pth")
    torch.save({
        "epoch": epoch,
        "optimizer_state_dict": optimizer.state_dict(),
        "val_metrics": val_metrics,
    }, state_path)
    
    logging.info(f"Checkpoint saved: {checkpoint_path}")
    
    return checkpoint_path


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device,
    strict: bool = False,
) -> None:
    """
    Load model checkpoint.
    
    Args:
        model: SurgViVQA model
        checkpoint_path: Path to checkpoint file
        device: Target device
        strict: Whether to require exact key match
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt
    
    missing, unexpected = model.load_state_dict(state, strict=strict)
    
    logging.info(f"Loaded checkpoint: {checkpoint_path}")
    logging.info(f"Missing keys: {len(missing)}")
    logging.info(f"Unexpected keys: {len(unexpected)}")
    
    if missing:
        logging.debug(f"Missing keys (first 10): {missing[:10]}")
    if unexpected:
        logging.debug(f"Unexpected keys (first 10): {unexpected[:10]}")
