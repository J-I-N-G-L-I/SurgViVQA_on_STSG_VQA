#!/usr/bin/env python
"""
Sanity Check Script for SSGVQA Fine-tuning Pipeline

Run this BEFORE submitting the full training job to verify:
1. Data paths are correct and accessible
2. Split files exist and have no overlap
3. Dataset can load samples correctly
4. Model can be created with correct trainable params (LoRA-only)
5. Loss computation produces valid gradients
6. At least one training step can complete

Usage:
    python fine-tune/sanity_check.py \
        --ssgvqa-root /path/to/ssg-qa/ \
        --image-root /path/to/CholecT45/data

Expected output:
    - All checks PASS
    - Trainable params should be ~0.15% (LoRA-only)
    - Loss should be ~3-5 (cross-entropy on 51 classes)
"""

import os
import sys
import argparse
import json
import logging

import torch

# Add paths
_FINE_TUNE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_FINE_TUNE_DIR)
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _FINE_TUNE_DIR)

# Imports
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType

from labels import SSGVQA_LABELS, LABEL2IDX, NORM_LABEL_MAP, NUM_CLASSES
from dataset import SSGVQADataset, collate_ssgvqa_train, load_split_videos
from models.model import SurgViVQA


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def check_pass(name: str, passed: bool):
    status = "✓ PASS" if passed else "✗ FAIL"
    logging.info(f"{status}: {name}")
    return passed


def main():
    parser = argparse.ArgumentParser(description="Sanity check for SSGVQA fine-tuning")
    parser.add_argument("--ssgvqa-root", type=str, 
                        default="/mnt/scratch/sc232jl/datasets/SSGVQA/ssg-qa/")
    parser.add_argument("--image-root", type=str,
                        default="/mnt/scratch/sc232jl/datasets/CholecT45/data")
    parser.add_argument("--splits-dir", type=str, default=None)
    parser.add_argument("--num-frames", type=int, default=16)
    args = parser.parse_args()
    
    setup_logging()
    
    logging.info("=" * 60)
    logging.info("SSGVQA Fine-tuning Sanity Check")
    logging.info("=" * 60)
    
    all_passed = True
    
    if args.splits_dir is None:
        args.splits_dir = os.path.join(_FINE_TUNE_DIR, "splits")
    
    # =========================================================================
    # Check 1: Data paths exist
    # =========================================================================
    logging.info("\n[Check 1] Data paths...")
    
    ssgvqa_exists = os.path.isdir(args.ssgvqa_root)
    all_passed &= check_pass(f"SSGVQA root exists: {args.ssgvqa_root}", ssgvqa_exists)
    
    image_exists = os.path.isdir(args.image_root)
    all_passed &= check_pass(f"Image root exists: {args.image_root}", image_exists)
    
    splits_exists = os.path.isdir(args.splits_dir)
    all_passed &= check_pass(f"Splits dir exists: {args.splits_dir}", splits_exists)
    
    if not all([ssgvqa_exists, image_exists, splits_exists]):
        logging.error("Data paths missing. Fix paths and re-run.")
        return 1
    
    # =========================================================================
    # Check 2: Split files and no overlap
    # =========================================================================
    logging.info("\n[Check 2] Split files...")
    
    train_file = os.path.join(args.splits_dir, "train_videos.json")
    val_file = os.path.join(args.splits_dir, "val_videos.json")
    test_file = os.path.join(args.splits_dir, "test_videos.json")
    
    all_passed &= check_pass(f"train_videos.json exists", os.path.isfile(train_file))
    all_passed &= check_pass(f"val_videos.json exists", os.path.isfile(val_file))
    all_passed &= check_pass(f"test_videos.json exists", os.path.isfile(test_file))
    
    try:
        train_videos = load_split_videos(train_file)
        val_videos = load_split_videos(val_file)
        test_videos = load_split_videos(test_file)
        
        logging.info(f"  Train videos: {len(train_videos)}")
        logging.info(f"  Val videos: {len(val_videos)}")
        logging.info(f"  Test videos: {len(test_videos)} - {test_videos}")
        
        # Check no overlap
        train_set = set(train_videos)
        val_set = set(val_videos)
        test_set = set(test_videos)
        
        no_train_val_overlap = len(train_set & val_set) == 0
        no_train_test_overlap = len(train_set & test_set) == 0
        no_val_test_overlap = len(val_set & test_set) == 0
        
        all_passed &= check_pass("No train-val overlap", no_train_val_overlap)
        all_passed &= check_pass("No train-test overlap", no_train_test_overlap)
        all_passed &= check_pass("No val-test overlap", no_val_test_overlap)
        
        # Check test set is the official split
        expected_test = {"VID02", "VID22", "VID43", "VID60", "VID74"}
        correct_test = test_set == expected_test
        all_passed &= check_pass(f"Test set is official: {expected_test}", correct_test)
        
    except Exception as e:
        logging.error(f"Error loading splits: {e}")
        all_passed = False
    
    # =========================================================================
    # Check 3: Labels vocabulary
    # =========================================================================
    logging.info("\n[Check 3] Labels vocabulary...")
    
    all_passed &= check_pass(f"NUM_CLASSES = {NUM_CLASSES}", NUM_CLASSES == 51)
    all_passed &= check_pass(f"SSGVQA_LABELS length = {len(SSGVQA_LABELS)}", len(SSGVQA_LABELS) == 51)
    all_passed &= check_pass(f"LABEL2IDX length = {len(LABEL2IDX)}", len(LABEL2IDX) == 51)
    
    # Check some expected labels
    expected_labels = ["grasp", "retract", "cut", "hook", "grasper", "specimen-bag"]
    for label in expected_labels:
        in_vocab = label in LABEL2IDX
        all_passed &= check_pass(f"Label '{label}' in vocabulary", in_vocab)
    
    # =========================================================================
    # Check 4: Model creation with LoRA
    # =========================================================================
    logging.info("\n[Check 4] Model creation...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"  Device: {device}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        all_passed &= check_pass("Tokenizer loaded", True)
        
        decoder_model = AutoModelForCausalLM.from_pretrained("gpt2")
        all_passed &= check_pass("GPT-2 loaded", True)
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["c_attn", "c_proj"],
        )
        
        model = SurgViVQA(
            device=device,
            tokenizer=tokenizer,
            decoder_model=decoder_model,
            peft_config=peft_config,
            num_frames=args.num_frames,
        )
        all_passed &= check_pass("SurgViVQA model created", True)
        
        # Freeze components for LoRA-only
        for param in model.text_encoder.parameters():
            param.requires_grad = False
        
        model = model.to(device)
        
        # Count trainable params
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        ratio = 100 * trainable / total
        
        logging.info(f"  Total params: {total:,}")
        logging.info(f"  Trainable params: {trainable:,}")
        logging.info(f"  Trainable ratio: {ratio:.4f}%")
        
        # LoRA-only should be < 1% trainable
        is_lora_only = ratio < 1.0
        all_passed &= check_pass(f"LoRA-only (trainable < 1%): {ratio:.4f}%", is_lora_only)
        
        if not is_lora_only:
            logging.error("CRITICAL: Too many trainable params! Check freeze logic.")
        
    except Exception as e:
        logging.error(f"Model creation error: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
        return 1
    
    # =========================================================================
    # Check 5: Dataset loading (small sample)
    # =========================================================================
    logging.info("\n[Check 5] Dataset loading...")
    
    try:
        # Load small sample from first train video
        test_video_id = train_videos[0] if train_videos else "VID01"
        
        dataset = SSGVQADataset(
            ssgvqa_root=args.ssgvqa_root,
            image_root=args.image_root,
            video_ids=[test_video_id],
            processor=model.processor,
            label2idx=LABEL2IDX,
            norm_label_map=NORM_LABEL_MAP,
            labels_list=SSGVQA_LABELS,
            num_frames=args.num_frames,
            prompt_mode="simple",
            max_samples=100,
        )
        
        dataset_loaded = len(dataset) > 0
        all_passed &= check_pass(f"Dataset loaded ({len(dataset)} samples from {test_video_id})", dataset_loaded)
        
        # Get one sample
        video, prompt, target, label_idx, vid, fid = dataset[0]
        logging.info(f"  Sample shape: {video.shape}")
        logging.info(f"  Prompt: {prompt[:80]}...")
        logging.info(f"  Target: {target}")
        logging.info(f"  Label idx: {label_idx}")
        
        correct_shape = video.shape == (args.num_frames, 3, 224, 224)
        all_passed &= check_pass(f"Video shape correct: {video.shape}", correct_shape)
        
        valid_label = 0 <= label_idx < NUM_CLASSES
        all_passed &= check_pass(f"Label idx valid: {label_idx}", valid_label)
        
    except Exception as e:
        logging.error(f"Dataset loading error: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    # =========================================================================
    # Check 6: Forward pass and loss
    # =========================================================================
    logging.info("\n[Check 6] Forward pass and loss...")
    
    try:
        from torch.utils.data import DataLoader
        
        loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=collate_ssgvqa_train,
        )
        
        batch = next(iter(loader))
        videos, prompts, targets, label_indices, vids, fids = batch
        
        model.train()
        
        # Tokenize
        full_sequences = [p + t for p, t in zip(prompts, targets)]
        tok = tokenizer(
            full_sequences,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt",
        )
        input_ids = tok["input_ids"].to(device)
        attention_mask = tok["attention_mask"].to(device)
        
        # Forward
        logits = model(
            video=videos.to(device),
            qa_inputs_ids=input_ids,
            qa_att_mask=attention_mask,
        )
        
        all_passed &= check_pass(f"Forward pass successful, logits shape: {logits.shape}", True)
        
        # Compute loss with masking
        labels = input_ids.clone()
        for idx, prompt in enumerate(prompts):
            prompt_tok = tokenizer(prompt, add_special_tokens=False)
            prompt_len = len(prompt_tok["input_ids"])
            labels[idx, :prompt_len] = -100
            labels[idx][input_ids[idx] == tokenizer.pad_token_id] = -100
        
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        
        loss_value = loss.item()
        logging.info(f"  Loss: {loss_value:.4f}")
        
        # Loss should be reasonable (not 0, not huge)
        loss_reasonable = 0.5 < loss_value < 10.0
        all_passed &= check_pass(f"Loss reasonable (0.5 < {loss_value:.4f} < 10.0)", loss_reasonable)
        
        # Check supervised tokens
        supervised = (shift_labels != -100).sum().item()
        logging.info(f"  Supervised tokens: {supervised}")
        
        has_supervised = supervised > 0
        all_passed &= check_pass(f"Has supervised tokens: {supervised}", has_supervised)
        
        # Backward pass
        loss.backward()
        all_passed &= check_pass("Backward pass successful", True)
        
        # Check gradients exist
        has_grads = False
        for name, p in model.named_parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grads = True
                break
        all_passed &= check_pass("Gradients computed", has_grads)
        
    except Exception as e:
        logging.error(f"Forward/loss error: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    # =========================================================================
    # Final Summary
    # =========================================================================
    logging.info("\n" + "=" * 60)
    if all_passed:
        logging.info("ALL CHECKS PASSED! Ready to submit training job.")
        logging.info("=" * 60)
        logging.info("\nTo train, run:")
        logging.info("  cd /path/to/SurgViVQA_on_STSG_VQA")
        logging.info("  sbatch fine-tune/run_finetune.sh")
        return 0
    else:
        logging.error("SOME CHECKS FAILED! Fix issues before training.")
        logging.info("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
