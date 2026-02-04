"""
Evaluate fine-tuned SurgViVQA on SSGVQA test set.

This is a wrapper script that calls the existing evaluation script
(utils/eval_surgvivqa_ssgvqa.py) with the correct parameters for
evaluating a fine-tuned checkpoint.

Usage:
    python eval_finetuned.py \
        --checkpoint fine-tune/ckpt/best_model.pth \
        --prompt-mode simple

The test split is FIXED: VID02, VID22, VID43, VID60, VID74
Metrics: Acc, mAP, mAR, mAF1, wF1 (same as SSGVQA-Net evaluation)
"""

import os
import sys
import argparse
import subprocess
import json


# Add project root to path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# Fixed test split (NEVER used for training)
TEST_VIDEOS = ["VID02", "VID22", "VID43", "VID60", "VID74"]


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned SurgViVQA on SSGVQA test set",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to fine-tuned checkpoint (.pth)"
    )
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
        "--prompt-mode", type=str, default="simple",
        choices=["simple", "choices"],
        help="Prompt mode (must match training)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Batch size"
    )
    parser.add_argument(
        "--num-frames", type=int, default=16,
        help="Number of frames"
    )
    parser.add_argument(
        "--max-input-tokens", type=int, default=256,
        help="Max input tokens for closed-set scoring"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: same as checkpoint dir)"
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of dataloader workers"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    
    return parser.parse_args()


def main():
    args = get_args()
    
    # Determine output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.checkpoint)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Output files
    log_file = os.path.join(args.output_dir, "eval_test.log")
    predictions_file = os.path.join(args.output_dir, "test_predictions.jsonl")
    
    # Build command for existing evaluation script
    eval_script = os.path.join(_PROJECT_ROOT, "utils", "eval_surgvivqa_ssgvqa.py")
    
    cmd = [
        sys.executable,
        eval_script,
        "--model-path", args.checkpoint,
        "--ssgvqa-root", args.ssgvqa_root,
        "--image-root", args.image_root,
        "--videos", *TEST_VIDEOS,
        "--prompt-mode", args.prompt_mode,
        "--batch-size", str(args.batch_size),
        "--num-frames", str(args.num_frames),
        "--max-input-tokens", str(args.max_input_tokens),
        "--workers", str(args.workers),
        "--seed", str(args.seed),
        "--log-file", log_file,
        "--predictions-file", predictions_file,
    ]
    
    print("=" * 80)
    print("Evaluating fine-tuned SurgViVQA on SSGVQA test set")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test videos: {TEST_VIDEOS}")
    print(f"Prompt mode: {args.prompt_mode}")
    print(f"Output dir: {args.output_dir}")
    print("=" * 80)
    print("\nRunning command:")
    print(" ".join(cmd))
    print("\n")
    
    # Run evaluation
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"\nEvaluation failed with return code {result.returncode}")
        sys.exit(result.returncode)
    
    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print(f"Log: {log_file}")
    print(f"Predictions: {predictions_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
