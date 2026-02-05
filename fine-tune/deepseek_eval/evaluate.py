#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main evaluation script: DeepSeek LLM-as-a-judge for SurgViVQA SSGVQA predictions.

This script:
1. Reads predictions from JSONL (output from utils/eval_surgvivqa_ssgvqa.py).
2. Uses DeepSeek API to judge semantic correctness (supporting multi-label GT).
3. Outputs judged JSONL and aggregated metrics JSON.

Input format (each line):
{
  "video": "...",
  "frame_id": "...",
  "image_path": "...",
  "question": "...",
  "gt_answer": "label_a, label_b, ...",  # may be comma-separated
  "pred_label": "some_label",
  "top5": [{"label": "...", "score": ...}, ...],
  "prompt_mode": "simple|choices",
  ...
}

Output judged JSONL (each line adds):
{
  ...original fields...,
  "mapped_label": "...",
  "is_correct": true|false,
  "confidence": 0.0-1.0,
  "note": "..."
}

Output metrics JSON:
{
  "overall": {acc, mAP, mAR, mAF1, wF1, anymatch_acc, judge_acc, ...},
  "per_video": {...},
  "label_vocab": [...]
}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Add paths for imports (handle hyphenated 'fine-tune' folder)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_FINE_TUNE_DIR = os.path.dirname(_THIS_DIR)
_PROJECT_ROOT = os.path.dirname(_FINE_TUNE_DIR)

# Add both project root and fine-tune dir for flexible imports
for p in [_PROJECT_ROOT, _FINE_TUNE_DIR, _THIS_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Import from same directory (works regardless of how script is invoked)
from judge import (
    DeepSeekJudge,
    GlobalRateLimiter,
    JudgeConfig,
    SSGVQA_LABELS,
)
from metrics import MetricsAggregator


def make_key(rec: Dict[str, Any]) -> str:
    """Create unique key for a record to support resume functionality."""
    return f"{rec.get('video', '')}\t{rec.get('frame_id', '')}\t{rec.get('question', '')}"


def load_predictions(path: Path) -> List[Dict[str, Any]]:
    """Load predictions from JSONL file."""
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] Skipping malformed JSON line: {e}")
                continue
    return items


def load_existing_keys(path: Path) -> Set[str]:
    """Load keys of already-judged records for resume support."""
    keys: Set[str] = set()
    if not path.exists():
        return keys
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                keys.add(make_key(rec))
            except json.JSONDecodeError:
                continue
    return keys


def load_all_judged_records(path: Path) -> List[Dict[str, Any]]:
    """Load all judged records from JSONL file."""
    records: List[Dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def evaluate_metrics_from_file(judged_path: Path) -> Dict[str, Any]:
    """
    Compute metrics from a judged JSONL file.
    
    This function reads all judged records and aggregates metrics.
    """
    aggregator = MetricsAggregator()
    
    records = load_all_judged_records(judged_path)
    for rec in records:
        aggregator.add_record(
            video=str(rec.get("video", "")),
            gt_answer=str(rec.get("gt_answer", "")),
            pred_label=str(rec.get("pred_label", "")),
            mapped_label=str(rec.get("mapped_label", "UNKNOWN")),
            is_correct=bool(rec.get("is_correct", False)),
            frame_id=rec.get("frame_id", ""),
            question=rec.get("question", ""),
            confidence=rec.get("confidence", 0.0),
            note=rec.get("note", ""),
        )
    
    return aggregator.compute_all_metrics()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="DeepSeek LLM-as-a-judge evaluation for SurgViVQA SSGVQA predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python evaluate.py \\
    --predictions-jsonl /path/to/predictions.jsonl \\
    --output-judged-jsonl /path/to/judged.jsonl \\
    --output-metrics-json /path/to/metrics.json

  # Resume from interrupted run
  python evaluate.py \\
    --predictions-jsonl /path/to/predictions.jsonl \\
    --output-judged-jsonl /path/to/judged.jsonl \\
    --output-metrics-json /path/to/metrics.json \\
    --resume

  # Metrics-only mode (skip judging)
  python evaluate.py \\
    --predictions-jsonl /path/to/predictions.jsonl \\
    --output-judged-jsonl /path/to/judged.jsonl \\
    --output-metrics-json /path/to/metrics.json \\
    --metrics-only
""",
    )
    
    # Input/Output
    parser.add_argument(
        "--predictions-jsonl",
        required=True,
        help="Input predictions JSONL from eval_surgvivqa_ssgvqa.py",
    )
    parser.add_argument(
        "--output-judged-jsonl",
        required=True,
        help="Output JSONL with judge verdicts",
    )
    parser.add_argument(
        "--output-metrics-json",
        required=True,
        help="Output aggregated metrics JSON",
    )
    
    # API configuration
    parser.add_argument(
        "--base-url",
        default=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        help="DeepSeek API base URL (default: from DEEPSEEK_BASE_URL env or https://api.deepseek.com/v1)",
    )
    parser.add_argument(
        "--judge-model",
        default=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
        help="DeepSeek model name (default: deepseek-chat)",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=60.0,
        help="API request timeout in seconds (default: 60)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum API retry attempts (default: 5)",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.1,
        help="Minimum seconds between API requests (default: 0.1)",
    )
    
    # Execution control
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Process only first N samples (for debugging)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of concurrent workers (default: 1)",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=100,
        help="Flush output every N records (default: 100)",
    )
    
    # Modes
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing judged JSONL (skip already-judged records)",
    )
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Skip judging, only compute metrics from existing judged JSONL",
    )
    parser.add_argument(
        "--save-raw",
        action="store_true",
        help="Store raw judge response in output JSONL",
    )
    
    return parser.parse_args()


def judge_record(
    rec: Dict[str, Any],
    judge: DeepSeekJudge,
    save_raw: bool = False,
) -> Dict[str, Any]:
    """
    Judge a single prediction record.
    
    Args:
        rec: Input record with question, gt_answer, pred_label.
        judge: DeepSeek judge instance.
        save_raw: Whether to include raw judge response.
        
    Returns:
        Record augmented with judge verdicts.
    """
    question = str(rec.get("question", ""))
    gt_answer = str(rec.get("gt_answer", ""))
    pred_label = str(rec.get("pred_label", ""))
    
    # Call judge API
    parsed, raw = judge.judge(question, gt_answer, pred_label)
    
    # Extract judge outputs
    mapped_label = str(parsed.get("mapped_label", "UNKNOWN"))
    is_correct = bool(parsed.get("is_correct", False))
    confidence = float(parsed.get("confidence", 0.0))
    note = str(parsed.get("note", ""))
    
    # Build output record
    out = dict(rec)
    out.update({
        "mapped_label": mapped_label,
        "is_correct": is_correct,
        "confidence": confidence,
        "note": note,
    })
    
    if save_raw:
        out["judge_raw"] = raw
    
    return out


def main() -> None:
    """Main entry point."""
    args = parse_args()
    
    pred_path = Path(args.predictions_jsonl)
    out_judged = Path(args.output_judged_jsonl)
    out_metrics = Path(args.output_metrics_json)
    
    # Ensure output directories exist
    out_judged.parent.mkdir(parents=True, exist_ok=True)
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    
    # Metrics-only mode: just compute from existing judged file
    if args.metrics_only:
        if not out_judged.exists():
            raise SystemExit(f"[ERROR] --metrics-only requires existing judged file: {out_judged}")
        
        print(f"[INFO] Computing metrics from: {out_judged}")
        metrics = evaluate_metrics_from_file(out_judged)
        
        with out_metrics.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        print(f"[Done] Saved metrics to: {out_metrics}")
        return
    
    # Check API key
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise SystemExit(
            "[ERROR] DEEPSEEK_API_KEY environment variable is not set.\n"
            "Please export it before running:\n"
            "  export DEEPSEEK_API_KEY='your-api-key'"
        )
    
    # Load predictions
    if not pred_path.exists():
        raise SystemExit(f"[ERROR] Predictions file not found: {pred_path}")
    
    all_records = load_predictions(pred_path)
    total_records = len(all_records)
    print(f"[INFO] Loaded {total_records} prediction records from: {pred_path}")
    
    # Apply max_samples limit
    if args.max_samples is not None:
        all_records = all_records[: max(args.max_samples, 0)]
        print(f"[INFO] Limited to first {len(all_records)} samples (--max-samples)")
    
    # Load existing keys for resume
    existing_keys: Set[str] = set()
    if args.resume and out_judged.exists():
        existing_keys = load_existing_keys(out_judged)
        print(f"[INFO] Resume mode: found {len(existing_keys)} already-judged records")
    
    # Initialize judge
    cfg = JudgeConfig(
        base_url=args.base_url,
        model=args.judge_model,
        api_key=api_key,
        request_timeout=args.request_timeout,
        max_retries=args.max_retries,
        sleep_seconds=args.sleep_seconds,
    )
    limiter = GlobalRateLimiter(args.sleep_seconds)
    judge = DeepSeekJudge(cfg, limiter)
    
    # Determine file mode
    mode = "a" if args.resume and out_judged.exists() else "w"
    
    # Filter records to process
    to_process = [r for r in all_records if make_key(r) not in existing_keys]
    print(f"[INFO] Records to judge: {len(to_process)} (skipped {len(all_records) - len(to_process)})")
    
    if not to_process:
        print("[INFO] No new records to judge.")
    else:
        # Process records
        processed = 0
        with out_judged.open(mode, encoding="utf-8") as fout:
            if args.num_workers <= 1:
                # Sequential processing
                for i, rec in enumerate(to_process):
                    judged = judge_record(rec, judge, args.save_raw)
                    fout.write(json.dumps(judged, ensure_ascii=False) + "\n")
                    processed += 1
                    
                    if processed % args.save_every == 0:
                        fout.flush()
                        os.fsync(fout.fileno())
                        print(f"[Progress] {processed}/{len(to_process)} records judged")
            else:
                # Parallel processing
                with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                    futures = [
                        executor.submit(judge_record, r, judge, args.save_raw)
                        for r in to_process
                    ]
                    
                    for fut in as_completed(futures):
                        try:
                            judged = fut.result()
                            fout.write(json.dumps(judged, ensure_ascii=False) + "\n")
                            processed += 1
                            
                            if processed % args.save_every == 0:
                                fout.flush()
                                os.fsync(fout.fileno())
                                print(f"[Progress] {processed}/{len(to_process)} records judged")
                        except Exception as e:
                            print(f"[ERROR] Failed to judge record: {e}")
        
        print(f"[INFO] Judged {processed} records, saved to: {out_judged}")
    
    # Compute final metrics
    print("[INFO] Computing final metrics...")
    metrics = evaluate_metrics_from_file(out_judged)
    
    # Add metadata
    metrics["metadata"] = {
        "predictions_file": str(pred_path),
        "judged_file": str(out_judged),
        "total_input_records": total_records,
        "judge_model": args.judge_model,
        "base_url": args.base_url,
    }
    
    with out_metrics.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(json.dumps(metrics["overall"], ensure_ascii=False, indent=2))
    print("=" * 60)
    print(f"\n[Done] Saved judged JSONL to: {out_judged}")
    print(f"[Done] Saved metrics JSON to: {out_metrics}")


if __name__ == "__main__":
    main()
