#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Metrics computation for SurgViVQA closed-set VQA evaluation.

Provides the same metric definitions as utils/eval_surgvivqa_ssgvqa.py:
- Accuracy (acc)
- Mean Average Precision (mAP)
- Mean Average Recall (mAR)  
- Mean Average F1 (mAF1)
- Weighted F1 (wF1)

Additionally supports:
- Multi-label GT evaluation (any-match accuracy)
- Per-video metrics breakdown
- Judge-based correctness statistics
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

# Import label vocabulary
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_FINE_TUNE_DIR = os.path.dirname(_THIS_DIR)
_PROJECT_ROOT = os.path.dirname(_FINE_TUNE_DIR)

# Add paths for imports
for p in [_PROJECT_ROOT, _FINE_TUNE_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from labels import SSGVQA_LABELS, NUM_CLASSES
except ImportError:
    SSGVQA_LABELS: List[str] = [
        "0", "1", "10", "2", "3", "4", "5", "6", "7", "8", "9",
        "False", "True",
        "abdominal_wall_cavity", "adhesion", "anatomy", "aspirate", "bipolar",
        "blood_vessel", "blue", "brown", "clip", "clipper", "coagulate", "cut",
        "cystic_artery", "cystic_duct", "cystic_pedicle", "cystic_plate", "dissect",
        "fluid", "gallbladder", "grasp", "grasper", "gut", "hook", "instrument",
        "irrigate", "irrigator", "liver", "omentum", "pack", "peritoneum", "red",
        "retract", "scissors", "silver", "specimen_bag", "specimenbag", "white", "yellow",
    ]
    NUM_CLASSES = len(SSGVQA_LABELS)


def build_label_maps() -> Tuple[Dict[str, int], Dict[int, str]]:
    """Build label <-> index mappings."""
    label2idx = {lbl: i for i, lbl in enumerate(SSGVQA_LABELS)}
    idx2label = {i: lbl for i, lbl in enumerate(SSGVQA_LABELS)}
    return label2idx, idx2label


def normalize_text(text: str) -> str:
    """Normalize text for fuzzy matching."""
    if text is None:
        return ""
    s = str(text).strip().lower()
    s = s.replace("_", " ")
    s = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in s)
    s = " ".join(s.split())
    return s


def build_norm_label_map() -> Dict[str, int]:
    """Build normalized label -> index mapping for fuzzy matching."""
    return {normalize_text(lbl): i for i, lbl in enumerate(SSGVQA_LABELS)}


def map_label_to_idx(
    label: str,
    label2idx: Dict[str, int],
    norm_map: Dict[str, int],
) -> int:
    """
    Map a label string to its index.
    
    Returns:
        Label index (0 to NUM_CLASSES-1), or -1 if no match found.
    """
    if label is None:
        return -1
    
    raw = str(label).strip()
    if not raw:
        return -1
    
    # Exact match
    if raw in label2idx:
        return label2idx[raw]
    
    # Normalized match
    raw_norm = normalize_text(raw)
    if raw_norm in norm_map:
        return norm_map[raw_norm]
    
    # Boolean aliases
    if raw_norm in ("yes", "true", "y", "t") and "True" in label2idx:
        return label2idx["True"]
    if raw_norm in ("no", "false", "n", "f") and "False" in label2idx:
        return label2idx["False"]
    
    # Numeric extraction
    for token in raw_norm.split():
        if token.isdigit() and token in label2idx:
            return label2idx[token]
    
    # Substring match
    for lbl in SSGVQA_LABELS:
        lbl_norm = normalize_text(lbl)
        if lbl_norm and f" {lbl_norm} " in f" {raw_norm} ":
            return label2idx[lbl]
    
    return -1


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    num_classes: int = NUM_CLASSES,
) -> Dict[str, float]:
    """
    Compute standard closed-set classification metrics.
    
    Args:
        y_true: List of ground-truth label indices.
        y_pred: List of predicted label indices (-1 for unknown).
        num_classes: Number of classes in the label vocabulary.
        
    Returns:
        Dictionary with acc, mAP, mAR, mAF1, wF1.
    """
    if len(y_true) == 0:
        return {"acc": 0.0, "mAP": 0.0, "mAR": 0.0, "mAF1": 0.0, "wF1": 0.0}

    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64)

    # Overall accuracy
    acc = float((y_true_arr == y_pred_arr).mean())

    # Per-class metrics
    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []
    supports: List[int] = []

    for c in range(num_classes):
        tp = int(((y_true_arr == c) & (y_pred_arr == c)).sum())
        fp = int(((y_true_arr != c) & (y_pred_arr == c)).sum())
        fn = int(((y_true_arr == c) & (y_pred_arr != c)).sum())

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        supports.append(int((y_true_arr == c).sum()))

    mAP = float(np.mean(precisions))
    mAR = float(np.mean(recalls))
    mAF1 = float(np.mean(f1s))

    total = float(np.sum(supports))
    if total > 0:
        wF1 = float(np.sum(np.asarray(f1s) * np.asarray(supports)) / total)
    else:
        wF1 = 0.0

    return {"acc": acc, "mAP": mAP, "mAR": mAR, "mAF1": mAF1, "wF1": wF1}


def compute_anymatch_accuracy(
    gt_labels_list: List[str],
    pred_labels: List[str],
    label2idx: Dict[str, int],
    norm_map: Dict[str, int],
) -> Dict[str, float]:
    """
    Compute any-match accuracy for multi-label GT.
    
    A prediction is correct if it matches ANY of the comma-separated GT labels.
    
    Args:
        gt_labels_list: List of GT strings (may be comma-separated).
        pred_labels: List of predicted label strings.
        label2idx: Label -> index mapping.
        norm_map: Normalized label -> index mapping.
        
    Returns:
        Dictionary with anymatch_acc, anymatch_correct, anymatch_total.
    """
    correct = 0
    total = 0
    
    for gt_str, pred_str in zip(gt_labels_list, pred_labels):
        # Parse multi-label GT
        gt_candidates = [s.strip() for s in str(gt_str).split(",") if s.strip()]
        if not gt_candidates:
            continue
            
        total += 1
        
        # Map GT candidates to normalized labels
        gt_set = set()
        for candidate in gt_candidates:
            idx = map_label_to_idx(candidate, label2idx, norm_map)
            if idx != -1:
                gt_set.add(SSGVQA_LABELS[idx])
        
        # Map prediction
        pred_idx = map_label_to_idx(pred_str, label2idx, norm_map)
        pred_label = SSGVQA_LABELS[pred_idx] if pred_idx != -1 else ""
        
        if pred_label in gt_set:
            correct += 1
    
    anymatch_acc = float(correct) / float(max(1, total))
    
    return {
        "anymatch_acc": anymatch_acc,
        "anymatch_correct": correct,
        "anymatch_total": total,
    }


def compute_judge_based_metrics(
    judge_correct: List[bool],
    gt_labels_list: List[str],
    pred_labels: List[str],
    label2idx: Dict[str, int],
    norm_map: Dict[str, int],
) -> Dict[str, float]:
    """
    Compute metrics using judge's is_correct verdicts.
    
    This provides a semantic correctness measure independent of exact label matching.
    
    Args:
        judge_correct: List of judge is_correct verdicts.
        gt_labels_list: List of GT strings.
        pred_labels: List of predicted labels.
        label2idx: Label -> index mapping.
        norm_map: Normalized label -> index mapping.
        
    Returns:
        Dictionary with judge accuracy and consistency metrics.
    """
    total = len(judge_correct)
    if total == 0:
        return {
            "judge_acc": 0.0,
            "judge_correct_count": 0,
            "judge_total": 0,
            "judge_inconsistency_count": 0,
        }
    
    correct_count = sum(1 for c in judge_correct if c)
    judge_acc = float(correct_count) / float(total)
    
    # Check for inconsistencies: judge says correct but labels don't match
    inconsistency_count = 0
    for is_correct, gt_str, pred_str in zip(judge_correct, gt_labels_list, pred_labels):
        if not is_correct:
            continue
            
        gt_candidates = [s.strip() for s in str(gt_str).split(",") if s.strip()]
        gt_set = set()
        for candidate in gt_candidates:
            idx = map_label_to_idx(candidate, label2idx, norm_map)
            if idx != -1:
                gt_set.add(SSGVQA_LABELS[idx])
        
        pred_idx = map_label_to_idx(pred_str, label2idx, norm_map)
        pred_label = SSGVQA_LABELS[pred_idx] if pred_idx != -1 else ""
        
        if pred_label not in gt_set:
            inconsistency_count += 1
    
    return {
        "judge_acc": judge_acc,
        "judge_correct_count": correct_count,
        "judge_total": total,
        "judge_inconsistency_count": inconsistency_count,
    }


class MetricsAggregator:
    """
    Aggregator for computing metrics over prediction records.
    
    Supports:
    - Overall metrics (first-label and any-match)
    - Per-video metrics breakdown
    - Judge-based correctness statistics
    """

    def __init__(self) -> None:
        self.label2idx, self.idx2label = build_label_maps()
        self.norm_map = build_norm_label_map()
        
        # Accumulate data
        self.records: List[Dict] = []
        self.y_true: List[int] = []  # First GT label indices
        self.y_pred: List[int] = []  # Predicted label indices
        self.gt_labels_list: List[str] = []  # Raw GT strings
        self.pred_labels: List[str] = []  # Predicted labels
        self.judge_correct: List[bool] = []  # Judge verdicts
        self.per_video: Dict[str, List[Dict]] = {}

    def add_record(
        self,
        video: str,
        gt_answer: str,
        pred_label: str,
        mapped_label: str,
        is_correct: bool,
        **kwargs,
    ) -> None:
        """
        Add a single judged record.
        
        Args:
            video: Video ID.
            gt_answer: Ground-truth answer (may be comma-separated).
            pred_label: Original predicted label from model.
            mapped_label: Judge-mapped label.
            is_correct: Judge's semantic correctness verdict.
            **kwargs: Additional fields to store.
        """
        # Parse GT labels (multi-label support)
        gt_candidates = [s.strip() for s in str(gt_answer).split(",") if s.strip()]
        
        # Map first GT label for single-label metrics
        gt_idx_first = -1
        for candidate in gt_candidates:
            idx = map_label_to_idx(candidate, self.label2idx, self.norm_map)
            if idx != -1:
                gt_idx_first = idx
                break
        
        # Map prediction
        # Use mapped_label if valid, otherwise fall back to pred_label
        effective_pred = mapped_label if mapped_label not in ("UNKNOWN", "") else pred_label
        pred_idx = map_label_to_idx(effective_pred, self.label2idx, self.norm_map)
        
        record = {
            "video": video,
            "gt_answer": gt_answer,
            "pred_label": pred_label,
            "mapped_label": mapped_label,
            "is_correct": is_correct,
            "gt_idx": gt_idx_first,
            "pred_idx": pred_idx,
            **kwargs,
        }
        
        self.records.append(record)
        
        # Only add to y_true/y_pred if GT is valid
        if gt_idx_first != -1:
            self.y_true.append(gt_idx_first)
            self.y_pred.append(pred_idx)
        
        self.gt_labels_list.append(gt_answer)
        self.pred_labels.append(effective_pred)
        self.judge_correct.append(is_correct)
        
        # Per-video tracking
        if video not in self.per_video:
            self.per_video[video] = []
        self.per_video[video].append(record)

    def compute_all_metrics(self) -> Dict[str, any]:
        """
        Compute all metrics.
        
        Returns:
            Dictionary with overall metrics, per-video metrics, and summary statistics.
        """
        # Standard closed-set metrics (single-label)
        overall = compute_metrics(self.y_true, self.y_pred, NUM_CLASSES)
        
        # Any-match metrics (multi-label)
        anymatch = compute_anymatch_accuracy(
            self.gt_labels_list, self.pred_labels,
            self.label2idx, self.norm_map
        )
        
        # Judge-based metrics
        judge_metrics = compute_judge_based_metrics(
            self.judge_correct, self.gt_labels_list, self.pred_labels,
            self.label2idx, self.norm_map
        )
        
        # Per-video metrics
        per_video_metrics: Dict[str, Dict[str, float]] = {}
        for video, video_records in sorted(self.per_video.items()):
            v_y_true = [r["gt_idx"] for r in video_records if r["gt_idx"] != -1]
            v_y_pred = [r["pred_idx"] for r in video_records if r["gt_idx"] != -1]
            v_metrics = compute_metrics(v_y_true, v_y_pred, NUM_CLASSES)
            
            v_gt_list = [r["gt_answer"] for r in video_records]
            v_pred_list = [r.get("mapped_label", r["pred_label"]) for r in video_records]
            v_anymatch = compute_anymatch_accuracy(
                v_gt_list, v_pred_list, self.label2idx, self.norm_map
            )
            
            v_judge = [r["is_correct"] for r in video_records]
            v_judge_acc = sum(v_judge) / max(1, len(v_judge))
            
            per_video_metrics[video] = {
                "n_samples": len(video_records),
                **v_metrics,
                **v_anymatch,
                "judge_acc": v_judge_acc,
            }
        
        # Unknown prediction stats
        unknown_gt_count = sum(1 for idx in self.y_true if idx == -1)
        unknown_pred_count = sum(1 for idx in self.y_pred if idx == -1)
        num_eval_samples = len(self.y_true)
        total_records = len(self.records)
        
        overall_out = {
            **overall,
            **anymatch,
            **judge_metrics,
            "num_samples": total_records,
            "num_eval_samples": num_eval_samples,
            "unknown_gt_count": unknown_gt_count,
            "unknown_gt_ratio": float(unknown_gt_count) / max(1, total_records),
            "unknown_pred_count": unknown_pred_count,
            "unknown_pred_ratio": float(unknown_pred_count) / max(1, num_eval_samples),
        }
        
        return {
            "overall": overall_out,
            "per_video": per_video_metrics,
            "label_vocab": SSGVQA_LABELS,
        }


if __name__ == "__main__":
    # Sanity check
    print(f"NUM_CLASSES: {NUM_CLASSES}")
    print(f"Sample labels: {SSGVQA_LABELS[:5]} ... {SSGVQA_LABELS[-5:]}")
    
    # Test metrics computation
    y_true = [0, 1, 2, 3, 4]
    y_pred = [0, 1, 2, 2, 4]
    metrics = compute_metrics(y_true, y_pred, num_classes=10)
    print(f"Test metrics: {metrics}")
