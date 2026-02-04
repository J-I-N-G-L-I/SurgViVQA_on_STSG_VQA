"""
SurgViVQA Fine-tuning on SSGVQA Dataset

This package provides the complete fine-tuning pipeline for training
SurgViVQA on SSGVQA's 52-class closed-set VQA task.

Modules:
- labels: 52-class label vocabulary and mapping utilities
- dataset: SSGVQA dataset loader for training
- trainer: Training utilities (SFT loss, checkpoint saving)
- train_ssgvqa: Main fine-tuning script

Usage:
    python -m fine-tune.train_ssgvqa --help
"""

from .labels import (
    SSGVQA_LABELS,
    NUM_CLASSES,
    LABEL2IDX,
    IDX2LABEL,
    NORM_LABEL_MAP,
    map_answer_to_label_idx,
)

from .dataset import (
    SSGVQADataset,
    SSGVQASample,
    collate_ssgvqa_train,
    load_split_videos,
    create_dataloaders,
)

__all__ = [
    # Labels
    "SSGVQA_LABELS",
    "NUM_CLASSES",
    "LABEL2IDX",
    "IDX2LABEL",
    "NORM_LABEL_MAP",
    "map_answer_to_label_idx",
    # Dataset
    "SSGVQADataset",
    "SSGVQASample",
    "collate_ssgvqa_train",
    "load_split_videos",
    "create_dataloaders",
]
