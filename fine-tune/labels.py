"""
SSGVQA 51-class label vocabulary for closed-set VQA.

This file defines the EXACT label space used by SSGVQA-Net and the SurgViVQA evaluation.
The order MUST remain unchanged to ensure consistent label indexing.

Total: 51 labels (indices 0-50)

Label categories:
- Numeric (0-10): counting answers (11 labels)
- Boolean (False, True): existence/yes-no answers (2 labels)
- Anatomical structures: various surgical anatomy terms
- Actions: surgical actions (aspirate, clip, coagulate, cut, dissect, etc.)
- Instruments: surgical tools (bipolar, clipper, grasper, hook, etc.)
- Colors: visual attributes (blue, brown, red, silver, white, yellow)
- Other: instrument, gut, omentum, peritoneum, fluid, etc.
"""

from typing import List, Dict, Tuple


# ============================================================================
# 52-class SSGVQA label vocabulary (MUST match evaluation script exactly)
# ============================================================================

SSGVQA_LABELS: List[str] = [
    "0",
    "1",
    "10",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "False",
    "True",
    "abdominal_wall_cavity",
    "adhesion",
    "anatomy",
    "aspirate",
    "bipolar",
    "blood_vessel",
    "blue",
    "brown",
    "clip",
    "clipper",
    "coagulate",
    "cut",
    "cystic_artery",
    "cystic_duct",
    "cystic_pedicle",
    "cystic_plate",
    "dissect",
    "fluid",
    "gallbladder",
    "grasp",
    "grasper",
    "gut",
    "hook",
    "instrument",
    "irrigate",
    "irrigator",
    "liver",
    "omentum",
    "pack",
    "peritoneum",
    "red",
    "retract",
    "scissors",
    "silver",
    "specimen_bag",
    "specimenbag",
    "white",
    "yellow",
]

NUM_CLASSES: int = len(SSGVQA_LABELS)  # 51 unique labels (note: "specimen_bag" and "specimenbag" are separate)

# ============================================================================
# Label mapping utilities
# ============================================================================

def build_label2idx() -> Dict[str, int]:
    """Build label string -> index mapping."""
    return {lbl: i for i, lbl in enumerate(SSGVQA_LABELS)}


def build_idx2label() -> Dict[int, str]:
    """Build index -> label string mapping."""
    return {i: lbl for i, lbl in enumerate(SSGVQA_LABELS)}


def normalize_text(text: str) -> str:
    """Normalize text for fuzzy matching (lowercase, replace underscores, remove punctuation)."""
    if text is None:
        return ""
    s = str(text).strip().lower()
    s = s.replace("_", " ")
    s = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in s)
    s = " ".join(s.split())
    return s


def build_norm_label_map() -> Dict[str, int]:
    """Build normalized label string -> index mapping for fuzzy matching."""
    return {normalize_text(lbl): i for i, lbl in enumerate(SSGVQA_LABELS)}


def map_answer_to_label_idx(
    answer: str,
    label2idx: Dict[str, int],
    norm_map: Dict[str, int],
) -> int:
    """
    Map a raw answer string to the label index.
    
    Args:
        answer: Raw answer string from dataset
        label2idx: Exact label -> index mapping
        norm_map: Normalized label -> index mapping
        
    Returns:
        Label index (0 to NUM_CLASSES-1), or -1 if no match found
    """
    if answer is None:
        return -1
    
    raw = str(answer).strip()
    if not raw:
        return -1
    
    # 1) Exact match
    if raw in label2idx:
        return label2idx[raw]
    
    # 2) Normalized match
    raw_norm = normalize_text(raw)
    if raw_norm in norm_map:
        return norm_map[raw_norm]
    
    # 3) Boolean aliases
    if raw_norm in ("yes", "true", "y", "t") and "True" in label2idx:
        return label2idx["True"]
    if raw_norm in ("no", "false", "n", "f") and "False" in label2idx:
        return label2idx["False"]
    
    # 4) Try to extract first digit for numeric answers
    for token in raw_norm.split():
        if token.isdigit() and token in label2idx:
            return label2idx[token]
    
    # 5) Substring match (last resort)
    for lbl in SSGVQA_LABELS:
        lbl_norm = normalize_text(lbl)
        if lbl_norm and f" {lbl_norm} " in f" {raw_norm} ":
            return label2idx[lbl]
    
    return -1


# Pre-built mappings for convenience
LABEL2IDX = build_label2idx()
IDX2LABEL = build_idx2label()
NORM_LABEL_MAP = build_norm_label_map()


if __name__ == "__main__":
    # Quick sanity check
    print(f"Number of SSGVQA labels: {NUM_CLASSES}")
    print(f"Labels: {SSGVQA_LABELS}")
    
    # Test mapping
    test_answers = ["True", "false", "grasper", "5", "cystic_duct", "invalid_answer"]
    for ans in test_answers:
        idx = map_answer_to_label_idx(ans, LABEL2IDX, NORM_LABEL_MAP)
        lbl = IDX2LABEL.get(idx, "NOT_FOUND")
        print(f"  '{ans}' -> idx={idx}, label='{lbl}'")
