"""
SurgViVQA evaluation on SSGVQA static QA (closed-set classification).

Key points:
- Label vocabulary and metric definitions are aligned with the baseline SSGVQA-Net test.py.
- Supports label-only prompting (closed decoding) and robust label mapping for free-text outputs.
- Outputs JSONL predictions and JSON metrics with overall and per-video results.
"""

import argparse
import sys
import json
import logging
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import Counter

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType

# Ensure repo root is on PYTHONPATH when running as a script
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from models.model import SurgViVQA
from utils.inference import batch_decode


# -----------------------------
# Label vocabulary (baseline)
# -----------------------------

# IMPORTANT: This label order is copied from baseline utils/test.py (SSGVQA-Net).
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


# -----------------------------
# Utilities
# -----------------------------

def seed_everything(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    s = str(text).strip().lower()
    s = s.replace("_", " ")
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _word_boundary_match(needle: str, haystack: str) -> bool:
    if not needle or not haystack:
        return False
    pat = r"\b" + re.escape(needle) + r"\b"
    return re.search(pat, haystack) is not None


def extract_final_answer(text: str) -> str:
    if text is None:
        return ""
    s = str(text).strip()
    if "\n" in s:
        s = s.splitlines()[-1].strip()
    low = s.lower()
    for pref in ("final answer:", "answer:", "ans:", "output:", "prediction:"):
        if low.startswith(pref):
            s = s[len(pref) :].strip()
            break
    return s


def build_label_maps(labels: List[str]) -> Tuple[Dict[str, int], Dict[str, int]]:
    exact_map = {lbl: i for i, lbl in enumerate(labels)}
    norm_map = {normalize_text(lbl): i for i, lbl in enumerate(labels)}
    return exact_map, norm_map


def map_pred_to_label(
    pred_text: str,
    labels: List[str],
    exact_map: Dict[str, int],
    norm_map: Dict[str, int],
) -> Tuple[str, int, str]:
    """
    Map free-form prediction to closed-set label.
    Strategy: exact match -> boolean/number -> normalized match -> longest substring match.
        Returns (label, idx, match_type) where match_type in:
            {exact, synonym/inflection, substring, unknown}
    """
    if pred_text is None:
                return "", -1, "unknown"

    raw = extract_final_answer(pred_text).strip()
    if not raw:
        return "", -1, "unknown"

    # 1) exact match (case-sensitive) then case-insensitive
    if raw in exact_map:
        return raw, exact_map[raw], "exact"
    raw_norm = normalize_text(raw)
    if raw_norm in norm_map:
        return labels[norm_map[raw_norm]], norm_map[raw_norm], "synonym/inflection"

    # 2) boolean mapping
    if raw_norm in ("yes", "true", "y", "t"):
        if "True" in exact_map:
            return "True", exact_map["True"], "synonym/inflection"
    if raw_norm in ("no", "false", "n", "f"):
        if "False" in exact_map:
            return "False", exact_map["False"], "synonym/inflection"

    # 3) numeric mapping (first integer)
    m = re.search(r"\b(\d+)\b", raw_norm)
    if m:
        num_str = m.group(1)
        if num_str in exact_map:
            return num_str, exact_map[num_str], "synonym/inflection"

    # 4) substring match with strict word-boundary constraint
    best_idx = -1
    best_len = -1
    for i, lbl in enumerate(labels):
        lbl_norm = normalize_text(lbl)
        if not lbl_norm:
            continue
        hit = _word_boundary_match(lbl_norm, raw_norm)
        if hit and len(lbl_norm) > best_len:
            best_len = len(lbl_norm)
            best_idx = i

    if best_idx >= 0:
        return labels[best_idx], best_idx, "substring"

    return "", -1, "unknown"


def compute_metrics(y_true: List[int], y_pred: List[int], num_classes: int) -> Dict[str, float]:
    """
    Baseline-style metrics from predicted class indices only.
    mAP = macro-precision, mAR = macro-recall, mAf1 = macro-F1, wF1 = weighted-F1.
    """
    if len(y_true) == 0:
        return {
            "acc": 0.0,
            "mAP": 0.0,
            "mAR": 0.0,
            "mAF1": 0.0,
            "wF1": 0.0,
        }

    y_true_arr = np.array(y_true, dtype=np.int64)
    y_pred_arr = np.array(y_pred, dtype=np.int64)
    acc = float(np.mean(y_true_arr == y_pred_arr))

    precisions, recalls, f1s, supports = [], [], [], []
    for c in range(num_classes):
        tp = float(np.sum((y_true_arr == c) & (y_pred_arr == c)))
        fp = float(np.sum((y_true_arr != c) & (y_pred_arr == c)))
        fn = float(np.sum((y_true_arr == c) & (y_pred_arr != c)))
        support = float(np.sum(y_true_arr == c))

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0

        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        supports.append(support)

    mAP = float(np.mean(precisions))
    mAR = float(np.mean(recalls))
    mAF1 = float(np.mean(f1s))
    total_support = float(np.sum(supports))
    wF1 = float(np.sum(np.array(f1s) * np.array(supports)) / total_support) if total_support > 0 else 0.0

    return {
        "acc": acc,
        "mAP": mAP,
        "mAR": mAR,
        "mAF1": mAF1,
        "wF1": wF1,
    }


def _entropy_from_counts(counts: Dict[int, int]) -> float:
    total = float(sum(counts.values()))
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        p = float(c) / total
        ent -= p * np.log(p + 1e-12)
    return float(ent)


def _topk_from_counts(counts: Dict[int, int], labels: List[str], k: int = 20) -> List[Tuple[str, int]]:
    items = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:k]
    return [(labels[i], int(c)) for i, c in items]


def _ratios_from_counts(counts: Dict[int, int], topk: int = 5) -> Tuple[float, float]:
    total = float(sum(counts.values()))
    if total <= 0:
        return 0.0, 0.0
    sorted_counts = sorted(counts.values(), reverse=True)
    top1 = float(sorted_counts[0]) / total if sorted_counts else 0.0
    topk_ratio = float(sum(sorted_counts[:topk])) / total
    return top1, topk_ratio


def _question_type(q: str) -> str:
    ql = (q or "").strip().lower()
    if ql.startswith("are there") or ql.startswith("is there") or ql.startswith("there is"):
        return "yesno"
    if ql.startswith("how many"):
        return "counting"
    if any(tok in ql for tok in ["doing", "action", "performed", "performing", "doing?"]):
        return "action"

    # anatomy / instrument / color (keyword cues)
    color_words = {"blue", "brown", "red", "white", "yellow", "silver"}
    instrument_words = {"instrument", "grasper", "hook", "clipper", "scissors", "irrigator", "bipolar"}
    anatomy_words = {"anatomy", "liver", "gallbladder", "gut", "omentum", "peritoneum", "cystic"}

    if any(w in ql for w in color_words):
        return "color"
    if any(w in ql for w in instrument_words):
        return "instrument"
    if any(w in ql for w in anatomy_words):
        return "anatomy"

    return "other"


# -----------------------------
# Dataset
# -----------------------------

@dataclass
class SSGVQASample:
    video_id: str
    frame_id: str
    image_path: str
    question: str
    answer: str


class SSGVQAStaticDataset(Dataset):
    def __init__(
        self,
        ssgvqa_root: str,
        image_root: str,
        video_ids: List[str],
        processor=None,
        max_samples: Optional[int] = None,
        strict_missing_image: bool = False,
    ):
        self.ssgvqa_root = ssgvqa_root
        self.image_root = image_root
        self.video_ids = video_ids
        self.processor = processor
        self.strict_missing_image = strict_missing_image
        self._log_processor_once = True

        self.samples: List[SSGVQASample] = []
        for vid in self.video_ids:
            vid_dir = os.path.join(self.ssgvqa_root, vid)
            if not os.path.isdir(vid_dir):
                raise FileNotFoundError(f"Missing SSGVQA folder: {vid_dir}")

            txt_files = [f for f in os.listdir(vid_dir) if f.endswith(".txt")]
            txt_files = sorted(txt_files, key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else x)

            for fname in txt_files:
                frame_stem = os.path.splitext(fname)[0]
                if frame_stem.isdigit():
                    frame_id = f"{int(frame_stem):06d}"
                else:
                    frame_id = frame_stem.zfill(6)

                img_path = os.path.join(self.image_root, vid, f"{frame_id}.png")
                if not os.path.isfile(img_path):
                    if self.strict_missing_image:
                        raise FileNotFoundError(f"Missing image: {img_path}")
                    else:
                        continue

                qa_path = os.path.join(vid_dir, fname)
                with open(qa_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or "|" not in line:
                            continue
                        parts = [p.strip() for p in line.split("|")]
                        if len(parts) < 2:
                            continue
                        question = parts[0]
                        answer = parts[1]
                        if not question or not answer:
                            continue
                        self.samples.append(
                            SSGVQASample(
                                video_id=vid,
                                frame_id=frame_id,
                                image_path=img_path,
                                question=question,
                                answer=answer,
                            )
                        )

        if max_samples is not None:
            self.samples = self.samples[: max_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def _process_image(self, image: Image.Image) -> torch.Tensor:
        if self.processor is None:
            raise RuntimeError("Processor is required for SurgViVQA inference.")

        # Prefer keyword 'images' for AutoImageProcessor (VideoMAE uses image processor)
        try:
            processed = self.processor(images=image, return_tensors="pt")
        except TypeError:
            # Fallback: positional call for processors without 'images' kwarg
            processed = self.processor(image, return_tensors="pt")

        pixel_values = processed.get("pixel_values", None) if isinstance(processed, dict) else getattr(processed, "pixel_values", None)
        if pixel_values is None:
            raise RuntimeError("Processor output missing 'pixel_values'.")

        # Log processor class and tensor shape only once to avoid spam
        if self._log_processor_once:
            self._log_processor_once = False
            logging.info(
                "Processor: %s | pixel_values shape: %s",
                self.processor.__class__.__name__,
                tuple(pixel_values.shape),
            )

        return pixel_values.squeeze(0).cpu().float()

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = Image.open(sample.image_path).convert("RGB")
        video_tensor = self._process_image(image)  # [C, H, W]

        return video_tensor, sample.question, sample.answer, sample.video_id, sample.frame_id, sample.image_path


def collate_ssgvqa(batch):
    videos, questions, answers, vids, frames, img_paths = [], [], [], [], [], []
    for v, q, a, vid, fid, ip in batch:
        videos.append(v)
        questions.append(q)
        answers.append(a)
        vids.append(vid)
        frames.append(fid)
        img_paths.append(ip)
    videos = torch.stack(videos, dim=0)  # [B, T, 3, 224, 224]
    return videos, questions, answers, vids, frames, img_paths


# -----------------------------
# Evaluation
# -----------------------------

def load_checkpoint(model, ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    logging.info("[CKPT] loaded: %s", ckpt_path)
    logging.info("[CKPT] missing keys: %d", len(missing))
    logging.info("[CKPT] unexpected keys: %d", len(unexpected))


def evaluate(args) -> None:
    pred_dir = os.path.dirname(args.predictions_file)
    metrics_dir = os.path.dirname(args.save_metrics)
    if pred_dir:
        os.makedirs(pred_dir, exist_ok=True)
    if metrics_dir:
        os.makedirs(metrics_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- tokenizer / decoder ----
    decoder_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(decoder_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    decoder_model = AutoModelForCausalLM.from_pretrained(decoder_name)

    # LoRA config (inference_mode=True)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=True,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["c_attn", "c_proj"],
    )

    # ---- model ----
    model = SurgViVQA(
        device=device,
        tokenizer=tokenizer,
        decoder_model=decoder_model,
        peft_config=lora_config,
        num_frames=args.num_frames,
    ).to(device)

    load_checkpoint(model, args.model_path)

    dataset = SSGVQAStaticDataset(
        ssgvqa_root=args.ssgvqa_root,
        image_root=args.image_root,
        video_ids=args.videos,
        processor=model.processor,
        max_samples=args.max_samples,
        strict_missing_image=args.strict_missing_image,
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_ssgvqa,
        pin_memory=True,
    )

    logging.info("Total samples: %d", len(dataset))

    exact_map, norm_map = build_label_maps(SSGVQA_LABELS)
    closed_labels = {"ssgvqa": SSGVQA_LABELS}

    # Stats
    unknown_gt_count = 0
    unknown_pred_count = 0
    match_type_counts = Counter()
    pred_counts = Counter()
    gt_counts = Counter()
    confusion_counts = Counter()
    soft_correct = 0
    soft_total = 0
    topk_correct = 0
    topk_total = 0

    y_true: List[int] = []
    y_pred: List[int] = []
    per_video_records: Dict[str, List[Tuple[int, int]]] = {}
    by_qtype_records: Dict[str, List[Tuple[int, int]]] = {}

    with open(args.predictions_file, "w", encoding="utf-8") as fout:
        model.eval()
        with torch.no_grad():
            processed = 0
            for videos, questions, answers, vids, frames, img_paths in loader:
                if args.debug_first_n is not None and processed >= args.debug_first_n:
                    break

                if args.ablate_image:
                    videos = torch.zeros_like(videos)

                if args.ablate_text:
                    questions = [args.ablate_text_prompt for _ in questions]

                metas = [
                    {
                        "video_id": vid,
                        "category": "ssgvqa",
                        "label_type": "text",
                    }
                    for vid in vids
                ]

                # Closed decoding (label-only) or greedy/hybrid
                raw_preds = batch_decode(
                    images=videos,
                    questions=questions,
                    metas=metas,
                    model=model,
                    tokenizer=tokenizer,
                    max_prompt_len=args.max_prompt_len,
                    max_new_tokens=args.max_new_tokens,
                    device=device,
                    decode_mode=args.decode_mode,
                    closed_labels=closed_labels,
                    count_max=20,
                    closed_text_topk=50,
                    strict_answer_format=args.strict_answer_format,
                )

                for q, gt, vid, fid, ip, pred_raw in zip(
                    questions, answers, vids, frames, img_paths, raw_preds
                ):
                    gt_label = gt.strip()
                    gt_idx = exact_map.get(gt_label, -1)
                    if gt_idx == -1:
                        # try normalized match for GT as a safe fallback
                        gt_norm = normalize_text(gt_label)
                        gt_idx = norm_map.get(gt_norm, -1)

                    if gt_idx == -1:
                        unknown_gt_count += 1
                    else:
                        gt_counts[gt_idx] += 1

                    pred_label, pred_idx, match_type = map_pred_to_label(
                        pred_raw, SSGVQA_LABELS, exact_map, norm_map
                    )
                    if pred_idx == -1:
                        unknown_pred_count += 1
                    else:
                        pred_counts[pred_idx] += 1
                    match_type_counts[match_type] += 1

                    # Soft correctness: normalized pred text contains normalized GT label
                    if gt_idx != -1:
                        gt_norm = normalize_text(gt_label)
                        pred_norm = normalize_text(pred_raw)
                        if _word_boundary_match(gt_norm, pred_norm):
                            soft_correct += 1
                        soft_total += 1

                    # Top-k accuracy if multiple label mentions appear in pred text
                    pred_norm = normalize_text(pred_raw)
                    cand_labels = [
                        lbl for lbl in SSGVQA_LABELS if _word_boundary_match(normalize_text(lbl), pred_norm)
                    ]
                    if len(cand_labels) >= args.topk:
                        topk_total += 1
                        if gt_label in cand_labels[: args.topk]:
                            topk_correct += 1

                    # Store JSONL record
                    fout.write(
                        json.dumps(
                            {
                                "video": vid,
                                "frame_id": fid,
                                "image_path": ip,
                                "question": q,
                                "gt_answer": gt_label,
                                "gt_idx": gt_idx,
                                "raw_pred_text": pred_raw,
                                "mapped_pred_label": pred_label,
                                "mapped_pred_idx": pred_idx,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                    # Keep for metrics only if GT is valid; unknown preds are skipped but counted
                    if gt_idx != -1 and pred_idx != -1:
                        y_true.append(gt_idx)
                        y_pred.append(pred_idx)
                        per_video_records.setdefault(vid, []).append((gt_idx, pred_idx))
                        confusion_counts[(gt_idx, pred_idx)] += 1
                        qtype = _question_type(q)
                        by_qtype_records.setdefault(qtype, []).append((gt_idx, pred_idx))

                    processed += 1
                    if args.debug_first_n is not None and processed >= args.debug_first_n:
                        break

    # Overall metrics
    metrics = compute_metrics(y_true, y_pred, num_classes=len(SSGVQA_LABELS))

    # Per-video metrics
    per_video = {}
    for vid, pairs in per_video_records.items():
        if not pairs:
            per_video[vid] = compute_metrics([], [], num_classes=len(SSGVQA_LABELS))
            continue
        gt_list = [p[0] for p in pairs]
        pred_list = [p[1] for p in pairs]
        per_video[vid] = compute_metrics(gt_list, pred_list, num_classes=len(SSGVQA_LABELS))

    unknown_pred_ratio = float(unknown_pred_count) / float(max(1, len(dataset)))
    unknown_gt_ratio = float(unknown_gt_count) / float(max(1, len(dataset)))

    # Diagnostics
    pred_entropy = _entropy_from_counts(pred_counts)
    gt_entropy = _entropy_from_counts(gt_counts)
    pred_top1_ratio, pred_top5_ratio = _ratios_from_counts(pred_counts, topk=5)
    gt_top1_ratio, gt_top5_ratio = _ratios_from_counts(gt_counts, topk=5)

    pred_topk = _topk_from_counts(pred_counts, SSGVQA_LABELS, k=20)
    gt_topk = _topk_from_counts(gt_counts, SSGVQA_LABELS, k=20)

    confusion_topk = []
    for (gt_i, pr_i), cnt in confusion_counts.most_common(30):
        confusion_topk.append(
            {
                "gt_label": SSGVQA_LABELS[gt_i],
                "pred_label": SSGVQA_LABELS[pr_i],
                "count": int(cnt),
            }
        )

    by_question_type = {}
    for qtype, pairs in by_qtype_records.items():
        if not pairs:
            by_question_type[qtype] = {"n": 0, "acc": 0.0, "mAF1": 0.0}
            continue
        gt_list = [p[0] for p in pairs]
        pred_list = [p[1] for p in pairs]
        m = compute_metrics(gt_list, pred_list, num_classes=len(SSGVQA_LABELS))
        by_question_type[qtype] = {
            "n": int(len(pairs)),
            "acc": m["acc"],
            "mAF1": m["mAF1"],
        }

    soft_acc = float(soft_correct) / float(soft_total) if soft_total > 0 else 0.0
    topk_acc = float(topk_correct) / float(topk_total) if topk_total > 0 else None

    report = {
        "overall": {
            **metrics,
            "num_samples": int(len(dataset)),
            "num_eval_samples": int(len(y_true)),
            "unknown_gt_count": int(unknown_gt_count),
            "unknown_gt_ratio": unknown_gt_ratio,
            "unknown_pred_count": int(unknown_pred_count),
            "unknown_pred_ratio": unknown_pred_ratio,
            "soft_acc": soft_acc,
            "topk_acc": topk_acc,
        },
        "per_video": per_video,
        "diagnostics": {
            "pred_topk": pred_topk,
            "gt_topk": gt_topk,
            "pred_entropy": pred_entropy,
            "gt_entropy": gt_entropy,
            "top1_ratio": pred_top1_ratio,
            "top5_ratio": pred_top5_ratio,
            "gt_top1_ratio": gt_top1_ratio,
            "gt_top5_ratio": gt_top5_ratio,
            "match_type_counts": dict(match_type_counts),
            "confusion_topk": confusion_topk,
            "by_question_type": by_question_type,
            "soft_acc": soft_acc,
            "topk_acc": topk_acc,
            "ablate_image": bool(args.ablate_image),
            "ablate_text": bool(args.ablate_text),
        },
        "label_vocab": SSGVQA_LABELS,
    }

    with open(args.save_metrics, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Console summary
    logging.info(
        "Overall: Acc=%.6f | mAP=%.6f | mAR=%.6f | mAF1=%.6f | wF1=%.6f",
        metrics["acc"],
        metrics["mAP"],
        metrics["mAR"],
        metrics["mAF1"],
        metrics["wF1"],
    )
    logging.info("Total samples: %d", len(dataset))
    logging.info("Evaluated samples: %d", len(y_true))
    logging.info("unknown_gt_count: %d", unknown_gt_count)
    logging.info("unknown_pred_count: %d", unknown_pred_count)
    logging.info("unknown_pred_ratio: %.6f", unknown_pred_ratio)
    logging.info("Pred top20 distribution: %s", pred_topk)
    logging.info("GT top20 distribution: %s", gt_topk)
    logging.info("Match type distribution: %s", dict(match_type_counts))
    logging.info("Most common confusions (top30): %s", confusion_topk)
    logging.info("By question type: %s", by_question_type)
    logging.info("Saved predictions: %s", args.predictions_file)
    logging.info("Saved metrics: %s", args.save_metrics)


def build_logger(log_file: str) -> None:
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("SurgViVQA evaluation on SSGVQA static QA")

    parser.add_argument("--model-path", required=True, help="Path to SurgViVQA checkpoint (.pth)")
    parser.add_argument("--ssgvqa-root", required=True, help="Root of SSGVQA static QA txt files")
    parser.add_argument("--image-root", required=True, help="Root of CholecT45 frames")
    parser.add_argument("--videos", nargs="+", required=True, help="Video IDs to evaluate")

    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--num-frames",
        type=int,
        default=16,
        help="VideoMAE expects multiple frames; single frames are repeated to this length.",
    )

    parser.add_argument("--max-prompt-len", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument(
        "--decode-mode",
        type=str,
        default="closed",
        choices=["closed", "greedy", "hybrid"],
        help="closed=label-only; hybrid=greedy then fallback to closed; greedy=raw generation",
    )
    parser.add_argument("--strict-answer-format", action="store_true", help="Use strict output format block")

    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--strict-missing-image", action="store_true")

    # Diagnostics / ablations
    parser.add_argument("--debug-first-n", type=int, default=None, help="Only process first N samples for quick diagnostics")
    parser.add_argument("--ablate-image", action="store_true", help="Replace input images with zeros")
    parser.add_argument("--ablate-text", action="store_true", help="Replace questions with a fixed prompt")
    parser.add_argument("--ablate-text-prompt", type=str, default="What is shown?", help="Fixed question for text ablation")
    parser.add_argument("--topk", type=int, default=3, help="Top-k for optional multi-label accuracy")

    parser.add_argument("--log-file", required=True)
    parser.add_argument("--predictions-file", required=True)
    parser.add_argument("--save-metrics", required=True)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    build_logger(args.log_file)
    seed_everything(args.seed)
    evaluate(args)