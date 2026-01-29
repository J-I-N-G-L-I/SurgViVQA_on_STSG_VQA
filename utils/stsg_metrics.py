#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ----------------------
# basic metrics utils
# ----------------------

def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b != 0 else 0.0


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


# ----------------------
# parsing helpers
# ----------------------

def _parse_choice_K_from_question(q: str) -> Optional[int]:
    """
    Parse choice option range K from question templates like:
      - "Answer with a number 1-5"
      - "Reply 1-3"
      - "Answer 1-4"
    Return K as int, or None.
    """
    if not q:
        return None
    qq = str(q).strip().lower()
    m = re.search(r"(?:answer|reply)\s*(?:with\s*(?:a\s*)?number\s*)?1\s*(?:\-|–|to)\s*(\d+)\b", qq)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def extract_final_answer(text: str) -> str:
    """
    Extract answer text from model output.
    Common patterns:
      - "Answer: xxx"
      - "Final answer: xxx"
      - "xxx"
    """
    if text is None:
        return ""
    s = str(text).strip()

    # take last line if multiple
    if "\n" in s:
        s = s.splitlines()[-1].strip()

    # strip common prefixes
    s_low = s.lower()
    for pref in ("final answer:", "answer:", "ans:", "output:", "prediction:"):
        if s_low.startswith(pref):
            s = s[len(pref) :].strip()
            break
    return s


def parse_bool_text(text: str) -> Optional[int]:
    if text is None:
        return None
    s = str(text).strip().lower()
    s = s.strip(".!?,")

    # strict
    if s in ("yes", "true", "y", "t"):
        return 1
    if s in ("no", "false", "n", "f"):
        return 0

    # tolerant: look for tokens
    if re.search(r"\b(yes|true)\b", s):
        return 1
    if re.search(r"\b(no|false)\b", s):
        return 0
    return None


def parse_bool_label(x: Any) -> Optional[int]:
    if x is None:
        return None
    if isinstance(x, bool):
        return 1 if x else 0
    s = str(x).strip().lower()
    if s in ("1", "true", "yes"):
        return 1
    if s in ("0", "false", "no"):
        return 0
    return None


def extract_first_int(text: str) -> Optional[int]:
    if text is None:
        return None
    m = re.search(r"[-+]?\b(\d+)\b", str(text))
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def extract_first_float(text: str) -> Optional[float]:
    if text is None:
        return None
    m = re.search(r"[-+]?\b(\d+(?:\.\d+)?)\b", str(text))
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def parse_numeric_gt(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip()
    try:
        return float(s)
    except Exception:
        # try extract
        return extract_first_float(s)


# ----------------------
# aggregator classes
# ----------------------

class BinAgg:
    def __init__(self):
        self.y_true: List[int] = []
        self.y_pred: List[int] = []
        self.y_score_pos: List[float] = []  # probability/score for positive class if available

    def add(self, gt: int, pred: int, score_pos: float):
        self.y_true.append(int(gt))
        self.y_pred.append(int(pred))
        self.y_score_pos.append(float(score_pos))

    def compute(self) -> Dict[str, Any]:
        if not self.y_true:
            return {"n": 0, "type": "bool", "acc": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "auroc": 0.0}

        y_true = np.array(self.y_true, dtype=np.int32)
        y_pred = np.array(self.y_pred, dtype=np.int32)
        acc = float(np.mean(y_true == y_pred))

        # precision/recall/f1 on positive class
        tp = float(np.sum((y_true == 1) & (y_pred == 1)))
        fp = float(np.sum((y_true == 0) & (y_pred == 1)))
        fn = float(np.sum((y_true == 1) & (y_pred == 0)))
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)

        # AUROC (if score exists)
        y_score = np.array(self.y_score_pos, dtype=np.float32)
        auroc = _binary_auroc(y_true, y_score)

        return {
            "n": int(len(y_true)),
            "type": "bool",
            "acc": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auroc": auroc,
        }


def _binary_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    # degenerate
    if y_true.size == 0:
        return 0.0
    if np.all(y_true == 0) or np.all(y_true == 1):
        return 0.5

    # rank-based AUROC
    order = np.argsort(y_score)
    y = y_true[order]
    n_pos = float(np.sum(y == 1))
    n_neg = float(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.5

    # compute rank sum for positives
    ranks = np.arange(1, y.size + 1, dtype=np.float64)
    rank_sum_pos = float(np.sum(ranks[y == 1]))
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


class CatAgg:
    def __init__(self):
        self.gt: List[int] = []
        self.pred: List[int] = []
        self.num_classes: int = 0

    def add(self, gt: int, pred: int):
        self.gt.append(int(gt))
        self.pred.append(int(pred))
        self.num_classes = max(self.num_classes, int(gt) + 1, int(pred) + 1)

    def compute(self) -> Dict[str, Any]:
        if not self.gt:
            return {"n": 0, "type": "cat", "top1_acc": 0.0, "top3_acc": 0.0, "macro_f1": 0.0}

        gt = np.array(self.gt, dtype=np.int32)
        pred = np.array(self.pred, dtype=np.int32)
        top1 = float(np.mean(gt == pred))

        # we don't have ranked list here; top3==top1
        top3 = top1

        macro_f1 = _macro_f1(gt, pred, self.num_classes)

        return {
            "n": int(len(gt)),
            "type": "cat",
            "top1_acc": top1,
            "top3_acc": top3,
            "macro_f1": macro_f1,
        }


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    if y_true.size == 0:
        return 0.0
    f1s: List[float] = []
    for c in range(num_classes):
        tp = float(np.sum((y_true == c) & (y_pred == c)))
        fp = float(np.sum((y_true != c) & (y_pred == c)))
        fn = float(np.sum((y_true == c) & (y_pred != c)))
        p = safe_div(tp, tp + fp)
        r = safe_div(tp, tp + fn)
        f1 = safe_div(2 * p * r, p + r)
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else 0.0


class SecAgg:
    def __init__(self):
        self.gt: List[float] = []
        self.pred: List[float] = []

    def add(self, gt: float, pred: float):
        self.gt.append(float(gt))
        self.pred.append(float(pred))

    def compute(self) -> Dict[str, Any]:
        if not self.gt:
            return {"n": 0, "type": "seconds", "mae": 0.0, "rmse": 0.0, "nmae_gt": 0.0, "within_tau": {}}

        gt = np.array(self.gt, dtype=np.float32)
        pred = np.array(self.pred, dtype=np.float32)
        err = np.abs(pred - gt)
        mae = float(np.mean(err))
        rmse = float(np.sqrt(np.mean((pred - gt) ** 2)))

        # normalized MAE by gt (avoid div by 0)
        denom = np.maximum(np.abs(gt), 1e-6)
        nmae = float(np.mean(err / denom))

        # within tolerance thresholds (seconds)
        taus = [1.0, 2.0, 5.0]
        within = {str(t): float(np.mean(err <= t)) for t in taus}

        return {
            "n": int(len(gt)),
            "type": "seconds",
            "mae": mae,
            "rmse": rmse,
            "nmae_gt": nmae,
            "within_tau": within,
        }


class CountAgg:
    def __init__(self):
        self.gt: List[int] = []
        self.pred: List[int] = []
        self.num_classes: int = 0

    def add(self, gt: int, pred: int):
        self.gt.append(int(gt))
        self.pred.append(int(pred))
        self.num_classes = max(self.num_classes, int(gt) + 1, int(pred) + 1)

    def compute(self) -> Dict[str, Any]:
        if not self.gt:
            return {"n": 0, "type": "count", "top1_acc": 0.0, "top3_acc": 0.0, "within1_acc": 0.0, "macro_f1": 0.0}

        gt = np.array(self.gt, dtype=np.int32)
        pred = np.array(self.pred, dtype=np.int32)
        top1 = float(np.mean(gt == pred))
        top3 = top1  # no ranked list
        within1 = float(np.mean(np.abs(gt - pred) <= 1))
        macro_f1 = _macro_f1(gt, pred, self.num_classes)

        return {
            "n": int(len(gt)),
            "type": "count",
            "top1_acc": top1,
            "top3_acc": top3,
            "within1_acc": within1,
            "macro_f1": macro_f1,
        }


# ----------------------
# vocab mapper
# ----------------------

class VocabMapper:
    """
    Load vocabs_train.json and provide mapping:
      category -> label string -> id
    """
    def __init__(self, vocabs_path: Path):
        with open(vocabs_path, "r", encoding="utf-8") as f:
            voc = json.load(f)

        # normalize keys
        self.voc: Dict[str, List[str]] = {}
        for k, arr in voc.items():
            if not isinstance(arr, list):
                continue
            kk = str(k).lower()
            self.voc[kk] = [str(x) for x in arr]

        self.id_map: Dict[str, Dict[str, int]] = {}
        for cat, labels in self.voc.items():
            self.id_map[cat] = {lbl: i for i, lbl in enumerate(labels)}

    def encode_gt(self, category: str, gt_label: str) -> int:
        cat = category.lower()
        labels = self.voc.get(cat, None)
        if labels is None:
            return 0
        if gt_label in self.id_map.get(cat, {}):
            return int(self.id_map[cat][gt_label])
        # unknown => 0 if <UNK> exists else 0
        if "<UNK>" in self.id_map.get(cat, {}):
            return int(self.id_map[cat]["<UNK>"])
        return 0

    def encode_pred(self, category: str, pred_label: str) -> int:
        return self.encode_gt(category, pred_label)


# ----------------------
# label type inference (Step b)
# ----------------------

def _infer_label_type(meta: Dict[str, Any], gt_answer: Any) -> str:
    """
    Robust label type inference for evaluation.

    Returns one of:
      {bool, count, duration, seconds, choice_index, text}

    Priority:
      1) meta.label_type / meta.answer_mode / meta.answer_type (with normalization)
      2) meta.choice_K / meta.choices / question template ("Answer ... 1-K") => choice_index
      3) category-based fallback (count/duration)
      4) gt_answer-based fallback (bool / numeric -> seconds / else text)
    """
    # ---- 1) explicit type fields (normalize) ----
    lt = (meta.get("label_type") or "").lower().strip()
    if not lt:
        lt = (meta.get("answer_mode") or "").lower().strip()
    if not lt:
        lt = (meta.get("answer_type") or "").lower().strip()

    if lt in ("binary", "boolean", "bin", "yesno", "yes/no"):
        lt = "bool"
    elif lt in ("choice", "mcq", "multi_choice", "multichoice", "option", "index"):
        lt = "choice_index"
    elif lt in ("numeric", "number", "float", "int", "reg", "sec", "time"):
        lt = "seconds"
    elif lt in ("cat", "text", "string", "phrase"):
        lt = "text"

    if lt in ("bool", "count", "duration", "seconds", "choice_index", "text"):
        return lt

    # ---- 2) choice detection from meta/question ----
    ck = meta.get("choice_K", None)
    if isinstance(ck, int) and ck > 0:
        return "choice_index"
    if isinstance(meta.get("choices", None), list) and len(meta.get("choices")) > 0:
        return "choice_index"

    q = ""
    raw = meta.get("qa_raw", None)
    if isinstance(raw, dict):
        q = raw.get("question") or ""
    if not q:
        q = meta.get("question") or ""

    Kq = _parse_choice_K_from_question(q)
    if isinstance(Kq, int) and Kq > 0:
        return "choice_index"

    # ---- 3) category-based fallback ----
    cat = (meta.get("category") or "").lower().strip()
    if cat == "phase":
        cat = "phase_transition"
    if cat == "count":
        return "count"
    if cat == "duration":
        return "duration"

    # ---- 4) gt_answer-based fallback ----
    if isinstance(gt_answer, bool):
        return "bool"
    if isinstance(gt_answer, (int, np.integer)) and int(gt_answer) in (0, 1) and cat in (
        "ordering", "boundary", "concurrency", "extreme", "phase_transition"
    ):
        # Avoid mis-treating 0/1 bool answers as seconds.
        return "bool"

    if isinstance(gt_answer, (int, float, np.integer, np.floating)):
        return "seconds"

    s = str(gt_answer).strip().lower().strip(".")
    if s in ("true", "false", "yes", "no"):
        return "bool"
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", s):
        # numeric but we failed to detect choice earlier => treat as seconds
        return "seconds"

    return "text"


# ----------------------
# evaluation main
# ----------------------

def evaluate_stsg(
    references: List[Any],
    predictions: List[str],
    metas: List[Dict[str, Any]],
    vocabs_json: Optional[str] = None,
) -> Dict[str, Any]:

    vocab_mapper: Optional[VocabMapper] = None
    if vocabs_json:
        vocab_mapper = VocabMapper(Path(vocabs_json))

    by_category: Dict[str, Dict[str, Any]] = {
        "boundary": {"bool": BinAgg(), "cat": CatAgg(), "seconds": SecAgg()},
        "concurrency": {"bool": BinAgg(), "seconds": SecAgg(), "choice": CatAgg()},
        "count": {"count": CountAgg()},
        "duration": {"duration": SecAgg()},
        "extreme": {"bool": BinAgg(), "cat": CatAgg(), "seconds": SecAgg()},
        "motion": {"cat": CatAgg()},
        "ordering": {"bool": BinAgg(), "cat": CatAgg(), "choice": CatAgg()},
        "phase_transition": {"bool": BinAgg(), "cat": CatAgg(), "seconds": SecAgg(), "choice": CatAgg()},
    }

    vocab_text_categories = {"boundary", "extreme", "motion", "ordering", "phase", "phase_transition"}

    task_hist: Dict[str, int] = {
        "count": 0,
        "duration": 0,
        "motion": 0,
        "ordering_text": 0,
        "ordering_bin": 0,
        "ordering_choice": 0,
        "extreme_text": 0,
        "extreme_reg": 0,
        "extreme_bin": 0,
        "concurrency_bin": 0,
        "concurrency_reg": 0,
        "concurrency_choice": 0,
        "boundary_bin": 0,
        "boundary_reg": 0,
        "boundary_text": 0,
        "phase_reg": 0,
        "phase_text": 0,
        "phase_bin": 0,
        "phase_choice": 0,
    }

    debug_info: Dict[str, int] = {
        "choice_total": 0,
        "choice_has_meta_K": 0,
        "choice_K_from_question": 0,
        "choice_missing_K": 0,
        "meta_missing_question": 0,
    }

    for gt_answer, pred_text_raw, meta in zip(references, predictions, metas):
        category = (meta.get("category") or "").lower()
        if category == "phase":
            category = "phase_transition"

        label_type = _infer_label_type(meta, gt_answer)
        pred_text = extract_final_answer(pred_text_raw)

        # ---- debug: question availability ----
        _q_dbg = ""
        _raw_dbg = meta.get("qa_raw", None)
        if isinstance(_raw_dbg, dict):
            _q_dbg = _raw_dbg.get("question") or ""
        if not _q_dbg:
            _q_dbg = meta.get("question") or ""
        if not _q_dbg:
            debug_info["meta_missing_question"] += 1

        def _canon_ordering_text(s: str) -> str:
            if not isinstance(s, str):
                s = str(s)
            x = s.strip().lower()

            # 常见同义词归一
            x = x.replace("prior to", "before").replace("earlier than", "before").replace("precede", "before")
            x = x.replace("later than", "after").replace("subsequent to", "after").replace("follow", "after")

            # 如果是 A after B -> 转成 B before A（尽量匹配你的 vocab）
            if " after " in x and " before " not in x:
                parts = x.split(" after ")
                if len(parts) == 2:
                    a, b = parts[0].strip(), parts[1].strip()
                    if a and b:
                        return f"{b} before {a}"
            return s

        # ---- BOOL ----
        if label_type == "bool":
            gt_bool = parse_bool_label(gt_answer)
            if gt_bool is None:
                continue
            pred_bool = parse_bool_text(pred_text)
            if pred_bool is None:
                pred_bool = 0
            score_pos = 1.0 if pred_bool == 1 else 0.0

            if category == "boundary":
                by_category["boundary"]["bool"].add(gt_bool, pred_bool, score_pos)
                task_hist["boundary_bin"] += 1
            elif category == "concurrency":
                by_category["concurrency"]["bool"].add(gt_bool, pred_bool, score_pos)
                task_hist["concurrency_bin"] += 1
            elif category == "ordering":
                by_category["ordering"]["bool"].add(gt_bool, pred_bool, score_pos)
                task_hist["ordering_bin"] += 1
            elif category == "phase_transition":
                by_category["phase_transition"]["bool"].add(gt_bool, pred_bool, score_pos)
                task_hist["phase_bin"] += 1
            elif category == "extreme":
                by_category["extreme"]["bool"].add(gt_bool, pred_bool, score_pos)
                task_hist["extreme_bin"] += 1
            continue

        # ---- COUNT ----
        if label_type == "count":
            gt_v = parse_numeric_gt(gt_answer)
            if gt_v is None:
                continue
            gt_i = int(round(gt_v))

            pred_i = extract_first_int(pred_text)
            if pred_i is None:
                pred_i = 0

            by_category["count"]["count"].add(gt_i, pred_i)
            task_hist["count"] += 1
            continue

        # ---- SECONDS (duration/reg) ----
        if label_type in ("seconds", "duration", "reg"):
            gt_v = parse_numeric_gt(gt_answer)
            if gt_v is None:
                continue
            pred_v = extract_first_float(pred_text)
            if pred_v is None:
                pred_v = 0.0

            if category == "duration":
                by_category["duration"]["duration"].add(gt_v, pred_v)
                task_hist["duration"] += 1
            elif category == "boundary":
                by_category["boundary"]["seconds"].add(gt_v, pred_v)
                task_hist["boundary_reg"] += 1
            elif category == "concurrency":
                by_category["concurrency"]["seconds"].add(gt_v, pred_v)
                task_hist["concurrency_reg"] += 1
            elif category == "extreme":
                by_category["extreme"]["seconds"].add(gt_v, pred_v)
                task_hist["extreme_reg"] += 1
            elif category == "phase_transition":
                by_category["phase_transition"]["seconds"].add(gt_v, pred_v)
                task_hist["phase_reg"] += 1
            continue

        # ---- CHOICE ----
        if label_type == "choice_index":
            gt_v = parse_numeric_gt(gt_answer)
            if gt_v is None:
                continue
            gt_idx = int(gt_v)

            pred_idx = extract_first_int(pred_text)
            if pred_idx is None:
                pred_idx = 1  # 默认当 1-based 输出

            K = meta.get("choice_K", None)
            if not isinstance(K, int) or K <= 0:
                q = ""
                raw = meta.get("qa_raw", None)
                if isinstance(raw, dict):
                    q = raw.get("question") or ""
                if not q:
                    q = meta.get("question") or ""
                Kq = _parse_choice_K_from_question(q)
                if isinstance(Kq, int) and Kq > 0:
                    K = Kq
                else:
                    choices = meta.get("choices", None)
                    if isinstance(choices, list) and len(choices) > 0:
                        K = len(choices)
            debug_info["choice_total"] += 1
            if isinstance(meta.get("choice_K", None), int) and meta.get("choice_K", None) > 0:
                debug_info["choice_has_meta_K"] += 1
            elif isinstance(_parse_choice_K_from_question(_q_dbg), int) and _parse_choice_K_from_question(_q_dbg) > 0:
                debug_info["choice_K_from_question"] += 1
            else:
                debug_info["choice_missing_K"] += 1

            # ---- 统一 GT/PRED 为 0-based ----
            gt_idx0 = gt_idx
            pred_idx0 = pred_idx

            if isinstance(K, int) and K > 0:
                # GT：如果在 1..K，视为 1-based；如果在 0..K-1，视为 0-based
                if 1 <= gt_idx <= K:
                    gt_idx0 = gt_idx - 1
                elif 0 <= gt_idx < K:
                    gt_idx0 = gt_idx
                else:
                    gt_idx0 = max(0, min(gt_idx, K - 1))

                # Pred：同理
                if 1 <= pred_idx <= K:
                    pred_idx0 = pred_idx - 1
                elif 0 <= pred_idx < K:
                    pred_idx0 = pred_idx
                else:
                    pred_idx0 = max(0, min(pred_idx, K - 1))
            else:
                # 没有 K 的兜底（一般不会发生，因为你已经能从 question 提取 K）
                if gt_idx >= 1:
                    gt_idx0 = gt_idx - 1
                if pred_idx >= 1:
                    pred_idx0 = pred_idx - 1

            if category == "ordering":
                by_category["ordering"]["choice"].add(int(gt_idx0), int(pred_idx0))
                task_hist["ordering_choice"] += 1
            elif category == "concurrency":
                by_category["concurrency"]["choice"].add(int(gt_idx0), int(pred_idx0))
                task_hist["concurrency_choice"] += 1
            elif category == "phase_transition":
                by_category["phase_transition"]["choice"].add(int(gt_idx0), int(pred_idx0))
                task_hist["phase_choice"] += 1
            continue

        # ---- TEXT ----
        gt_label_raw = str(gt_answer)
        pred_label_raw = pred_text

        if vocab_mapper is not None and category in vocab_text_categories:
            gt_id = vocab_mapper.encode_gt(category, gt_label_raw)
            pr_id = vocab_mapper.encode_pred(category, _canon_ordering_text(pred_label_raw) if category == "ordering" else pred_label_raw)
        else:
            # if no vocab, fallback exact match on strings (degenerate)
            gt_id = 0
            pr_id = 0
            if str(gt_label_raw).strip().lower() == str(pred_label_raw).strip().lower():
                pr_id = 0
            else:
                pr_id = 1

        if category == "motion":
            by_category["motion"]["cat"].add(gt_id, pr_id)
            task_hist["motion"] += 1
        elif category == "ordering":
            by_category["ordering"]["cat"].add(gt_id, pr_id)
            task_hist["ordering_text"] += 1
        elif category == "extreme":
            by_category["extreme"]["cat"].add(gt_id, pr_id)
            task_hist["extreme_text"] += 1
        elif category == "boundary":
            by_category["boundary"]["cat"].add(gt_id, pr_id)
            task_hist["boundary_text"] += 1
        elif category in ("phase", "phase_transition"):
            by_category["phase_transition"]["cat"].add(gt_id, pr_id)
            task_hist["phase_text"] += 1

    # compute
    bd_bool = by_category["boundary"]["bool"].compute()
    bd_cat = by_category["boundary"]["cat"].compute()
    bd_sec = by_category["boundary"]["seconds"].compute()

    cc_bool = by_category["concurrency"]["bool"].compute()
    cc_sec = by_category["concurrency"]["seconds"].compute()
    cc_choice = by_category["concurrency"]["choice"].compute()

    ct = by_category["count"]["count"].compute()

    dur = by_category["duration"]["duration"].compute()

    ex_bool = by_category["extreme"]["bool"].compute()
    ex_cat = by_category["extreme"]["cat"].compute()
    ex_sec = by_category["extreme"]["seconds"].compute()

    mo_cat = by_category["motion"]["cat"].compute()

    od_bool = by_category["ordering"]["bool"].compute()
    od_cat = by_category["ordering"]["cat"].compute()
    od_choice = by_category["ordering"]["choice"].compute()

    ph_bool = by_category["phase_transition"]["bool"].compute()
    ph_cat = by_category["phase_transition"]["cat"].compute()
    ph_sec = by_category["phase_transition"]["seconds"].compute()
    ph_choice = by_category["phase_transition"]["choice"].compute()

    # overall macro score: mean of key cells
    cell_scores: List[float] = []

    # bool: f1
    cell_scores.extend([bd_bool["f1"], cc_bool["f1"], od_bool["f1"], ph_bool["f1"], ex_bool["f1"]])

    # count/duration: within_tau or within1
    cell_scores.append(ct["within1_acc"])
    cell_scores.append(dur["within_tau"].get("2.0", 0.0))

    # choice: top1
    cell_scores.extend([od_choice["top1_acc"], cc_choice["top1_acc"], ph_choice["top1_acc"]])

    # text cat: top1 (motion/extreme/boundary/ordering/phase)
    cell_scores.extend([mo_cat["top1_acc"], ex_cat["top1_acc"], bd_cat["top1_acc"], od_cat["top1_acc"], ph_cat["top1_acc"]])

    result = {
        "task_hist": {k: int(v) for k, v in task_hist.items()},
        "by_category": {
            "boundary": {"bool": bd_bool, "cat": bd_cat, "seconds": bd_sec},
            "concurrency": {"bool": cc_bool, "seconds": cc_sec, "choice": cc_choice},
            "count": {"count": ct},
            "duration": {"duration": dur},
            "extreme": {"bool": ex_bool, "cat": ex_cat, "seconds": ex_sec},
            "motion": {"cat": mo_cat},
            "ordering": {"bool": od_bool, "cat": od_cat, "choice": od_choice},
            "phase_transition": {"bool": ph_bool, "cat": ph_cat, "seconds": ph_sec, "choice": ph_choice},
        },
        "overall_macro_score": float(np.mean(cell_scores)) if cell_scores else 0.0,
        "debug_info": {k: int(v) for k, v in debug_info.items()},
    }
    return result
