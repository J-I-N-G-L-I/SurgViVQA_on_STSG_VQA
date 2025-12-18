#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# basic metrics utils

def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b != 0 else 0.0


def bin_acc(y_true: List[int], y_pred: List[int]) -> float:
    if not y_true:
        return 0.0
    return float(np.mean(np.array(y_true) == np.array(y_pred)))


def precision_recall_f1(y_true: List[int], y_pred: List[int]) -> Tuple[float, float, float]:
    if not y_true:
        return 0.0, 0.0, 0.0
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tp = float(((y_true == 1) & (y_pred == 1)).sum())
    fp = float(((y_true == 0) & (y_pred == 1)).sum())
    fn = float(((y_true == 1) & (y_pred == 0)).sum())
    prec = safe_div(tp, tp + fp)
    rec = safe_div(tp, tp + fn)
    f1 = safe_div(2 * prec * rec, prec + rec)
    return prec, rec, f1


def auroc_from_scores(y_true: List[int], y_score: List[float]) -> float:
    if not y_true:
        return 0.5
    y_true = np.array(y_true, dtype=np.int32)
    y_score = np.array(y_score, dtype=np.float32)

    if y_true.min() == y_true.max():
        return 0.5

    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    score_sorted = y_score[order]

    P = float((y_true == 1).sum())
    N = float((y_true == 0).sum())

    tp_cum = 0.0
    fp_cum = 0.0
    prev = None
    auc = 0.0
    tpr_prev = 0.0
    fpr_prev = 0.0

    for yi, si in zip(y_sorted, score_sorted):
        if prev is None:
            prev = si
        if si != prev:
            tpr = safe_div(tp_cum, P)
            fpr = safe_div(fp_cum, N)
            auc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2.0
            tpr_prev = tpr
            fpr_prev = fpr
            prev = si
        if yi == 1:
            tp_cum += 1.0
        else:
            fp_cum += 1.0

    tpr = safe_div(tp_cum, P)
    fpr = safe_div(fp_cum, N)
    auc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2.0
    return float(auc)


# ----------------------
# parsing utilities
# ----------------------

def extract_final_answer(text: Any) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    s = text

    if "FINAL_ANSWER:" in s:
        s = s.split("FINAL_ANSWER:", 1)[-1]

    s = s.strip()
    s = re.sub(r"^[\s:;\-\>\(\)\[\]\{\}]+", "", s)
    s = s.strip()

    # Keep only first line (many models append extra stuff)
    if "\n" in s:
        s = s.split("\n", 1)[0].strip()

    # Remove trailing punctuation
    s = s.strip().strip(" .,:;")
    return s


def parse_bool_label(x: Any) -> Optional[int]:
    if x is None:
        return None
    if isinstance(x, bool):
        return 1 if x else 0
    s = str(x).strip().lower()
    if s in ("1", "true", "yes", "y", "t"):
        return 1
    if s in ("0", "false", "no", "n", "f"):
        return 0
    return None


def parse_bool_text(pred: str) -> Optional[int]:
    if pred is None:
        return None
    s = str(pred).strip().lower()
    if any(k in s for k in ["yes", "true"]):
        return 1
    if any(k in s for k in ["no", "false"]):
        return 0
    if re.search(r"\b1\b", s):
        return 1
    if re.search(r"\b0\b", s):
        return 0
    return None


def extract_first_int(pred: str) -> Optional[int]:
    if pred is None:
        return None
    m = re.search(r"[-+]?\b(\d+)\b", str(pred))
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def extract_first_number(pred: str) -> Optional[float]:
    if pred is None:
        return None
    m = re.search(r"[-+]?\d+(?:\.\d+)?", str(pred))
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def parse_numeric_gt(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def normalize_for_vocab_text(s: Any) -> str:
    """
    Normalize text for vocab lookup / fuzzy matching.
    - lower
    - replace separators
    - remove punctuation
    - collapse spaces
    """
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    x = s.strip().lower()
    x = x.replace("-", " ").replace("_", " ")
    x = re.sub(r"[^a-z0-9\s]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def canon_ordering_text(s: Any) -> str:
    """
    Canonicalize ordering text into the vocab-preferred form: 'A before B'.

    - Maps common synonyms to 'before' / 'after'
    - Converts 'A after B' -> 'B before A'
    - Normalizes whitespace
    """
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)

    x = s.strip().lower()
    x = re.sub(r"\s+", " ", x)

    x = x.replace("prior to", "before").replace("earlier than", "before").replace("precede", "before")
    x = x.replace("later than", "after").replace("subsequent to", "after").replace("follow", "after")

    if " after " in x and " before " not in x:
        parts = x.split(" after ")
        if len(parts) == 2:
            a, b = parts[0].strip(), parts[1].strip()
            if a and b:
                x = f"{b} before {a}"

    return x


# ----------------------
# label encoder & vocab mapper
# ----------------------

class LabelEncoder:
    def __init__(self):
        self._map: Dict[str, int] = {}
        self._next = 0

    def encode(self, s: str) -> int:
        if s not in self._map:
            self._map[s] = self._next
            self._next += 1
        return self._map[s]


class VocabMapper:
    def __init__(self, vocabs_json: str):
        with open(vocabs_json, "r", encoding="utf-8") as f:
            voc = json.load(f)

        self.vocabs: Dict[str, List[str]] = {}
        self.norm2id: Dict[str, Dict[str, int]] = {}
        self.id2norm: Dict[str, Dict[int, str]] = {}
        self.label_tokens: Dict[str, List[set]] = {}

        for k, arr in voc.items():
            if not isinstance(arr, list):
                continue
            cat = str(k).lower()
            labels = [str(x) for x in arr]
            self.vocabs[cat] = labels

            m = {}
            r = {}
            tok_sets = []
            for i, lab in enumerate(labels):
                nl = normalize_for_vocab_text(lab)
                m[nl] = i
                r[i] = nl
                tok_sets.append(set(nl.split()))
            self.norm2id[cat] = m
            self.id2norm[cat] = r
            self.label_tokens[cat] = tok_sets

    def encode_gt(self, category: str, gt: str) -> int:
        cat = category.lower()
        nl = normalize_for_vocab_text(gt)
        return int(self.norm2id.get(cat, {}).get(nl, 0))

    def encode_pred(self, category: str, pred: str) -> int:
        cat = category.lower()
        nl = normalize_for_vocab_text(pred)
        m = self.norm2id.get(cat, {})
        if nl in m:
            return int(m[nl])

        # fuzzy matching (token coverage / jaccard)
        pred_toks = set(nl.split())
        if not pred_toks:
            return 0

        tok_sets = self.label_tokens.get(cat, [])
        best_i = -1
        best_jacc = -1.0

        for i, lab_toks in enumerate(tok_sets):
            if not lab_toks:
                continue
            cover = safe_div(len(lab_toks & pred_toks), len(lab_toks))
            if cover >= 1.0:
                return i
            inter = len(lab_toks & pred_toks)
            union = len(lab_toks | pred_toks)
            jacc = safe_div(inter, union)
            if jacc > best_jacc:
                best_jacc = jacc
                best_i = i

        thr = 0.4 if cat == "ordering" else 0.5
        if best_i >= 0 and best_jacc >= thr:
            return int(best_i)
        return 0


# ----------------------
# aggregators
# ----------------------

class BinAgg:
    def __init__(self):
        self.y_true: List[int] = []
        self.y_pred: List[int] = []
        self.y_score: List[float] = []

    def add(self, gt: int, pred: int, score_pos: float):
        self.y_true.append(int(gt))
        self.y_pred.append(int(pred))
        self.y_score.append(float(score_pos))

    def summarize(self) -> Dict[str, Any]:
        acc = bin_acc(self.y_true, self.y_pred)
        prec, rec, f1 = precision_recall_f1(self.y_true, self.y_pred)
        auroc = auroc_from_scores(self.y_true, self.y_score)
        return {"acc": acc, "precision": prec, "recall": rec, "f1": f1, "auroc": auroc, "n": len(self.y_true), "type": "bool"}


class CatAgg:
    def __init__(self):
        self.y_true: List[int] = []
        self.y_pred: List[int] = []

    def add(self, gt: int, pred: int):
        self.y_true.append(int(gt))
        self.y_pred.append(int(pred))

    def summarize(self) -> Dict[str, Any]:
        if not self.y_true:
            return {"top1_acc": 0.0, "top3_acc": 0.0, "macro_f1": 0.0, "n": 0, "type": "cat"}
        y_true = np.array(self.y_true)
        y_pred = np.array(self.y_pred)
        top1 = float(np.mean(y_true == y_pred))

        # macro-f1
        labels = np.unique(np.concatenate([y_true, y_pred]))
        f1s = []
        for lab in labels:
            yt = (y_true == lab).astype(np.int32)
            yp = (y_pred == lab).astype(np.int32)
            tp = float(((yt == 1) & (yp == 1)).sum())
            fp = float(((yt == 0) & (yp == 1)).sum())
            fn = float(((yt == 1) & (yp == 0)).sum())
            prec = safe_div(tp, tp + fp)
            rec = safe_div(tp, tp + fn)
            f1 = safe_div(2 * prec * rec, prec + rec)
            f1s.append(f1)
        macro_f1 = float(np.mean(f1s)) if f1s else 0.0

        return {"top1_acc": top1, "top3_acc": top1, "macro_f1": macro_f1, "n": len(self.y_true), "type": "cat"}


class CountAgg:
    def __init__(self):
        self.gts: List[int] = []
        self.preds: List[int] = []

    def add(self, gt: int, pred: int):
        self.gts.append(int(gt))
        self.preds.append(int(pred))

    def summarize(self) -> Dict[str, Any]:
        if not self.gts:
            return {"top1_acc": 0.0, "within1_acc": 0.0, "macro_f1": 0.0, "top3_acc": 0.0, "n": 0, "type": "count"}
        gt = np.array(self.gts)
        pr = np.array(self.preds)
        top1 = float(np.mean(gt == pr))
        within1 = float(np.mean(np.abs(gt - pr) <= 1))

        labels = np.unique(np.concatenate([gt, pr]))
        f1s = []
        for lab in labels:
            yt = (gt == lab).astype(np.int32)
            yp = (pr == lab).astype(np.int32)
            tp = float(((yt == 1) & (yp == 1)).sum())
            fp = float(((yt == 0) & (yp == 1)).sum())
            fn = float(((yt == 1) & (yp == 0)).sum())
            prec = safe_div(tp, tp + fp)
            rec = safe_div(tp, tp + fn)
            f1 = safe_div(2 * prec * rec, prec + rec)
            f1s.append(f1)
        macro_f1 = float(np.mean(f1s)) if f1s else 0.0

        return {"top1_acc": top1, "within1_acc": within1, "macro_f1": macro_f1, "top3_acc": top1, "n": len(self.gts), "type": "count"}


class SecAgg:
    def __init__(self, taus=(1.0, 2.0, 5.0)):
        self.gts: List[float] = []
        self.preds: List[float] = []
        self.taus = taus

    def add(self, gt: float, pred: float):
        self.gts.append(float(gt))
        self.preds.append(float(pred))

    def summarize(self) -> Dict[str, Any]:
        if not self.gts:
            return {"mae": 0.0, "rmse": 0.0, "nmae_gt": 0.0, "within_tau": {str(t): 0.0 for t in self.taus}, "n": 0, "type": "seconds"}
        gt = np.array(self.gts, dtype=np.float32)
        pr = np.array(self.preds, dtype=np.float32)
        mae = float(np.mean(np.abs(gt - pr)))
        rmse = float(np.sqrt(np.mean((gt - pr) ** 2)))
        nmae_gt = float(mae / (float(np.mean(np.abs(gt))) + 1e-6))

        within = {}
        for t in self.taus:
            within[str(t)] = float(np.mean(np.abs(gt - pr) <= float(t)))

        return {"mae": mae, "rmse": rmse, "nmae_gt": nmae_gt, "within_tau": within, "n": len(self.gts), "type": "seconds"}


# ----------------------
# task type inference (from meta)
# ----------------------

def _infer_label_type(meta: Dict[str, Any], gt_answer: Any) -> str:
    lt = (meta.get("label_type") or "").lower().strip()
    if not lt:
        lt = (meta.get("answer_mode") or "").lower().strip()
    if not lt:
        lt = (meta.get("answer_type") or "").lower().strip()

    if lt in ("binary", "boolean", "bin", "yesno", "yes/no"):
        return "bool"
    if lt in ("choice", "mcq", "multi_choice", "multichoice", "option", "index"):
        return "choice_index"
    if lt in ("numeric", "number", "float", "int", "reg", "seconds", "duration"):
        return "seconds"
    if lt in ("count",):
        return "count"
    if lt in ("cat", "text", "string", "phrase"):
        return "text"

    # fallback: look at gt
    if isinstance(gt_answer, bool):
        return "bool"
    s = str(gt_answer).strip().lower()
    if s in ("true", "false", "yes", "no"):
        return "bool"
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", s):
        return "seconds"
    if re.fullmatch(r"[-+]?\d+", s):
        return "count"
    return "text"


# ----------------------
# main evaluation
# ----------------------

def evaluate_stsg(
    references: List[Any],
    predictions: List[str],
    metas: List[Dict[str, Any]],
    vocabs_json: Optional[str] = None,
) -> Dict[str, Any]:

    by_category = {
        "count": {"count": CountAgg()},
        "duration": {"duration": SecAgg()},
        "motion": {"cat": CatAgg()},
        "ordering": {"bool": BinAgg(), "cat": CatAgg(), "choice": CatAgg()},
        "extreme": {"bool": BinAgg(), "cat": CatAgg(), "seconds": SecAgg()},
        "concurrency": {"bool": BinAgg(), "seconds": SecAgg(), "choice": CatAgg()},
        "boundary": {"bool": BinAgg(), "cat": CatAgg(), "seconds": SecAgg()},
        "phase_transition": {"bool": BinAgg(), "cat": CatAgg(), "seconds": SecAgg(), "choice": CatAgg()},
    }

    task_hist = {
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
        "ordering_choice_dup": 0,
        "concurrency_choice_dup": 0,
        "phase_choice_dup": 0,
    }

    vocab_mapper = VocabMapper(vocabs_json) if vocabs_json else None
    vocab_text_categories = set(["motion", "boundary", "ordering", "extreme", "phase_transition"])

    for gt_answer, pred_text_raw, meta in zip(references, predictions, metas):
        category = (meta.get("category") or "").lower().strip()
        if category == "phase":
            category = "phase_transition"

        label_type = _infer_label_type(meta, gt_answer)
        pred_text = extract_final_answer(pred_text_raw)

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
            elif category == "extreme":
                by_category["extreme"]["bool"].add(gt_bool, pred_bool, score_pos)
                task_hist["extreme_bin"] += 1
            elif category == "phase_transition":
                by_category["phase_transition"]["bool"].add(gt_bool, pred_bool, score_pos)
                task_hist["phase_bin"] += 1
            else:
                # unknown bool category
                pass
            continue

        # ---- COUNT ----
        if category == "count" or label_type == "count":
            gt_num = parse_numeric_gt(gt_answer)
            if gt_num is None:
                continue
            gt_int = int(round(gt_num))
            pred_int = extract_first_int(pred_text)
            if pred_int is None:
                pred_int = 0
            by_category["count"]["count"].add(gt_int, pred_int)
            task_hist["count"] += 1
            continue

        # ---- DURATION ----
        if category == "duration":
            gt_num = parse_numeric_gt(gt_answer)
            if gt_num is None:
                continue
            pred_num = extract_first_number(pred_text)
            if pred_num is None:
                pred_num = 0.0
            by_category["duration"]["duration"].add(float(gt_num), float(pred_num))
            task_hist["duration"] += 1
            continue

        # ---- CHOICE ----
        # choice types are encoded under their category but meta may include choice_K
        if label_type == "choice_index":
            gt_i = extract_first_int(str(gt_answer))
            pr_i = extract_first_int(pred_text)
            if gt_i is None:
                continue
            if pr_i is None:
                pr_i = -1

            # unify to 0-based id (support both 0/1-based outputs)
            if gt_i >= 1 and pr_i >= 1:
                gt_id = gt_i - 1
                pr_id = pr_i - 1
            else:
                gt_id = gt_i
                pr_id = pr_i

            if category == "ordering":
                by_category["ordering"]["choice"].add(gt_id, pr_id)
                task_hist["ordering_choice"] += 1
            elif category == "concurrency":
                by_category["concurrency"]["choice"].add(gt_id, pr_id)
                task_hist["concurrency_choice"] += 1
            elif category == "phase_transition":
                by_category["phase_transition"]["choice"].add(gt_id, pr_id)
                task_hist["phase_choice"] += 1
            continue

        # ---- SECONDS / REG ----
        if label_type == "seconds":
            gt_num = parse_numeric_gt(gt_answer)
            if gt_num is None:
                continue
            pred_num = extract_first_number(pred_text)
            if pred_num is None:
                pred_num = 0.0

            if category == "boundary":
                by_category["boundary"]["seconds"].add(float(gt_num), float(pred_num))
                task_hist["boundary_reg"] += 1
            elif category == "concurrency":
                by_category["concurrency"]["seconds"].add(float(gt_num), float(pred_num))
                task_hist["concurrency_reg"] += 1
            elif category == "extreme":
                by_category["extreme"]["seconds"].add(float(gt_num), float(pred_num))
                task_hist["extreme_reg"] += 1
            elif category == "phase_transition":
                by_category["phase_transition"]["seconds"].add(float(gt_num), float(pred_num))
                task_hist["phase_reg"] += 1
            else:
                # unknown seconds category
                pass
            continue

        # ---- TEXT ----
        gt_label_raw = str(gt_answer)
        pred_label_raw = pred_text

        # Canonicalize ordering BEFORE vocab mapping (important!)
        if category == "ordering":
            gt_label_raw = canon_ordering_text(gt_label_raw)
            pred_label_raw = canon_ordering_text(pred_label_raw)

        if vocab_mapper is not None and category in vocab_text_categories:
            gt_id = vocab_mapper.encode_gt(category, gt_label_raw)
            pred_id = vocab_mapper.encode_pred(category, pred_label_raw)
        else:
            norm_gt = normalize_for_vocab_text(gt_label_raw)
            norm_pr = normalize_for_vocab_text(pred_label_raw)
            enc = LabelEncoder()
            gt_id = enc.encode(norm_gt)
            pred_id = enc.encode(norm_pr)

        if category == "boundary":
            by_category["boundary"]["cat"].add(gt_id, pred_id)
            task_hist["boundary_text"] += 1
        elif category == "ordering":
            by_category["ordering"]["cat"].add(gt_id, pred_id)
            task_hist["ordering_text"] += 1
        elif category == "phase_transition":
            by_category["phase_transition"]["cat"].add(gt_id, pred_id)
            task_hist["phase_text"] += 1
        elif category == "extreme":
            by_category["extreme"]["cat"].add(gt_id, pred_id)
            task_hist["extreme_text"] += 1
        elif category == "motion":
            by_category["motion"]["cat"].add(gt_id, pred_id)
            task_hist["motion"] += 1

    # summarize
    out = {"task_hist": task_hist, "by_category": {}, "overall_macro_score": 0.0}

    # boundary
    out["by_category"]["boundary"] = {
        "bool": by_category["boundary"]["bool"].summarize(),
        "cat": by_category["boundary"]["cat"].summarize(),
        "seconds": by_category["boundary"]["seconds"].summarize(),
    }
    # concurrency
    out["by_category"]["concurrency"] = {
        "bool": by_category["concurrency"]["bool"].summarize(),
        "seconds": by_category["concurrency"]["seconds"].summarize(),
        "choice": by_category["concurrency"]["choice"].summarize(),
    }
    # count
    out["by_category"]["count"] = {"count": by_category["count"]["count"].summarize()}
    # duration
    out["by_category"]["duration"] = {"duration": by_category["duration"]["duration"].summarize()}
    # extreme
    out["by_category"]["extreme"] = {
        "bool": by_category["extreme"]["bool"].summarize(),
        "cat": by_category["extreme"]["cat"].summarize(),
        "seconds": by_category["extreme"]["seconds"].summarize(),
    }
    # motion
    out["by_category"]["motion"] = {"cat": by_category["motion"]["cat"].summarize()}
    # ordering
    out["by_category"]["ordering"] = {
        "bool": by_category["ordering"]["bool"].summarize(),
        "cat": by_category["ordering"]["cat"].summarize(),
        "choice": by_category["ordering"]["choice"].summarize(),
    }
    # phase_transition
    out["by_category"]["phase_transition"] = {
        "bool": by_category["phase_transition"]["bool"].summarize(),
        "cat": by_category["phase_transition"]["cat"].summarize(),
        "seconds": by_category["phase_transition"]["seconds"].summarize(),
        "choice": by_category["phase_transition"]["choice"].summarize(),
    }

    # overall macro score (simple average of available primary metrics)
    primary = []
    primary.append(out["by_category"]["count"]["count"].get("top1_acc", 0.0))
    primary.append(out["by_category"]["duration"]["duration"].get("within_tau", {}).get("5.0", 0.0))
    primary.append(out["by_category"]["motion"]["cat"].get("top1_acc", 0.0))
    primary.append(out["by_category"]["ordering"]["bool"].get("f1", 0.0))
    primary.append(out["by_category"]["ordering"]["cat"].get("top1_acc", 0.0))
    primary.append(out["by_category"]["ordering"]["choice"].get("top1_acc", 0.0))
    primary.append(out["by_category"]["extreme"]["bool"].get("f1", 0.0))
    primary.append(out["by_category"]["extreme"]["cat"].get("top1_acc", 0.0))
    primary.append(out["by_category"]["extreme"]["seconds"].get("within_tau", {}).get("5.0", 0.0))
    primary.append(out["by_category"]["concurrency"]["bool"].get("f1", 0.0))
    primary.append(out["by_category"]["concurrency"]["choice"].get("top1_acc", 0.0))
    primary.append(out["by_category"]["concurrency"]["seconds"].get("within_tau", {}).get("5.0", 0.0))
    primary.append(out["by_category"]["boundary"]["bool"].get("f1", 0.0))
    primary.append(out["by_category"]["boundary"]["cat"].get("top1_acc", 0.0))
    primary.append(out["by_category"]["boundary"]["seconds"].get("within_tau", {}).get("5.0", 0.0))
    primary.append(out["by_category"]["phase_transition"]["bool"].get("f1", 0.0))
    primary.append(out["by_category"]["phase_transition"]["cat"].get("top1_acc", 0.0))
    primary.append(out["by_category"]["phase_transition"]["choice"].get("top1_acc", 0.0))
    primary.append(out["by_category"]["phase_transition"]["seconds"].get("within_tau", {}).get("5.0", 0.0))

    out["overall_macro_score"] = float(np.mean(primary)) if primary else 0.0
    return out
