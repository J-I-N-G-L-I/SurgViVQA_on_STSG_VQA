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
    return float(a) / float(b) if b > 0 else 0.0


def macro_f1_multiclass(y_true: List[int], y_pred: List[int]) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    classes = np.unique(y_true)
    if classes.size == 0:
        return 0.0
    f1s: List[float] = []
    for c in classes:
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        prec = safe_div(tp, tp + fp)
        rec = safe_div(tp, tp + fn)
        f1 = safe_div(2 * prec * rec, (prec + rec)) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else 0.0


def binary_prf_auroc(y_true: List[int], y_pred: List[int], y_score_pos: List[float]) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.int32)
    y_score = np.asarray(y_score_pos, dtype=np.float64)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    acc = safe_div(tp + tn, len(y_true))
    prec = safe_div(tp, tp + fp)
    rec = safe_div(tp, tp + fn)
    f1 = safe_div(2 * prec * rec, (prec + rec)) if (prec + rec) > 0 else 0.0

    if y_true.min() == y_true.max():
        auroc = 0.5
    else:
        order = np.argsort(-y_score)
        y_sorted = y_true[order]
        score_sorted = y_score[order]

        P = float((y_true == 1).sum())
        N = float((y_true == 0).sum())

        tp_cum = 0.0
        fp_cum = 0.0
        prev = None
        roc: List[Tuple[float, float]] = [(0.0, 0.0)]

        for i in range(len(y_sorted)):
            s = score_sorted[i]
            if prev is None or s != prev:
                roc.append((safe_div(fp_cum, N), safe_div(tp_cum, P)))
                prev = s
            if y_sorted[i] == 1:
                tp_cum += 1.0
            else:
                fp_cum += 1.0
        roc.append((safe_div(fp_cum, N), safe_div(tp_cum, P)))

        auroc = 0.0
        for i in range(1, len(roc)):
            x0, y0 = roc[i - 1]
            x1, y1 = roc[i]
            auroc += (x1 - x0) * (y0 + y1) / 2.0
        auroc = float(auroc)

    return {"acc": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1), "auroc": float(auroc)}


# ----------------------
# aggregators
# ----------------------

class BinAgg:
    def __init__(self):
        self.yt: List[int] = []
        self.yp: List[int] = []
        self.score: List[float] = []

    def add(self, yt: int, yp: int, score_pos: float):
        self.yt.append(int(yt))
        self.yp.append(int(yp))
        self.score.append(float(score_pos))

    def finalize(self) -> Dict[str, Any]:
        n = len(self.yt)
        out: Dict[str, Any] = {"n": n, "type": "bool", "acc": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "auroc": 0.5}
        if n == 0:
            return out
        out.update(binary_prf_auroc(self.yt, self.yp, self.score))
        return out


class CatAgg:
    def __init__(self):
        self.yt: List[int] = []
        self.yp: List[int] = []

    def add(self, yt: int, yp: int):
        self.yt.append(int(yt))
        self.yp.append(int(yp))

    def finalize_basic(self) -> Dict[str, Any]:
        n = len(self.yt)
        out: Dict[str, Any] = {"n": n, "type": "cat", "top1_acc": 0.0, "top3_acc": 0.0}
        if n == 0:
            return out
        y_true = np.asarray(self.yt, dtype=np.int64)
        y_pred = np.asarray(self.yp, dtype=np.int64)
        top1 = float((y_true == y_pred).mean())
        out["top1_acc"] = top1
        out["top3_acc"] = top1  # 生成式没有rank就先等于top1
        return out

    def finalize_with_macro_f1(self) -> Dict[str, Any]:
        z = self.finalize_basic()
        if z["n"] == 0:
            z["macro_f1"] = 0.0
            return z
        z["macro_f1"] = macro_f1_multiclass(self.yt, self.yp)
        return z


class CountAgg:
    def __init__(self):
        self.yt: List[int] = []
        self.yp: List[int] = []

    def add(self, yt: int, yp: int):
        self.yt.append(int(yt))
        self.yp.append(int(yp))

    def finalize(self) -> Dict[str, Any]:
        n = len(self.yt)
        out: Dict[str, Any] = {"n": n, "type": "count", "top1_acc": 0.0, "within1_acc": 0.0, "macro_f1": 0.0, "top3_acc": 0.0}
        if n == 0:
            return out
        y_true = np.asarray(self.yt, dtype=np.int64)
        y_pred = np.asarray(self.yp, dtype=np.int64)
        out["top1_acc"] = float((y_true == y_pred).mean())
        out["within1_acc"] = float((np.abs(y_true - y_pred) <= 1).mean())
        out["macro_f1"] = macro_f1_multiclass(self.yt, self.yp)
        out["top3_acc"] = out["top1_acc"]
        return out


class SecAgg:
    def __init__(self):
        self.gt: List[float] = []
        self.pr: List[float] = []

    def add(self, gt: float, pr: float):
        self.gt.append(float(gt))
        self.pr.append(float(pr))

    def finalize(self) -> Dict[str, Any]:
        n = len(self.gt)
        out: Dict[str, Any] = {"n": n, "type": "seconds", "mae": 0.0, "rmse": 0.0, "nmae_gt": 0.0, "within_tau": {"1.0": 0.0, "2.0": 0.0, "5.0": 0.0}}
        if n == 0:
            return out
        gt = np.asarray(self.gt, dtype=np.float64)
        pr = np.asarray(self.pr, dtype=np.float64)
        err = np.abs(pr - gt)
        out["mae"] = float(err.mean())
        out["rmse"] = float(np.sqrt(np.mean((pr - gt) ** 2)))
        denom = np.maximum(np.abs(gt), 1e-6)
        out["nmae_gt"] = float(np.mean(err / denom))
        for tau in (1.0, 2.0, 5.0):
            out["within_tau"][f"{tau}"] = float((err <= tau).mean())
        return out


# ----------------------
# parsing helpers
# ----------------------

_INT_RE = re.compile(r"-?\d+")
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
_FINAL_ANSWER_RE = re.compile(r"FINAL_ANSWER\s*:\s*(.*)", re.IGNORECASE | re.DOTALL)


def extract_final_answer(text: Any) -> str:
    if not isinstance(text, str):
        text = str(text)
    s = text.strip()
    m = _FINAL_ANSWER_RE.search(s)
    if not m:
        return s
    ans = m.group(1).strip()
    for sep in ("\n", "\r"):
        if sep in ans:
            ans = ans.split(sep, 1)[0].strip()
    return ans


def extract_first_int(text: Any) -> Optional[int]:
    if not isinstance(text, str):
        text = str(text)
    m = _INT_RE.search(text)
    if not m:
        return None
    try:
        return int(m.group(0))
    except Exception:
        return None


def extract_first_number(text: Any) -> Optional[float]:
    if not isinstance(text, str):
        text = str(text)
    m = _NUM_RE.search(text)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def parse_bool_label(x: Any) -> Optional[int]:
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, (int, np.integer)):
        return 1 if int(x) != 0 else 0
    if isinstance(x, str):
        s = x.strip().lower().strip(".!,;:()[]{}")
        if s in {"yes", "y", "true", "1"}:
            return 1
        if s in {"no", "n", "false", "0"}:
            return 0
    return None


def parse_bool_text(x: Any) -> Optional[int]:
    s = extract_final_answer(x).strip().lower()
    s = s.strip().strip(".!,;:()[]{}")
    if not s:
        return None

    # 优先看首 token
    first = s.split()[0].strip(".!,;:()[]{}")
    if first in {"yes", "y", "true", "1"}:
        return 1
    if first in {"no", "n", "false", "0"}:
        return 0

    # 退化：全文包含
    if ("yes" in s or "true" in s) and ("no" not in s and "false" not in s):
        return 1
    if ("no" in s or "false" in s) and ("yes" not in s and "true" not in s):
        return 0
    return None


def parse_numeric_gt(x: Any) -> Optional[float]:
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    if isinstance(x, str):
        x = x.strip()
        try:
            return float(x)
        except Exception:
            pass
        v = extract_first_number(x)
        if v is not None:
            return float(v)
    return None


# ----------------------
# vocab mapping for text tasks
# ----------------------

def simple_stem(token: str) -> str:
    t = token
    irregular = {
        "retraction": "retract",
        "retracting": "retract",
        "retracted": "retract",
        "dissection": "dissect",
        "dissecting": "dissect",
        "dissected": "dissect",
        "coagulation": "coagulate",
        "coagulating": "coagulate",
        "coagulated": "coagulate",
        "extraction": "extract",
        "extracting": "extract",
        "extracted": "extract",
        "grasping": "grasp",
        "clipping": "clip",
    }
    if t in irregular:
        return irregular[t]
    for suf in ("ization", "isation", "ation", "tion", "sion"):
        if t.endswith(suf) and len(t) > len(suf) + 2:
            t = t[: -len(suf)]
            break
    for suf in ("ing", "ed", "es", "s"):
        if t.endswith(suf) and len(t) > len(suf) + 2:
            t = t[: -len(suf)]
            break
    return t

def canon_ordering_text(s: Any) -> str:
    """
    Canonicalize ordering text so that semantically equivalent forms map to the same vocab label.
    - Normalize common synonyms to "before/after"
    - Convert "A after B" -> "B before A" (better matching your ordering vocab that uses "before")
    """
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)

    x = s.strip().lower()

    # unify whitespace
    x = re.sub(r"\s+", " ", x)

    # synonym normalization
    x = x.replace("prior to", "before").replace("earlier than", "before").replace("precede", "before")
    x = x.replace("later than", "after").replace("subsequent to", "after").replace("follow", "after")

    # Convert: A after B  ->  B before A
    # Only do this when "before" is not already present to avoid weird double forms.
    if " after " in x and " before " not in x:
        parts = x.split(" after ", 1)
        if len(parts) == 2:
            a, b = parts[0].strip(), parts[1].strip()
            if a and b:
                x = f"{b} before {a}"

    return x

def normalize_for_vocab_text(s: Any) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = s.strip().lower()

    replacements = {
        "gall bladder": "gallbladder",
        "gall-bladder": "gallbladder",
        "cystic duct": "cystic_duct",
        "cystic artery": "cystic_artery",
        "cystic pedicle": "cystic_pedicle",
        "cystic plate": "cystic_plate",
        "specimen bag": "specimen_bag",
        "abdominal wall cavity": "abdominal_wall_cavity",
    }
    for k, v in replacements.items():
        s = s.replace(k, v)

    s = s.replace("_", " ")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    tokens = re.split(r"\s+", s)

    stopwords = {"", "the", "a", "an", "of", "on", "in", "to", "at", "for",
                 "is", "was", "are", "were", "be", "been", "being",
                 "by", "from", "with", "without", "this", "that", "these", "those"}

    processed: List[str] = []
    for t in tokens:
        if not t or t in stopwords:
            continue
        processed.append(simple_stem(t))
    return " ".join(processed)


class LabelEncoder:
    def __init__(self):
        self.str2id: Dict[str, int] = {}

    def encode(self, s: str) -> int:
        if s not in self.str2id:
            self.str2id[s] = len(self.str2id)
        return self.str2id[s]


class VocabMapper:
    def __init__(self, vocabs_path: Path):
        with vocabs_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        self.vocabs: Dict[str, List[str]] = {}
        self.norm2id: Dict[str, Dict[str, int]] = {}
        self.id2tokens: Dict[str, Dict[int, set]] = {}

        known_keys = ["boundary", "extreme", "motion", "ordering", "phase_transition"]
        for key in known_keys:
            if key not in data:
                continue
            labels = data[key]
            if not isinstance(labels, list):
                raise ValueError(f"vocabs[{key}] must be a list.")
            self.vocabs[key] = labels

            norm_map: Dict[str, int] = {}
            tok_map: Dict[int, set] = {}
            for idx, lab in enumerate(labels):
                norm = normalize_for_vocab_text(lab)
                if norm:
                    norm_map.setdefault(norm, idx)
                tok_map[idx] = set(norm.split()) if norm else set()

            self.norm2id[key] = norm_map
            self.id2tokens[key] = tok_map

        self.fallback_encoders: Dict[str, LabelEncoder] = {}

    @staticmethod
    def _canon_category_name(category: str) -> str:
        c = category.lower()
        if c in ("phase", "phase_transition"):
            return "phase_transition"
        return c

    def _get_fallback_encoder(self, key: str) -> LabelEncoder:
        if key not in self.fallback_encoders:
            self.fallback_encoders[key] = LabelEncoder()
        return self.fallback_encoders[key]

    def encode_gt(self, category: str, label: Any) -> int:
        key = self._canon_category_name(category)

        # category-specific canonicalization BEFORE normalization
        if key == "ordering":
            label = canon_ordering_text(label)

        norm = normalize_for_vocab_text(label)
        if key in self.vocabs:
            return self.norm2id[key].get(norm, 0)
        return self._get_fallback_encoder(key).encode(norm)

    def encode_pred(self, category: str, label: Any) -> int:
        key = self._canon_category_name(category)

        # category-specific canonicalization BEFORE normalization
        if key == "ordering":
            label = canon_ordering_text(label)

        norm = normalize_for_vocab_text(label)
        if key in self.vocabs:
            id_map = self.norm2id[key]
            tok_map = self.id2tokens[key]

            # 1) exact normalized match
            if norm in id_map:
                return id_map[norm]

            tokens_pred = set(norm.split())
            if not tokens_pred:
                return 0

            best_full_idx = 0
            best_full_cover = 0.0
            best_jacc_idx = 0
            best_jacc = 0.0

            for idx, tokens_lab in tok_map.items():
                if idx == 0 or not tokens_lab:
                    continue
                inter = len(tokens_pred & tokens_lab)
                if inter == 0:
                    continue

                cover = inter / float(len(tokens_lab))
                if cover > best_full_cover:
                    best_full_cover = cover
                    best_full_idx = idx

                union = len(tokens_pred | tokens_lab)
                jacc = inter / float(union)
                if jacc > best_jacc:
                    best_jacc = jacc
                    best_jacc_idx = idx

            # 2) full coverage of label tokens
            if best_full_cover >= 1.0:
                return best_full_idx

            # 3) fallback by jaccard threshold (ordering a bit looser)
            min_jacc = 0.5
            if key == "ordering":
                min_jacc = 0.4

            if best_jacc >= min_jacc:
                return best_jacc_idx
            return 0

        return self._get_fallback_encoder(key).encode(norm)


# main evaluation

def _infer_label_type(meta: Dict[str, Any], gt_answer: Any) -> str:
    """
    统一题型推断（评测侧兜底）：
    - label_type 优先
    - 其次 answer_mode
    - 有 choice_K / choices → choice_index
    - 再看 answer_type / category / gt_answer
    """
    lt = (meta.get("label_type") or "").lower()
    if not lt:
        lt = (meta.get("answer_mode") or "").lower()
    if lt:
        return lt

    if meta.get("choice_K", None) is not None:
        return "choice_index"
    if meta.get("choices", None) is not None:
        return "choice_index"

    at = (meta.get("answer_type") or "").lower()
    cat = (meta.get("category") or "").lower()
    if cat == "phase":
        cat = "phase_transition"

    if at in ("bool", "boolean", "binary", "bin"):
        return "bool"

    if cat == "count":
        return "count"
    if cat == "duration":
        return "duration"

    # 0/1 + 这些类别 → 更像 bool（避免被当 seconds）
    if isinstance(gt_answer, (int, np.integer)) and int(gt_answer) in (0, 1) and cat in (
        "ordering", "boundary", "concurrency", "extreme", "phase_transition"
    ):
        return "bool"

    if isinstance(gt_answer, (int, float, np.integer, np.floating)):
        return "seconds"

    return "text"


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

    for gt_answer, pred_text_raw, meta in zip(references, predictions, metas):
        category = (meta.get("category") or "").lower()
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
            by_category["count"]["count"].add(gt_i, int(pred_i))
            task_hist["count"] += 1
            continue

        # ---- SECONDS ----
        if label_type in ("seconds", "duration", "reg"):
            gt_v = parse_numeric_gt(gt_answer)
            if gt_v is None:
                continue
            pred_v = extract_first_number(pred_text)
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
            pred_id = vocab_mapper.encode_pred(category, pred_label_raw)
        else:
            # 没有 vocabs_json 的情况下，退化成“严格规范化字符串”
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

    # finalize
    bd_bool = by_category["boundary"]["bool"].finalize()
    bd_cat = by_category["boundary"]["cat"].finalize_with_macro_f1()
    bd_sec = by_category["boundary"]["seconds"].finalize()

    cc_bool = by_category["concurrency"]["bool"].finalize()
    cc_sec = by_category["concurrency"]["seconds"].finalize()
    cc_choice = by_category["concurrency"]["choice"].finalize_basic()

    ct = by_category["count"]["count"].finalize()

    dur = by_category["duration"]["duration"].finalize()
    dur["type"] = "duration"

    ex_bool = by_category["extreme"]["bool"].finalize()
    ex_cat = by_category["extreme"]["cat"].finalize_with_macro_f1()
    ex_sec = by_category["extreme"]["seconds"].finalize()

    mo_cat = by_category["motion"]["cat"].finalize_with_macro_f1()

    od_bool = by_category["ordering"]["bool"].finalize()
    od_cat = by_category["ordering"]["cat"].finalize_with_macro_f1()
    od_choice = by_category["ordering"]["choice"].finalize_basic()

    ph_bool = by_category["phase_transition"]["bool"].finalize()
    ph_cat = by_category["phase_transition"]["cat"].finalize_with_macro_f1()
    ph_sec = by_category["phase_transition"]["seconds"].finalize()
    ph_choice = by_category["phase_transition"]["choice"].finalize_basic()

    def pick_bool(d): return d["f1"] if d.get("n", 0) > 0 else None
    def pick_cat(d): return d["top1_acc"] if d.get("n", 0) > 0 else None
    def pick_count(d): return d["within1_acc"] if d.get("n", 0) > 0 else None
    def pick_secs(d): return d["within_tau"]["5.0"] if d.get("n", 0) > 0 else None

    cell_scores: List[float] = []
    for sc in [
        pick_bool(bd_bool), pick_cat(bd_cat), pick_secs(bd_sec),
        pick_bool(cc_bool), pick_secs(cc_sec), pick_cat(cc_choice),
        pick_count(ct),
        pick_secs(dur),
        pick_bool(ex_bool), pick_cat(ex_cat), pick_secs(ex_sec),
        pick_cat(mo_cat),
        pick_bool(od_bool), pick_cat(od_cat), pick_cat(od_choice),
        pick_bool(ph_bool), pick_cat(ph_cat), pick_secs(ph_sec), pick_cat(ph_choice),
    ]:
        if sc is not None:
            cell_scores.append(sc)

    result: Dict[str, Any] = {
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
    }
    return result
