#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate SSGVQA predictions with a DeepSeek LLM-as-a-judge.

Why an LLM judge?
- Generative models often output sentences rather than a single closed-set label.
- Rule-based or substring mapping can be unreliable for semantic correctness.
- We use DeepSeek to (1) map the prediction to exactly one closed-set label,
  and (2) judge semantic equivalence between prediction and GT under the question.

Closed-set metrics (Acc / mAP / mAR / mAF1 / wF1) follow the same definition
as utils/test_llava_med_ssgvqa.py and remain unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import requests

# ----------------------
# Closed-set label vocabulary (order must remain unchanged)
# ----------------------
LABELS: List[str] = [
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

# ----------------------
# Metrics (copied from utils/test_llava_med_ssgvqa.py to keep identical definitions)
# ----------------------

def compute_metrics(y_true: List[int], y_pred: List[int], num_classes: int) -> Dict[str, float]:
    """
    Compute Acc, mAP, mAR, mAF1, wF1 using classic per-class statistics.
    Unknown predictions are allowed in y_pred as -1 (count as FN for their GT).
    """
    if not y_true:
        return {"acc": 0.0, "mAP": 0.0, "mAR": 0.0, "mAF1": 0.0, "wF1": 0.0}

    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64)

    acc = float((y_true_arr == y_pred_arr).mean())

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


# ----------------------
# Example JSONL records (for sanity checks / unit-test-like validation)
# ----------------------
# {
#   "video":"VID02",
#   "frame_id":"102",
#   "image_path":"/mnt/scratch/sc232jl/datasets/CholecT45/data/VID02/000102.png",
#   "question":"Which target is being retracted in the scene?",
#   "gt_answer":"omentum",
#   "pred_answer":"The target being retracted in the scene is the cystic plate.",
#   "match_type":"substring"
# }
# {
#   "video":"VID02",
#   "frame_id":"102",
#   "image_path":"/mnt/scratch/sc232jl/datasets/CholecT45/data/VID02/000102.png",
#   "question":"What is the grasper doing?",
#   "gt_answer":"retract",
#   "pred_answer":"The grasper is being used to grasp the cystic pedicle, which is a structure that connects the gallbladder to the cystic duct.",
#   "match_type":"substring"
# }
# {
#   "video":"VID02",
#   "frame_id":"102",
#   "image_path":"/mnt/scratch/sc232jl/datasets/CholecT45/data/VID02/000102.png",
#   "question":"Which target is being cutted in the scene?",
#   "gt_answer":"adhesion",
#   "pred_answer":"The target being cutted in the scene is the cystic plate.",
#   "match_type":"substring"
# }


SYSTEM_PROMPT = (
    "You are a strict evaluator for a closed-set VQA benchmark.\n"
    "Given QUESTION, GT_LABEL, and PRED_ANSWER, you must:\n"
    "(1) map PRED_ANSWER to exactly ONE label from CHOICES (or UNKNOWN if truly impossible);\n"
    "(2) decide whether PRED_ANSWER is semantically equivalent to GT_LABEL under the QUESTION (true/false).\n"
    "Return JSON only with keys: mapped_label, is_correct, confidence, note.\n\n"
    "* mapped_label must be one of CHOICES or 'UNKNOWN'.\n"
    "* is_correct must be true only if the predicted meaning matches GT_LABEL (not just ‘sounds plausible’).\n"
    "* Do NOT include any text outside JSON."
)


@dataclass
class JudgeConfig:
    base_url: str
    model: str
    api_key: str
    request_timeout: float
    max_retries: int
    sleep_seconds: float


class GlobalRateLimiter:
    """Global throttle to avoid excessive rate-limit errors across threads."""

    def __init__(self, min_interval: float) -> None:
        self.min_interval = max(0.0, float(min_interval))
        self._lock = threading.Lock()
        self._last_ts = 0.0

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            delta = now - self._last_ts
            if delta < self.min_interval:
                time.sleep(self.min_interval - delta)
            self._last_ts = time.monotonic()


class DeepSeekJudge:
    def __init__(self, cfg: JudgeConfig, limiter: GlobalRateLimiter) -> None:
        self.cfg = cfg
        self.limiter = limiter
        self.session = requests.Session()
        self._response_format_supported = True

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.cfg.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
        }
        backoff = 1.0
        for attempt in range(self.cfg.max_retries):
            try:
                self.limiter.wait()
                resp = self.session.post(
                    url, headers=headers, json=payload, timeout=self.cfg.request_timeout
                )
                if resp.status_code == 400 and self._response_format_supported:
                    # Some deployments may not support response_format.
                    self._response_format_supported = False
                    raise requests.HTTPError("response_format not supported", response=resp)
                if resp.status_code in (429,) or 500 <= resp.status_code <= 599:
                    raise requests.HTTPError(f"Retryable status: {resp.status_code}", response=resp)
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException:
                if attempt >= self.cfg.max_retries - 1:
                    raise
                time.sleep(backoff + random.random() * 0.1)
                backoff = min(backoff * 2.0, 30.0)
        raise RuntimeError("Exhausted retries")

    def _build_messages(self, question: str, gt_label: str, pred_answer: str) -> List[Dict[str, str]]:
        choices = ", ".join(LABELS)
        user_msg = (
            f"CHOICES: {choices}\n"
            f"QUESTION: {question}\n"
            f"GT_LABEL: {gt_label}\n"
            f"PRED_ANSWER: {pred_answer}"
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

    def _call(self, question: str, gt_label: str, pred_answer: str, add_json_hint: bool) -> Dict[str, Any]:
        messages = self._build_messages(question, gt_label, pred_answer)
        if add_json_hint:
            messages.append({"role": "user", "content": "Return valid JSON only."})

        payload: Dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": 0.0,
        }
        if self._response_format_supported:
            payload["response_format"] = {"type": "json_object"}

        return self._post(payload)

    def judge(self, question: str, gt_label: str, pred_answer: str) -> Tuple[Dict[str, Any], str]:
        """
        Returns (parsed_json, raw_content).
        If parsing fails twice, returns default UNKNOWN output.
        """
        default = {
            "mapped_label": "UNKNOWN",
            "is_correct": False,
            "confidence": 0.0,
            "note": "parse_failed",
        }
        for attempt in range(2):
            try:
                resp = self._call(question, gt_label, pred_answer, add_json_hint=(attempt == 1))
                content = (
                    resp.get("choices", [{}])[0].get("message", {}).get("content", "")
                )
                parsed = json.loads(content)
                return parsed, content
            except Exception:
                continue
        return default, ""


def make_key(rec: Dict[str, Any]) -> str:
    return f"{rec.get('video','')}\t{rec.get('frame_id','')}\t{rec.get('question','')}"


def load_predictions(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return items


def load_existing_keys(path: Path) -> set:
    keys = set()
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


def evaluate_metrics(judged_path: Path) -> Dict[str, Any]:
    label_to_idx = {lab: i for i, lab in enumerate(LABELS)}

    y_true: List[int] = []
    y_pred: List[int] = []
    per_video: Dict[str, Dict[str, List[int]]] = {}

    total_records = 0
    unknown_gt_count = 0
    unknown_pred_count = 0
    judge_correct_count = 0
    judge_inconsistency_count = 0

    with judged_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            total_records += 1
            video = str(rec.get("video", ""))
            gt_label = str(rec.get("gt_answer", ""))
            mapped_label = str(rec.get("mapped_label", "UNKNOWN"))
            is_correct = bool(rec.get("is_correct", False))

            if is_correct:
                judge_correct_count += 1
                if mapped_label != gt_label:
                    judge_inconsistency_count += 1

            gt_idx = label_to_idx.get(gt_label, None)
            if gt_idx is None:
                unknown_gt_count += 1
                continue

            if mapped_label in label_to_idx:
                pred_idx = label_to_idx[mapped_label]
            else:
                pred_idx = -1
                unknown_pred_count += 1

            y_true.append(gt_idx)
            y_pred.append(pred_idx)

            if video not in per_video:
                per_video[video] = {"y_true": [], "y_pred": []}
            per_video[video]["y_true"].append(gt_idx)
            per_video[video]["y_pred"].append(pred_idx)

    overall = compute_metrics(y_true, y_pred, num_classes=len(LABELS))
    per_video_metrics: Dict[str, Dict[str, float]] = {}
    for vid in sorted(per_video.keys()):
        v_metrics = compute_metrics(
            per_video[vid]["y_true"],
            per_video[vid]["y_pred"],
            num_classes=len(LABELS),
        )
        per_video_metrics[vid] = {"n": len(per_video[vid]["y_true"]), **v_metrics}

    num_eval_samples = len(y_true)
    overall_out = {
        **overall,
        "num_samples": total_records,
        "num_eval_samples": num_eval_samples,
        "unknown_gt_count": unknown_gt_count,
        "unknown_gt_ratio": float(unknown_gt_count / total_records) if total_records > 0 else 0.0,
        "unknown_pred_count": unknown_pred_count,
        "unknown_pred_ratio": float(unknown_pred_count / num_eval_samples) if num_eval_samples > 0 else 0.0,
        "judge_correct_ratio": float(judge_correct_count / total_records) if total_records > 0 else 0.0,
        "judge_inconsistency_count": judge_inconsistency_count,
    }

    return {"overall": overall_out, "per_video": per_video_metrics, "label_vocab": LABELS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SSGVQA DeepSeek judge evaluation")
    parser.add_argument("--predictions-jsonl", required=True, help="Input predictions JSONL")
    parser.add_argument("--output-judged-jsonl", required=True, help="Output judged JSONL")
    parser.add_argument("--output-metrics-json", required=True, help="Output metrics JSON")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        help="DeepSeek base URL",
    )
    parser.add_argument(
        "--judge-model",
        default="deepseek-chat",
        help="DeepSeek model name",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Only run first N samples")
    parser.add_argument("--resume", action="store_true", help="Resume from existing judged JSONL")
    parser.add_argument("--request-timeout", type=float, default=60.0, help="Request timeout (s)")
    parser.add_argument("--sleep-seconds", type=float, default=0.0, help="Sleep between requests")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--save-raw", action="store_true", help="Store judge_raw in output JSONL")
    parser.add_argument("--save-every", type=int, default=200, help="Flush every N records")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("DEEPSEEK_API_KEY is not set. Please export it before running.")

    pred_path = Path(args.predictions_jsonl)
    out_judged = Path(args.output_judged_jsonl)
    out_metrics = Path(args.output_metrics_json)
    out_judged.parent.mkdir(parents=True, exist_ok=True)
    out_metrics.parent.mkdir(parents=True, exist_ok=True)

    all_records = load_predictions(pred_path)
    total_records = len(all_records)
    if args.max_samples is not None:
        all_records = all_records[: max(args.max_samples, 0)]

    existing_keys = load_existing_keys(out_judged) if args.resume else set()

    cfg = JudgeConfig(
        base_url=args.base_url,
        model=args.judge_model,
        api_key=api_key,
        request_timeout=args.request_timeout,
        max_retries=5,
        sleep_seconds=args.sleep_seconds,
    )
    limiter = GlobalRateLimiter(cfg.sleep_seconds)
    judge = DeepSeekJudge(cfg, limiter)

    mode = "a" if args.resume and out_judged.exists() else "w"
    processed = 0

    def judge_one(rec: Dict[str, Any]) -> Dict[str, Any]:
        question = str(rec.get("question", ""))
        gt = str(rec.get("gt_answer", ""))
        pred = str(rec.get("pred_answer", ""))
        parsed, raw = judge.judge(question, gt, pred)

        mapped_label = str(parsed.get("mapped_label", "UNKNOWN"))
        is_correct = bool(parsed.get("is_correct", False))
        confidence = float(parsed.get("confidence", 0.0))
        note = str(parsed.get("note", ""))

        out = dict(rec)
        out.update(
            {
                "mapped_label": mapped_label,
                "is_correct": is_correct,
                "confidence": confidence,
                "note": note,
            }
        )
        if args.save_raw:
            out["judge_raw"] = raw
        return out

    with out_judged.open(mode, encoding="utf-8") as fout:
        if args.num_workers <= 1:
            for rec in all_records:
                if args.resume and make_key(rec) in existing_keys:
                    continue
                judged = judge_one(rec)
                fout.write(json.dumps(judged, ensure_ascii=False) + "\n")
                processed += 1
                if processed % args.save_every == 0:
                    fout.flush()
                    os.fsync(fout.fileno())
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            to_run: List[Dict[str, Any]] = []
            for rec in all_records:
                if args.resume and make_key(rec) in existing_keys:
                    continue
                to_run.append(rec)

            with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
                futures = [ex.submit(judge_one, r) for r in to_run]
                for fut in as_completed(futures):
                    judged = fut.result()
                    fout.write(json.dumps(judged, ensure_ascii=False) + "\n")
                    processed += 1
                    if processed % args.save_every == 0:
                        fout.flush()
                        os.fsync(fout.fileno())

    metrics = evaluate_metrics(out_judged)
    metrics["overall"]["num_samples"] = total_records

    with out_metrics.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[Done] Saved judged JSONL to: {out_judged}")
    print(f"[Done] Saved metrics JSON to: {out_metrics}")


if __name__ == "__main__":
    main()
