#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DeepSeek LLM-as-a-judge for SurgViVQA closed-set VQA evaluation.

This module provides:
- A configurable DeepSeek API client with retry and rate-limiting.
- Prompt construction for mapping predicted labels to the closed-set vocabulary.
- Semantic equivalence judgment between prediction and GT (including multi-label GT).
"""

from __future__ import annotations

import json
import os
import random
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests

# Import label vocabulary from fine-tune/labels.py
import sys
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_FINE_TUNE_DIR = os.path.dirname(_THIS_DIR)
_PROJECT_ROOT = os.path.dirname(_FINE_TUNE_DIR)

# Add paths for imports
for p in [_PROJECT_ROOT, _FINE_TUNE_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from labels import SSGVQA_LABELS
except ImportError:
    # Last resort: define locally (keep in sync with fine-tune/labels.py)
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


# ============================================================================
# System prompt for LLM judge (multi-label aware)
# ============================================================================

SYSTEM_PROMPT = """\
You are a strict evaluator for a closed-set surgical VQA benchmark.

Given:
- QUESTION: The question asked about a surgical frame.
- GT_LABELS: Ground-truth label(s). May contain multiple comma-separated labels (any one is correct).
- PRED_LABEL: The model's predicted label (already from the closed set).

Your tasks:
1. Confirm PRED_LABEL is in CHOICES (or mark UNKNOWN if not).
2. Decide if PRED_LABEL is semantically equivalent to ANY of the GT_LABELS under the QUESTION.
   - Return is_correct=true ONLY if the prediction matches at least one GT label.
   - Synonyms or obvious semantic equivalents count as matches (e.g., "specimen_bag" ≈ "specimenbag").
   - Do NOT accept partial or plausible but incorrect answers.

Return ONLY valid JSON with keys:
{
  "mapped_label": "<PRED_LABEL if valid, else 'UNKNOWN'>",
  "is_correct": <true|false>,
  "confidence": <0.0-1.0>,
  "note": "<brief explanation>"
}

Do NOT include any text outside the JSON object.
"""


@dataclass
class JudgeConfig:
    """Configuration for the DeepSeek judge API client."""
    base_url: str
    model: str
    api_key: str
    request_timeout: float = 60.0
    max_retries: int = 5
    sleep_seconds: float = 0.0


class GlobalRateLimiter:
    """Thread-safe global rate limiter to avoid excessive API calls."""

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
    """
    DeepSeek API client for judging VQA predictions.
    
    Handles:
    - Retry logic with exponential backoff
    - Rate limiting across threads
    - Fallback when response_format is unsupported
    """

    def __init__(self, cfg: JudgeConfig, limiter: GlobalRateLimiter) -> None:
        self.cfg = cfg
        self.limiter = limiter
        self.session = requests.Session()
        self._response_format_supported = True

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Send POST request to DeepSeek API with retry logic."""
        url = f"{self.cfg.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
        }
        backoff = 1.0
        last_error = None

        for attempt in range(self.cfg.max_retries):
            try:
                self.limiter.wait()
                resp = self.session.post(
                    url, headers=headers, json=payload, timeout=self.cfg.request_timeout
                )
                
                # Handle response_format not supported
                if resp.status_code == 400 and self._response_format_supported:
                    self._response_format_supported = False
                    raise requests.HTTPError("response_format not supported", response=resp)
                
                # Retry on rate limit or server errors
                if resp.status_code in (429,) or 500 <= resp.status_code <= 599:
                    raise requests.HTTPError(f"Retryable status: {resp.status_code}", response=resp)
                
                resp.raise_for_status()
                return resp.json()
                
            except requests.RequestException as e:
                last_error = e
                if attempt >= self.cfg.max_retries - 1:
                    raise
                time.sleep(backoff + random.random() * 0.1)
                backoff = min(backoff * 2.0, 30.0)

        raise RuntimeError(f"Exhausted retries: {last_error}")

    def _build_messages(
        self,
        question: str,
        gt_labels: str,
        pred_label: str,
    ) -> List[Dict[str, str]]:
        """Build chat messages for the judge API."""
        choices = ", ".join(SSGVQA_LABELS)
        user_msg = (
            f"CHOICES: {choices}\n"
            f"QUESTION: {question}\n"
            f"GT_LABELS: {gt_labels}\n"
            f"PRED_LABEL: {pred_label}"
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

    def _call(
        self,
        question: str,
        gt_labels: str,
        pred_label: str,
        add_json_hint: bool = False,
    ) -> Dict[str, Any]:
        """Make a single API call."""
        messages = self._build_messages(question, gt_labels, pred_label)
        
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

    def judge(
        self,
        question: str,
        gt_labels: str,
        pred_label: str,
    ) -> Tuple[Dict[str, Any], str]:
        """
        Judge a single prediction.
        
        Args:
            question: The VQA question.
            gt_labels: Ground-truth label(s), may be comma-separated.
            pred_label: The predicted label.
            
        Returns:
            (parsed_json, raw_content) tuple.
            If parsing fails, returns default UNKNOWN output.
        """
        default = {
            "mapped_label": "UNKNOWN",
            "is_correct": False,
            "confidence": 0.0,
            "note": "parse_failed",
        }

        for attempt in range(2):
            try:
                resp = self._call(
                    question, gt_labels, pred_label,
                    add_json_hint=(attempt == 1)
                )
                content = (
                    resp.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                
                # Try to extract JSON from response
                parsed = self._parse_json_response(content)
                if parsed is not None:
                    return parsed, content
                    
            except Exception:
                continue

        return default, ""

    def _parse_json_response(self, content: str) -> Optional[Dict[str, Any]]:
        """Parse JSON from response content, handling markdown code blocks."""
        content = content.strip()
        
        # Try direct parsing
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code block
        if "```" in content:
            lines = content.split("\n")
            json_lines = []
            in_block = False
            for line in lines:
                if line.strip().startswith("```"):
                    if in_block:
                        break
                    in_block = True
                    continue
                if in_block:
                    json_lines.append(line)
            if json_lines:
                try:
                    return json.loads("\n".join(json_lines))
                except json.JSONDecodeError:
                    pass

        # Try finding JSON object in text
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(content[start:end])
            except json.JSONDecodeError:
                pass

        return None


def create_judge_from_env(
    sleep_seconds: float = 0.0,
    request_timeout: float = 60.0,
    max_retries: int = 5,
) -> Tuple[DeepSeekJudge, GlobalRateLimiter]:
    """
    Create a DeepSeek judge from environment variables.
    
    Required env vars:
        DEEPSEEK_API_KEY: API key for DeepSeek.
        
    Optional env vars:
        DEEPSEEK_BASE_URL: Base URL (default: https://api.deepseek.com/v1)
        DEEPSEEK_MODEL: Model name (default: deepseek-chat)
    """
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY environment variable is not set.")

    cfg = JudgeConfig(
        base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
        api_key=api_key,
        request_timeout=request_timeout,
        max_retries=max_retries,
        sleep_seconds=sleep_seconds,
    )

    limiter = GlobalRateLimiter(sleep_seconds)
    judge = DeepSeekJudge(cfg, limiter)

    return judge, limiter


if __name__ == "__main__":
    # Quick sanity check
    print(f"SSGVQA_LABELS count: {len(SSGVQA_LABELS)}")
    print(f"Sample labels: {SSGVQA_LABELS[:5]} ... {SSGVQA_LABELS[-5:]}")
