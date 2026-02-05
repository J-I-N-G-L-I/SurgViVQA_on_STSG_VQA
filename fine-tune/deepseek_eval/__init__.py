"""
DeepSeek-based LLM-as-a-judge evaluation for SurgViVQA on SSGVQA.

This package provides:
- judge.py: DeepSeek API client for semantic correctness evaluation
- metrics.py: Closed-set metric computation (Acc, mAP, mAR, mAF1, wF1)
- evaluate.py: Main evaluation script combining judge + metrics
"""
