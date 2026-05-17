"""Evaluation harness for measuring orchestration quality.

Provides multiplicative scoring, LLM-based code quality judging,
failure taxonomy, and golden benchmark task management.
"""

from __future__ import annotations

from bernstein.eval.baseline import EvalBaseline, load_baseline, save_baseline
from bernstein.eval.harness import EvalHarness, EvalResult, EvalTier
from bernstein.eval.incident_synthesizer import (
    IncidentEvalCase,
    IncidentSyncResult,
    IncidentSynthesizer,
    run_incident_eval_gate,
)

__all__ = [
    "EvalBaseline",
    "EvalHarness",
    "EvalResult",
    "EvalTier",
    "IncidentEvalCase",
    "IncidentSyncResult",
    "IncidentSynthesizer",
    "load_baseline",
    "run_incident_eval_gate",
    "save_baseline",
]
