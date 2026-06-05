"""MudraML: glass-box autonomous data science.

The decision engine that drives the pipeline is rule-based and statistical.
It is deterministic, logged, and explainable. The machine learning models are
the output it produces, not the mechanism by which it chooses what to do.
"""

from __future__ import annotations

from .core import Mudra, RunResult
from .evaluate import evaluate
from .goal import Goal, infer_goal
from .ingest import load
from .preprocess import build_pipeline
from .profile import DataProfile, DataProfiler
from .recommend import recommend_models
from .report import write_report

__version__ = "0.1.0"

__all__ = [
    "Mudra",
    "RunResult",
    "Goal",
    "infer_goal",
    "load",
    "DataProfiler",
    "DataProfile",
    "build_pipeline",
    "recommend_models",
    "evaluate",
    "write_report",
    "__version__",
]
