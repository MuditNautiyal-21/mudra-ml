"""MudraML: glass-box autonomous data science.

The decision engine that drives the pipeline is rule-based and statistical.
It is deterministic, logged, and explainable. The machine learning models are
the output it produces, not the mechanism by which it chooses what to do.
"""

from __future__ import annotations

from .core import Mudra, RunResult
from .errors import DataError, MudraError
from .evaluate import evaluate
from .goal import Goal, infer_goal
from .ingest import load
from .preprocess import build_pipeline
from .profile import DataProfile, DataProfiler
from .quality import QualityReport, QualityWarning, check_quality
from .recommend import recommend_models
from .report import write_report

__version__ = "0.3.0"

__all__ = [
    "Mudra",
    "RunResult",
    "MudraError",
    "DataError",
    "Goal",
    "infer_goal",
    "load",
    "DataProfiler",
    "DataProfile",
    "QualityReport",
    "QualityWarning",
    "check_quality",
    "build_pipeline",
    "recommend_models",
    "evaluate",
    "write_report",
    "__version__",
]
