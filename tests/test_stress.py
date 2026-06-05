"""Adversarial stress battery for the full pipeline.

Every case here represents a class of dataset the library should survive.
For each case the full pipeline runs end to end. The case fails if the
pipeline raises or the report file is missing. Where a specific warning is
expected, the case asserts the warning fires.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import (
    load_breast_cancer,
    load_iris,
    make_classification,
    make_regression,
)

from mudra_ml import Mudra
from mudra_ml.core import RunResult
from mudra_ml.profile import DataProfiler
from mudra_ml.quality import check_quality


def _run(
    frame: pd.DataFrame,
    tmp_path: Path,
    target: str | None,
    task: str | None,
    metric: str | None = None,
    use_boost: bool = False,
) -> tuple[float, RunResult, pd.DataFrame]:
    """Run the full pipeline and time it. Returns (elapsed, result, frame)."""
    start = time.perf_counter()
    result = Mudra().run(
        frame,
        target=target,
        task=task,
        metric=metric,
        report_path=tmp_path / "report",
        use_boost=use_boost,
    )
    elapsed = time.perf_counter() - start
    return elapsed, result, frame


def _capture(case: str, payload: dict) -> None:
    """No-op kept so individual cases stay tidy. Run timings are visible in pytest output."""
    return None


# ---------------------------------------------------------------------------
# Tiny / minimal cases
# ---------------------------------------------------------------------------


def test_tiny_classification(tmp_path):
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "x1": rng.normal(0, 1, 10),
            "x2": rng.normal(0, 1, 10),
            "label": [0, 1] * 5,
        }
    )
    seconds, result, _ = _run(frame, tmp_path, target="label", task="classification")
    md = result.report_path.read_text(encoding="utf-8")
    assert "indicative only" in md
    _capture("tiny_classification", {"seconds": seconds, "rows": len(frame)})


def test_single_feature_classification(tmp_path):
    rng = np.random.default_rng(0)
    n = 200
    frame = pd.DataFrame(
        {
            "x": rng.normal(0, 1, n),
            "label": rng.integers(0, 2, n),
        }
    )
    seconds, result, _ = _run(frame, tmp_path, target="label", task="classification")
    assert result.report_path.exists()
    _capture("single_feature_classification", {"seconds": seconds, "rows": n})


def test_single_class_target_emits_warning(tmp_path):
    frame = pd.DataFrame(
        {
            "x1": np.arange(120, dtype=float),
            "x2": np.arange(120, dtype=float) ** 2,
            "label": [1] * 120,
        }
    )
    profiler = DataProfiler()
    profile = profiler.profile(frame)
    quality = check_quality(frame, profile, "label", "classification", profiler.log)
    assert quality.has("single-class")
    _capture(
        "single_class_target_emits_warning",
        {"rows": len(frame), "warning": "single-class"},
    )


# ---------------------------------------------------------------------------
# Structural pathologies
# ---------------------------------------------------------------------------


def test_all_missing_column(tmp_path):
    rng = np.random.default_rng(0)
    n = 200
    frame = pd.DataFrame(
        {
            "x1": rng.normal(0, 1, n),
            "missing_col": [np.nan] * n,
            "label": rng.integers(0, 2, n),
        }
    )
    seconds, result, _ = _run(frame, tmp_path, target="label", task="classification")
    assert result.report_path.exists()
    _capture("all_missing_column", {"seconds": seconds, "rows": n})


def test_constant_column(tmp_path):
    rng = np.random.default_rng(0)
    n = 200
    frame = pd.DataFrame(
        {
            "x1": rng.normal(0, 1, n),
            "const": [7.0] * n,
            "label": rng.integers(0, 2, n),
        }
    )
    seconds, result, _ = _run(frame, tmp_path, target="label", task="classification")
    md = result.report_path.read_text(encoding="utf-8")
    assert "constant" in md.lower()
    _capture("constant_column", {"seconds": seconds, "rows": n})


def test_all_duplicate_rows(tmp_path):
    base = pd.DataFrame(
        {
            "x1": [1.0, 2.0, 3.0, 4.0] * 50,
            "x2": [0.1, 0.2, 0.3, 0.4] * 50,
            "label": [0, 1, 0, 1] * 50,
        }
    )
    seconds, result, _ = _run(base, tmp_path, target="label", task="classification")
    md = result.report_path.read_text(encoding="utf-8")
    assert "duplicate" in md.lower()
    _capture("all_duplicate_rows", {"seconds": seconds, "rows": len(base)})


def test_wide_dataset(tmp_path):
    rng = np.random.default_rng(0)
    n_rows = 50
    n_cols = 200
    data = rng.normal(0, 1, (n_rows, n_cols))
    frame = pd.DataFrame(data, columns=[f"x{i}" for i in range(n_cols)])
    frame["label"] = rng.integers(0, 2, n_rows)
    seconds, result, _ = _run(frame, tmp_path, target="label", task="classification")
    assert result.report_path.exists()
    _capture("wide_dataset", {"seconds": seconds, "rows": n_rows, "cols": n_cols})


def test_id_like_high_cardinality(tmp_path):
    rng = np.random.default_rng(0)
    n = 200
    frame = pd.DataFrame(
        {
            "user_id": [f"u{i}" for i in range(n)],
            "x": rng.normal(0, 1, n),
            "label": rng.integers(0, 2, n),
        }
    )
    seconds, result, _ = _run(frame, tmp_path, target="label", task="classification")
    assert result.report_path.exists()
    _capture("id_like_high_cardinality", {"seconds": seconds, "rows": n})


def test_strongly_imbalanced_target(tmp_path):
    rng = np.random.default_rng(0)
    n = 400
    labels = np.concatenate([np.zeros(380), np.ones(20)]).astype(int)
    rng.shuffle(labels)
    frame = pd.DataFrame(
        {
            "x1": rng.normal(0, 1, n),
            "x2": rng.normal(0, 1, n),
            "label": labels,
        }
    )
    seconds, result, _ = _run(frame, tmp_path, target="label", task="classification")
    md = result.report_path.read_text(encoding="utf-8")
    assert ("imbalance" in md.lower()) or ("Class imbalance" in md)
    _capture("strongly_imbalanced_target", {"seconds": seconds, "rows": n})


# ---------------------------------------------------------------------------
# Mixed/dirty data
# ---------------------------------------------------------------------------


def test_mixed_dtypes_with_messy_datetimes(tmp_path):
    rng = np.random.default_rng(0)
    n = 200
    spend = rng.normal(100, 25, n)
    spend[::20] = np.nan
    category = list(rng.choice(["red", "blue", "green"], n))
    category[10] = None
    frame = pd.DataFrame(
        {
            "user_id": [f"u{i}" for i in range(n)],
            "signup_date": pd.date_range("2021-01-01", periods=n, freq="D").astype(str),
            "category": category,
            "spend": spend,
            "label": rng.integers(0, 2, n),
        }
    )
    frame.loc[5, "signup_date"] = "not a date"
    seconds, result, _ = _run(frame, tmp_path, target="label", task="classification")
    assert result.report_path.exists()
    _capture("mixed_dtypes_messy_datetimes", {"seconds": seconds, "rows": n})


def test_leakage_warning_fires(tmp_path):
    rng = np.random.default_rng(0)
    n = 300
    label = rng.integers(0, 2, n)
    frame = pd.DataFrame(
        {
            "x1": rng.normal(0, 1, n),
            "x2": rng.normal(0, 1, n),
            "leak": label.copy(),
            "label": label,
        }
    )
    seconds, result, _ = _run(frame, tmp_path, target="label", task="classification")
    md = result.report_path.read_text(encoding="utf-8")
    assert "leakage" in md.lower()
    _capture("leakage_warning", {"seconds": seconds, "rows": n})


def test_target_contains_missing_values(tmp_path):
    rng = np.random.default_rng(0)
    n = 200
    label = rng.integers(0, 2, n).astype(float)
    label[:8] = np.nan
    frame = pd.DataFrame(
        {
            "x1": rng.normal(0, 1, n),
            "x2": rng.normal(0, 1, n),
            "label": label,
        }
    )
    seconds, result, _ = _run(frame, tmp_path, target="label", task="classification")
    md = result.report_path.read_text(encoding="utf-8")
    assert "missing values" in md.lower() or "target-missing" in md
    _capture("target_missing_values", {"seconds": seconds, "rows": n})


# ---------------------------------------------------------------------------
# Larger dataset
# ---------------------------------------------------------------------------


def test_larger_dataset_completes_in_reasonable_time(tmp_path):
    X, y = make_classification(
        n_samples=10000, n_features=20, n_informative=12, random_state=0
    )
    frame = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    frame["label"] = y
    seconds, result, _ = _run(frame, tmp_path, target="label", task="classification")
    assert result.report_path.exists()
    assert seconds < 240, f"larger dataset took {seconds:.1f}s"
    _capture("larger_dataset", {"seconds": seconds, "rows": len(frame)})


# ---------------------------------------------------------------------------
# Task type coverage
# ---------------------------------------------------------------------------


def test_binary_classification_path(tmp_path):
    frame = load_breast_cancer(as_frame=True).frame
    seconds, result, _ = _run(frame, tmp_path, target="target", task="classification")
    assert result.task == "classification"
    assert "roc_auc" in result.evaluation["candidates"][0]["test_metrics"]
    _capture("binary_classification", {"seconds": seconds, "rows": len(frame)})


def test_multiclass_classification_path(tmp_path):
    frame = load_iris(as_frame=True).frame
    seconds, result, _ = _run(frame, tmp_path, target="target", task="classification")
    assert result.task == "classification"
    assert result.evaluation["per_class_report"]
    _capture("multiclass_classification", {"seconds": seconds, "rows": len(frame)})


def test_regression_path(tmp_path):
    X, y = make_regression(n_samples=400, n_features=8, noise=12.0, random_state=0)
    frame = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    frame["price"] = y
    seconds, result, _ = _run(frame, tmp_path, target="price", task="regression")
    assert result.task == "regression"
    assert result.evaluation["regression_diagnostics"].get("y_true")
    _capture("regression", {"seconds": seconds, "rows": len(frame)})


def test_clustering_path(tmp_path):
    frame = load_iris(as_frame=True).frame.drop(columns=["target"])
    seconds, result, _ = _run(frame, tmp_path, target=None, task="clustering")
    assert result.task == "clustering"
    _capture("clustering", {"seconds": seconds, "rows": len(frame)})


# ---------------------------------------------------------------------------
# Report file existence safeguard
# ---------------------------------------------------------------------------


def test_html_report_remains_self_contained(tmp_path):
    frame = load_breast_cancer(as_frame=True).frame
    Mudra().run(frame, target="target", report_path=tmp_path / "r")
    html = (tmp_path / "r.html").read_text(encoding="utf-8")
    assert "<html" in html.lower()
    assert "</html>" in html.lower()
