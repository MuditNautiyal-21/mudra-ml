"""Tests for the data-quality checks and the leakage detector."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mudra_ml.decisions import DecisionLog
from mudra_ml.profile import DataProfile, DataProfiler
from mudra_ml.quality import (
    QualityReport,
    assess_dataset,
    check_quality,
)


def _profile(frame: pd.DataFrame) -> tuple[DataProfiler, DataProfile]:
    profiler = DataProfiler()
    return profiler, profiler.profile(frame)


def test_small_dataset_warning_fires():
    frame = pd.DataFrame(
        {"x": range(10), "y": [0, 1] * 5}
    )
    profiler, profile = _profile(frame)
    quality = check_quality(frame, profile, "y", "classification", profiler.log)
    assert quality.has("small-dataset")
    codes = [w.code for w in quality.warnings]
    assert "small-dataset" in codes


def test_constant_column_warning_fires():
    frame = pd.DataFrame(
        {
            "x": [1] * 200,
            "z": np.arange(200) % 5,
            "label": np.random.default_rng(0).integers(0, 2, 200),
        }
    )
    profiler, profile = _profile(frame)
    quality = check_quality(frame, profile, "label", "classification", profiler.log)
    assert quality.has("constant-column")


def test_duplicate_rows_warning_fires():
    base = pd.DataFrame(
        {"x": [1, 2, 3] * 50, "y": [0, 1, 0] * 50}
    )
    profiler, profile = _profile(base)
    quality = check_quality(base, profile, "y", "classification", profiler.log)
    assert quality.has("duplicate-rows")


def test_leakage_warning_fires_when_feature_equals_target():
    rng = np.random.default_rng(0)
    target = rng.integers(0, 2, 200)
    frame = pd.DataFrame(
        {
            "x": rng.normal(0, 1, 200),
            "leak": target.copy(),
            "label": target,
        }
    )
    profiler, profile = _profile(frame)
    quality = check_quality(frame, profile, "label", "classification", profiler.log)
    assert quality.has("leakage-suspect")
    suspects = quality.leakage_suspects
    assert any(s["feature"] == "leak" for s in suspects)


def test_single_class_warning_fires():
    frame = pd.DataFrame({"x": range(200), "y": [1] * 200})
    profiler, profile = _profile(frame)
    quality = check_quality(frame, profile, "y", "classification", profiler.log)
    assert quality.has("single-class")


def test_imbalanced_classes_warning_fires():
    rng = np.random.default_rng(0)
    label = np.concatenate([np.zeros(190), np.ones(10)]).astype(int)
    frame = pd.DataFrame({"x": rng.normal(0, 1, 200), "y": label})
    profiler, profile = _profile(frame)
    quality = check_quality(frame, profile, "y", "classification", profiler.log)
    assert quality.has("imbalanced-classes")


def test_missing_target_warning():
    rng = np.random.default_rng(0)
    label = rng.integers(0, 2, 200).astype(float)
    label[:5] = np.nan
    frame = pd.DataFrame({"x": rng.normal(0, 1, 200), "y": label})
    profiler, profile = _profile(frame)
    quality = check_quality(frame, profile, "y", "classification", profiler.log)
    assert quality.has("target-missing")


def test_next_steps_present_when_no_issues():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "x": rng.normal(0, 1, 500),
            "y": rng.integers(0, 2, 500),
        }
    )
    profiler, profile = _profile(frame)
    quality = check_quality(frame, profile, "y", "classification", profiler.log)
    assert quality.next_steps


def test_assess_dataset_returns_pair():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"x": rng.normal(0, 1, 200), "y": rng.integers(0, 2, 200)})
    profile, quality = assess_dataset(frame, "y", "classification")
    assert profile.n_rows == 200
    assert isinstance(quality, QualityReport)


def test_quality_decisions_recorded_in_log():
    frame = pd.DataFrame({"x": [1] * 50, "y": [0] * 50})
    log = DecisionLog()
    profile = DataProfiler(log).profile(frame)
    check_quality(frame, profile, "y", "classification", log)
    assert any(d.stage == "quality" for d in log)
