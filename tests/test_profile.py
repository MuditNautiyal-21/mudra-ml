from __future__ import annotations

import numpy as np
import pandas as pd

from mudra_ml.profile import (
    BOOLEAN,
    CATEGORICAL,
    DATETIME,
    ID,
    NUMERIC,
    TEXT,
    DataProfiler,
)


def test_numeric_and_categorical_inference(mixed_frame):
    profile = DataProfiler().profile(mixed_frame)
    assert profile.column("age").inferred_type == NUMERIC
    assert profile.column("spend").inferred_type == NUMERIC
    assert profile.column("city").inferred_type == CATEGORICAL


def test_id_detection(mixed_frame):
    profile = DataProfiler().profile(mixed_frame)
    assert profile.column("user_id").inferred_type == ID


def test_datetime_detection(mixed_frame):
    profile = DataProfiler().profile(mixed_frame)
    assert profile.column("signup").inferred_type == DATETIME


def test_text_detection(mixed_frame):
    profile = DataProfiler().profile(mixed_frame)
    assert profile.column("note").inferred_type == TEXT


def test_boolean_detection():
    frame = pd.DataFrame({"flag": ["yes", "no", "yes", "no"], "x": [1, 2, 3, 4]})
    profile = DataProfiler().profile(frame)
    assert profile.column("flag").inferred_type == BOOLEAN


def test_missingness_recorded(frame_with_missing):
    profile = DataProfiler().profile(frame_with_missing)
    assert profile.column("age").missing_count == 30
    assert profile.column("mostly_empty").missing_fraction > 0.5


def test_candidate_targets_prefers_named_column():
    frame = pd.DataFrame(
        {
            "feature": np.arange(100),
            "churn": np.random.default_rng(0).integers(0, 2, 100),
        }
    )
    profile = DataProfiler().profile(frame)
    assert "churn" in profile.candidate_targets


def test_numeric_stats_present(regression_frame):
    profile = DataProfiler().profile(regression_frame)
    stats = profile.column("price").stats
    assert "mean" in stats and "std" in stats and "median" in stats


def test_columns_of_type(mixed_frame):
    profile = DataProfiler().profile(mixed_frame)
    numeric_cols = profile.columns_of_type(NUMERIC)
    assert "age" in numeric_cols and "spend" in numeric_cols


def test_decision_log_populated(mixed_frame):
    profiler = DataProfiler()
    profiler.profile(mixed_frame)
    assert len(profiler.log) > 0
    assert any(d.rule == "type-inference" for d in profiler.log)
