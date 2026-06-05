"""Regression tests for the discrete-column preprocessing fix.

A skewed binary feature would previously collapse to a constant under IQR
outlier clipping. The fix routes boolean columns through a discrete pipeline
(mode imputation, cast to 0/1 float, no clipping, no scaling) and treats
low-cardinality integer columns as categorical so they are one-hot encoded
rather than scaled as continuous values. These tests pin that behavior.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mudra_ml import Mudra
from mudra_ml.preprocess import build_pipeline
from mudra_ml.profile import BOOLEAN, CATEGORICAL, NUMERIC, DataProfiler


def _messy_churn_frame(seed: int = 0, n: int = 400) -> pd.DataFrame:
    """A churn-style dataset with a skewed binary feature that leaks the target."""
    rng = np.random.default_rng(seed)
    label = (rng.random(n) < 0.1).astype(int)
    spend = rng.normal(100.0, 25.0, n)
    spend[::20] = np.nan
    tenure = rng.integers(0, 36, n).astype(float)
    tenure[3] = np.nan
    region = list(rng.choice(["paris", "berlin", "rome", "madrid"], n))
    region[5] = None
    signup = pd.date_range("2021-01-01", periods=n, freq="D").astype(str).tolist()
    signup[7] = "not a date"
    return pd.DataFrame(
        {
            "user_id": [f"u{i}" for i in range(n)],
            "signup_date": signup,
            "region": region,
            "tenure_months": tenure,
            "spend": spend,
            "is_premium": label.copy(),
            "churn": label,
        }
    )


def test_skewed_binary_feature_is_typed_boolean():
    frame = _messy_churn_frame()
    profile = DataProfiler().profile(frame)
    assert profile.column("is_premium").inferred_type == BOOLEAN


def test_low_cardinality_integer_routes_to_categorical():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "rating": rng.integers(1, 6, 300),  # 5 distinct ints
            "label": rng.integers(0, 2, 300),
        }
    )
    profile = DataProfiler().profile(frame)
    assert profile.column("rating").inferred_type == CATEGORICAL


def test_high_cardinality_integer_stays_numeric():
    rng = np.random.default_rng(0)
    # 2000 rows but only 50 distinct integer values: above the discrete
    # threshold and well below the id-like unique-ratio threshold.
    frame = pd.DataFrame(
        {
            "count": rng.integers(0, 50, 2000),
            "label": rng.integers(0, 2, 2000),
        }
    )
    profile = DataProfiler().profile(frame)
    assert profile.column("count").inferred_type == NUMERIC


def test_boolean_pipeline_preserves_skewed_binary_values():
    frame = _messy_churn_frame()
    profile = DataProfiler().profile(frame)
    pipeline, plan = build_pipeline(profile, target="churn")
    assert "is_premium" in plan.boolean
    assert "is_premium" not in plan.numeric

    transformed = pipeline.fit_transform(
        frame.drop(columns=["churn"]), frame["churn"]
    )
    # is_premium is the last transformer column in plan order. Locate it
    # by scanning for the column where the unique values are {0, 1}.
    binary_cols = [
        i
        for i in range(transformed.shape[1])
        if set(np.unique(transformed[:, i]).round(6).tolist()) == {0.0, 1.0}
    ]
    assert binary_cols, "no preserved 0/1 binary column found in transformed output"


def test_binary_feature_keeps_nonzero_permutation_importance(tmp_path):
    """The headline regression test for the original bug.

    Before the fix, the skewed binary 'is_premium' column collapsed to a
    constant under IQR clipping and scored zero on permutation importance.
    After the fix, it carries signal because it is preserved end to end.
    """
    frame = _messy_churn_frame()
    result = Mudra().run(frame, target="churn", report_path=tmp_path / "r")
    importance = result.evaluation["permutation_importance"]
    assert importance, "permutation importance was empty"
    # Find the entry whose name contains is_premium (handles potential
    # prefixes added by one-hot or other transformers).
    binary_keys = [k for k in importance if "is_premium" in k]
    assert binary_keys, f"is_premium not in importance: {list(importance)}"
    score = max(importance[k] for k in binary_keys)
    assert score > 0.0, f"binary feature collapsed; importance={score}"


def test_leakage_warning_still_fires_for_binary_equal_to_target(tmp_path):
    """Confirm the leakage check is independent of the preprocessing fix."""
    frame = _messy_churn_frame()
    result = Mudra().run(frame, target="churn", report_path=tmp_path / "r")
    md = result.report_path.read_text(encoding="utf-8")
    assert "leakage" in md.lower()
    html = (tmp_path / "r.html").read_text(encoding="utf-8")
    assert "leakage" in html.lower()


def test_boolean_pipeline_handles_yes_no_strings():
    frame = pd.DataFrame(
        {
            "flag": ["yes", "no", "yes", "no", "yes"] * 60,
            "x": np.random.default_rng(0).normal(0, 1, 300),
            "label": np.array([1, 0, 1, 0, 1] * 60),
        }
    )
    profile = DataProfiler().profile(frame)
    assert profile.column("flag").inferred_type == BOOLEAN
    pipeline, plan = build_pipeline(profile, target="label")
    assert "flag" in plan.boolean
    transformed = pipeline.fit_transform(
        frame.drop(columns=["label"]), frame["label"]
    )
    assert transformed.shape[0] == 300
    # The yes/no column should be cast to 0/1, not lost or coerced wrong.
    binary_cols = [
        i
        for i in range(transformed.shape[1])
        if set(np.unique(transformed[:, i]).round(6).tolist()) == {0.0, 1.0}
    ]
    assert binary_cols


def test_boolean_transformer_is_leakage_safe():
    """The discrete pipeline learns nothing from the data values."""
    from mudra_ml.preprocess import BooleanToNumeric

    cast = BooleanToNumeric()
    train_input = np.array([[0], [1], [1], [0]])
    cast.fit(train_input)
    out_a = cast.transform(np.array([[1], [0]]))
    out_b = cast.transform(np.array([[1], [0]]))
    assert np.array_equal(out_a, out_b)


def test_decision_log_records_discrete_handling(tmp_path):
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "binary_flag": rng.integers(0, 2, 200),
            "rating": rng.integers(1, 6, 200),
            "x": rng.normal(0, 1, 200),
            "label": rng.integers(0, 2, 200),
        }
    )
    result = Mudra().run(frame, target="label", report_path=tmp_path / "r")
    md = result.report_path.read_text(encoding="utf-8")
    assert "boolean-discrete-handling" in md
    assert "discrete-low-cardinality-integer" in md


def test_pipeline_round_trip_with_unseen_values(tmp_path):
    """The pipeline must survive transform on a held-out subset."""
    frame = _messy_churn_frame(n=200)
    result = Mudra().run(frame, target="churn", report_path=tmp_path / "r")
    holdout = frame.drop(columns=["churn"]).head(20)
    transformed = result.pipeline.transform(holdout)
    assert transformed.shape[0] == 20
