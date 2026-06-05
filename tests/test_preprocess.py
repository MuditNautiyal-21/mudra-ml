from __future__ import annotations

import numpy as np
import pandas as pd

from mudra_ml.preprocess import OutlierClipper, build_pipeline, plan_preprocess
from mudra_ml.profile import DataProfiler


def test_pipeline_fit_transform_shape(mixed_frame):
    profile = DataProfiler().profile(mixed_frame)
    pipeline, plan = build_pipeline(profile, target="churn")
    X = mixed_frame.drop(columns=["churn"])
    transformed = pipeline.fit_transform(X)
    assert transformed.shape[0] == len(X)
    assert "user_id" in plan.dropped


def test_high_missing_column_dropped(frame_with_missing):
    profile = DataProfiler().profile(frame_with_missing)
    _, plan = build_pipeline(profile, target="label")
    assert "mostly_empty" in plan.dropped


def test_no_nan_after_transform(frame_with_missing):
    profile = DataProfiler().profile(frame_with_missing)
    pipeline, _ = build_pipeline(profile, target="label")
    X = frame_with_missing.drop(columns=["label"])
    transformed = pipeline.fit_transform(X)
    assert not np.isnan(transformed).any()


def test_leakage_imputation_uses_train_only():
    """The imputed value must come from train data, never the full dataset.

    Train ages average 10. Test ages average 1000. If preprocessing leaked,
    the missing train value would be filled toward the global median rather
    than the train median.
    """
    train = pd.DataFrame({"age": [10.0, 10.0, np.nan, 10.0, 10.0], "y": [0, 1, 0, 1, 0]})
    test = pd.DataFrame({"age": [1000.0, 1000.0, 1000.0], "y": [1, 0, 1]})

    profile = DataProfiler().profile(pd.concat([train, test], ignore_index=True))
    pipeline, _ = build_pipeline(profile, target="y")

    pipeline.fit(train.drop(columns=["y"]), train["y"])
    imputer = pipeline.named_steps["columns"].named_transformers_["numeric"].named_steps["impute"]
    learned = float(imputer.statistics_[0])
    assert abs(learned - 10.0) < 1e-9


def test_leakage_scaler_mean_from_train_only():
    train = pd.DataFrame({"x": [0.0, 0.0, 0.0, 0.0], "y": [0, 1, 0, 1]})
    test = pd.DataFrame({"x": [100.0, 100.0], "y": [1, 0]})
    profile = DataProfiler().profile(pd.concat([train, test], ignore_index=True))
    pipeline, _ = build_pipeline(profile, target="y")
    pipeline.fit(train.drop(columns=["y"]), train["y"])
    scaler = pipeline.named_steps["columns"].named_transformers_["numeric"].named_steps["scale"]
    assert abs(float(scaler.mean_[0]) - 0.0) < 1e-9


def test_outlier_clipper_iqr_learns_bounds():
    clipper = OutlierClipper(strategy="iqr")
    data = np.array([[1.0], [2.0], [3.0], [4.0], [100.0]])
    clipper.fit(data)
    out = clipper.transform(data)
    assert out.max() < 100.0


def test_outlier_clipper_zscore():
    clipper = OutlierClipper(strategy="zscore")
    data = np.array([[1.0], [1.0], [1.0], [50.0]])
    clipper.fit(data)
    out = clipper.transform(np.array([[1000.0]]))
    assert out[0, 0] < 1000.0


def test_frequency_encoding_for_high_cardinality():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "high_card": [f"id_{i}" for i in rng.integers(0, 60, 300)],
            "y": rng.integers(0, 2, 300),
        }
    )
    profile = DataProfiler().profile(frame)
    plan = plan_preprocess(profile, "y", None, DataProfiler().log)
    assert "high_card" in plan.categorical_high


def test_missing_threshold_constraint(frame_with_missing):
    profile = DataProfiler().profile(frame_with_missing)
    plan = plan_preprocess(profile, "label", {"missing_threshold": 0.9}, DataProfiler().log)
    assert "mostly_empty" not in plan.dropped


def test_unseen_category_maps_to_zero():
    from mudra_ml.preprocess import FrequencyEncoder

    train = pd.DataFrame({"c": ["a", "a", "b"]})
    enc = FrequencyEncoder().fit(train)
    out = enc.transform(pd.DataFrame({"c": ["z"]}))
    assert out[0, 0] == 0.0
