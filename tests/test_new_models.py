"""Every model added in 0.4.0 trains, can be selected, and survives dirty data.

Each test builds the candidate directly from the pool so the model is
exercised even when the size-based shortlist rules would not pick it for the
small fixture dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mudra_ml.decisions import DecisionLog
from mudra_ml.evaluate import evaluate
from mudra_ml.ingest import coerce_numeric_like
from mudra_ml.preprocess import build_pipeline
from mudra_ml.profile import DataProfiler
from mudra_ml.recommend import _classification_candidates, _regression_candidates

NEW_CLASSIFIERS = ["svc", "k_nearest_neighbors", "gaussian_naive_bayes", "extra_trees"]
NEW_REGRESSORS = ["svr", "k_nearest_neighbors", "extra_trees", "elastic_net"]


def _classification_arrays(seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(120, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X[:90], y[:90], X[90:], y[90:]


def _regression_arrays(seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(120, 4))
    y = 3.0 * X[:, 0] - 2.0 * X[:, 1] + rng.normal(scale=0.1, size=120)
    return X[:90], y[:90], X[90:], y[90:]


@pytest.mark.parametrize("name", NEW_CLASSIFIERS)
def test_new_classifier_trains_and_is_selected(name):
    candidate = _classification_candidates(42)[name]
    X_train, y_train, X_test, y_test = _classification_arrays()
    result = evaluate(
        candidates=[candidate],
        task="classification",
        metric="f1",
        feature_names=[f"f{i}" for i in range(4)],
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    assert result.best_name == name
    assert result.best.test_metrics["accuracy"] > 0.6


@pytest.mark.parametrize("name", NEW_REGRESSORS)
def test_new_regressor_trains_and_is_selected(name):
    candidate = _regression_candidates(42)[name]
    X_train, y_train, X_test, y_test = _regression_arrays()
    result = evaluate(
        candidates=[candidate],
        task="regression",
        metric="rmse",
        feature_names=[f"f{i}" for i in range(4)],
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    assert result.best_name == name
    assert result.best.test_metrics["r2"] > 0.5


def _dirty_frame(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 160
    amount = [f"${v:,.2f}" if i % 3 else f"{v:.2f}" for i, v in enumerate(rng.uniform(10, 9000, n))]
    score = rng.normal(size=n)
    frame = pd.DataFrame(
        {
            "amount": amount,
            "score": score,
            "city": rng.choice(["york", "leeds", "bath"], size=n),
            "target": (score > 0).astype(int),
        }
    )
    frame.loc[::11, "score"] = np.nan
    frame.loc[::13, "amount"] = "--"
    return frame


def _transformed_dirty(target_kind: str):
    frame = _dirty_frame()
    if target_kind == "regression":
        frame["target"] = frame["score"].fillna(0.0) * 10.0 + 5.0
    log = DecisionLog()
    clean = coerce_numeric_like(frame, log)
    profile = DataProfiler(log).profile(clean)
    pipeline, _ = build_pipeline(profile, "target", None, log)
    X = clean.drop(columns=["target"])
    y = clean["target"].to_numpy()
    X_t = pipeline.fit_transform(X, y)
    return X_t, y


@pytest.mark.parametrize("name", NEW_CLASSIFIERS)
def test_new_classifier_survives_dirty_data(name):
    X_t, y = _transformed_dirty("classification")
    candidate = _classification_candidates(42)[name]
    result = evaluate(
        candidates=[candidate],
        task="classification",
        metric="f1",
        feature_names=[f"f{i}" for i in range(X_t.shape[1])],
        X_train=X_t[:120],
        y_train=y[:120],
        X_test=X_t[120:],
        y_test=y[120:],
    )
    assert result.best_name == name


@pytest.mark.parametrize("name", NEW_REGRESSORS)
def test_new_regressor_survives_dirty_data(name):
    X_t, y = _transformed_dirty("regression")
    candidate = _regression_candidates(42)[name]
    result = evaluate(
        candidates=[candidate],
        task="regression",
        metric="rmse",
        feature_names=[f"f{i}" for i in range(X_t.shape[1])],
        X_train=X_t[:120],
        y_train=y[:120],
        X_test=X_t[120:],
        y_test=y[120:],
    )
    assert result.best_name == name
    assert "rmse" in result.best.test_metrics
