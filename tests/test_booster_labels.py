"""Booster candidates work with string class labels.

XGBoost rejects labels that are not consecutive integers, so its candidate
is wrapped to encode labels during fit and return the original labels from
predict. These tests run only when the optional boosters are installed; the
offline suite skips them cleanly otherwise.
"""

from __future__ import annotations

import numpy as np
import pytest

from mudra_ml.decisions import DecisionLog
from mudra_ml.evaluate import evaluate
from mudra_ml.recommend import _boost_candidates


def _string_label_data(seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(160, 5))
    y = np.where(X[:, 0] + 0.5 * X[:, 1] > 0, "approved", "declined")
    return X[:120], y[:120], X[120:], y[120:]


def _candidate(name: str):
    pytest.importorskip(name)
    candidates = _boost_candidates("classification", 42, DecisionLog())
    return candidates[name]


@pytest.mark.parametrize("name", ["xgboost", "lightgbm", "catboost"])
def test_booster_trains_and_predicts_string_labels(name):
    candidate = _candidate(name)
    X_train, y_train, X_test, y_test = _string_label_data()
    result = evaluate(
        candidates=[candidate],
        task="classification",
        metric="f1",
        feature_names=[f"f{i}" for i in range(5)],
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    assert result.best_name == name
    assert result.best.test_metrics["accuracy"] > 0.7
    preds = result.best_estimator.predict(X_test)
    assert set(np.unique(preds)) <= {"approved", "declined"}


@pytest.mark.parametrize("name", ["xgboost", "lightgbm", "catboost"])
def test_booster_predict_proba_columns_follow_classes(name):
    candidate = _candidate(name)
    X_train, y_train, X_test, _ = _string_label_data()
    candidate.estimator.fit(X_train, y_train)
    proba = candidate.estimator.predict_proba(X_test)
    assert proba.shape == (len(X_test), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert list(candidate.estimator.classes_) == ["approved", "declined"]
