"""Tests for cross-validation based model selection.

The held-out test set must not influence which candidate is chosen. These
tests pin two properties:
  - The pure selection helper picks the candidate with the best CV score.
  - The full evaluate path scores the test set only for the selected model
    (non-winners have no test metrics) and ranks the candidates list by CV.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

from mudra_ml.evaluate import (
    CandidateResult,
    _select_best_by_cv,
    evaluate,
)
from mudra_ml.recommend import Candidate, recommend_models


def _candidate(name: str, cv_mean: float, test_f1: float) -> CandidateResult:
    return CandidateResult(
        name=name,
        cv_score=cv_mean,
        cv_mean=cv_mean,
        cv_std=0.01,
        cv_scores=[cv_mean - 0.005, cv_mean + 0.005],
        test_metrics={"f1": test_f1},
        train_metrics={},
        best_params={},
    )


def test_select_best_by_cv_picks_cv_winner_not_test_winner():
    """The headline regression test.

    Candidate A has the higher cross-validation mean. Candidate B has the
    higher held-out test metric. The selection must follow CV, so A wins.
    """
    cv_winner = _candidate("A_cv_winner", cv_mean=0.92, test_f1=0.50)
    test_winner = _candidate("B_test_winner", cv_mean=0.70, test_f1=0.99)
    best = _select_best_by_cv([cv_winner, test_winner], "f1")
    assert best.name == "A_cv_winner"


def test_select_best_by_cv_handles_lower_is_better_metrics():
    """For rmse and mae a smaller CV value is better."""
    low = _candidate("low_rmse", cv_mean=2.5, test_f1=0.0)
    high = _candidate("high_rmse", cv_mean=15.0, test_f1=0.0)
    best = _select_best_by_cv([low, high], "rmse")
    assert best.name == "low_rmse"


def test_select_best_by_cv_raises_on_empty_list():
    with pytest.raises(ValueError):
        _select_best_by_cv([], "f1")


def test_select_best_by_cv_is_deterministic_on_ties():
    """When candidates tie, max returns the first match."""
    first = _candidate("first", cv_mean=0.8, test_f1=0.0)
    second = _candidate("second", cv_mean=0.8, test_f1=0.0)
    best = _select_best_by_cv([first, second], "f1")
    assert best.name == "first"


class _FixedPredictionClassifier(BaseEstimator, ClassifierMixin):
    """Predict a fixed array. fit ignores its inputs.

    The predict_proba method returns a deterministic distribution so the
    classification metric helpers do not crash on the fake estimator.
    """

    def __init__(self, predictions: np.ndarray) -> None:
        self.predictions = np.asarray(predictions)

    def fit(self, X, y=None):
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1] if hasattr(X, "shape") else 1
        return self

    def predict(self, X):
        n = X.shape[0] if hasattr(X, "shape") else len(X)
        return self.predictions[:n]

    def predict_proba(self, X):
        preds = self.predict(X)
        proba = np.zeros((len(preds), 2))
        proba[np.arange(len(preds)), preds.astype(int)] = 1.0
        return proba


def test_evaluate_selects_cv_winner_when_test_winner_differs():
    """End-to-end proof using fake classifiers with controlled outputs.

    The CV winner gets every training prediction right and every test
    prediction wrong. The test winner gets the opposite. Selection must
    follow CV, so the CV winner is chosen and only it gets test metrics.
    """
    rng = np.random.default_rng(0)
    n_train = 80
    n_test = 20
    X_train = rng.normal(0, 1, (n_train, 4))
    y_train = rng.integers(0, 2, n_train)
    X_test = rng.normal(0, 1, (n_test, 4))
    y_test = rng.integers(0, 2, n_test)

    cv_winner = Candidate(
        name="cv_winner",
        estimator=_FixedPredictionClassifier(np.concatenate([y_train, y_test])),
        param_grid={},
    )
    test_winner = Candidate(
        name="test_winner",
        estimator=_FixedPredictionClassifier(
            np.concatenate([1 - y_train, y_test])
        ),
        param_grid={},
    )

    result = evaluate(
        candidates=[cv_winner, test_winner],
        task="classification",
        metric="f1",
        feature_names=[f"f{i}" for i in range(X_train.shape[1])],
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        cv=2,
    )

    assert result.best_name == "cv_winner"
    winner = result.best
    other = next(c for c in result.candidates if c.name != "cv_winner")
    # The held-out test set was scored only for the winner.
    assert winner.test_metrics, "winner should have test metrics"
    assert other.test_metrics == {}, "non-winner must not have test metrics"
    assert winner.cv_mean >= other.cv_mean


def test_candidates_table_ranked_by_cross_validation():
    """The candidates list must be sorted by CV best first."""
    X, y = make_classification(n_samples=300, n_features=8, random_state=0)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)
    candidates = recommend_models(
        "classification", len(X_tr), X_tr.shape[1], use_boost=False
    )
    result = evaluate(
        candidates,
        "classification",
        "f1",
        [f"f{i}" for i in range(X_tr.shape[1])],
        X_tr,
        y_tr,
        X_te,
        y_te,
    )
    cv_means = [c.cv_mean for c in result.candidates]
    assert cv_means == sorted(cv_means, reverse=True)
    # Only the first candidate (the winner) carries test metrics.
    assert result.candidates[0].test_metrics
    for cand in result.candidates[1:]:
        assert cand.test_metrics == {}


def test_evaluate_picks_cv_winner_in_real_dataset():
    """The selected name must match the CV best across all candidates."""
    X, y = make_regression(n_samples=300, n_features=6, noise=15.0, random_state=0)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)
    candidates = recommend_models(
        "regression", len(X_tr), X_tr.shape[1], use_boost=False
    )
    result = evaluate(
        candidates,
        "regression",
        "rmse",
        [f"f{i}" for i in range(X_tr.shape[1])],
        X_tr,
        y_tr,
        X_te,
        y_te,
    )
    # rmse is lower-is-better; min cv_mean wins.
    expected = min(result.candidates, key=lambda c: c.cv_mean)
    assert result.best_name == expected.name


def test_report_table_uses_cv_only(tmp_path):
    """The Candidates compared section must not present a per-candidate test column."""
    from mudra_ml import Mudra

    rng = np.random.default_rng(0)
    n = 300
    frame = pd.DataFrame(
        {
            "x1": rng.normal(0, 1, n),
            "x2": rng.normal(0, 1, n),
            "label": rng.integers(0, 2, n),
        }
    )
    result = Mudra().run(frame, target="label", report_path=tmp_path / "r")
    md = result.report_path.read_text(encoding="utf-8")
    assert "Ranked by cross-validation" in md
    assert "Selected |" in md
    # The old column header is gone.
    assert "Test f1 |" not in md
    assert "Test rmse |" not in md
