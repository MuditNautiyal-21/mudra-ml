"""Tests for the trust and diagnostic additions in evaluate.py."""

from __future__ import annotations

import pandas as pd
from sklearn.datasets import load_iris, make_classification, make_regression
from sklearn.model_selection import train_test_split

from mudra_ml.evaluate import evaluate
from mudra_ml.recommend import recommend_models


def _split(frame: pd.DataFrame, target: str):
    X = frame.drop(columns=[target]).to_numpy()
    y = frame[target].to_numpy()
    return train_test_split(X, y, test_size=0.25, random_state=42)


def test_classification_result_includes_baseline_and_per_class():
    iris = load_iris(as_frame=True).frame
    X_tr, X_te, y_tr, y_te = _split(iris, "target")
    candidates = recommend_models("classification", len(X_tr), X_tr.shape[1], use_boost=False)
    result = evaluate(
        candidates, "classification", "f1",
        [f"f{i}" for i in range(X_tr.shape[1])],
        X_tr, y_tr, X_te, y_te,
    )
    assert result.baseline_name == "dummy_most_frequent"
    assert "f1" in result.baseline_metrics
    assert result.per_class_report
    assert all(
        "precision" in v for k, v in result.per_class_report.items()
        if k not in ("accuracy",)
    )


def test_classification_cv_mean_std_populated():
    iris = load_iris(as_frame=True).frame
    X_tr, X_te, y_tr, y_te = _split(iris, "target")
    candidates = recommend_models("classification", len(X_tr), X_tr.shape[1], use_boost=False)
    result = evaluate(
        candidates, "classification", "f1",
        [f"f{i}" for i in range(X_tr.shape[1])],
        X_tr, y_tr, X_te, y_te,
    )
    for cand in result.candidates:
        assert cand.cv_mean is not None
        assert cand.cv_std >= 0.0
        assert len(cand.cv_scores) >= 2


def test_train_metrics_present_alongside_test():
    iris = load_iris(as_frame=True).frame
    X_tr, X_te, y_tr, y_te = _split(iris, "target")
    candidates = recommend_models("classification", len(X_tr), X_tr.shape[1], use_boost=False)
    result = evaluate(
        candidates, "classification", "f1",
        [f"f{i}" for i in range(X_tr.shape[1])],
        X_tr, y_tr, X_te, y_te,
    )
    assert result.best.train_metrics
    assert "accuracy" in result.best.train_metrics


def test_binary_classification_has_roc_and_pr_curves():
    X, y = make_classification(n_samples=400, n_features=10, random_state=0)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)
    candidates = recommend_models("classification", len(X_tr), X_tr.shape[1], use_boost=False)
    result = evaluate(
        candidates, "classification", "f1",
        [f"f{i}" for i in range(X_tr.shape[1])],
        X_tr, y_tr, X_te, y_te,
    )
    assert result.roc_curve.get("fpr")
    assert result.pr_curve.get("recall")
    assert "auc" in result.roc_curve


def test_regression_diagnostics_populated():
    X, y = make_regression(n_samples=200, n_features=6, noise=8.0, random_state=0)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)
    candidates = recommend_models("regression", len(X_tr), X_tr.shape[1], use_boost=False)
    result = evaluate(
        candidates, "regression", "rmse",
        [f"f{i}" for i in range(X_tr.shape[1])],
        X_tr, y_tr, X_te, y_te,
    )
    diag = result.regression_diagnostics
    assert diag.get("y_true")
    assert "residual_mean" in diag
    assert "residual_std" in diag


def test_permutation_importance_populated():
    iris = load_iris(as_frame=True).frame
    X_tr, X_te, y_tr, y_te = _split(iris, "target")
    candidates = recommend_models("classification", len(X_tr), X_tr.shape[1], use_boost=False)
    result = evaluate(
        candidates, "classification", "f1",
        [f"f{i}" for i in range(X_tr.shape[1])],
        X_tr, y_tr, X_te, y_te,
    )
    assert result.permutation_importance
    assert result.permutation_importance_std
    keys_a = set(result.permutation_importance.keys())
    keys_b = set(result.permutation_importance_std.keys())
    assert keys_a == keys_b


def test_small_sample_warning_fires():
    X, y = make_classification(n_samples=30, n_features=4, random_state=0)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    candidates = recommend_models("classification", len(X_tr), X_tr.shape[1], use_boost=False)
    result = evaluate(
        candidates, "classification", "f1",
        [f"f{i}" for i in range(X_tr.shape[1])],
        X_tr, y_tr, X_te, y_te,
    )
    assert result.small_sample_warning is True
    assert result.test_set_size <= 50


def test_regression_baseline_present():
    X, y = make_regression(n_samples=200, n_features=6, noise=8.0, random_state=0)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)
    candidates = recommend_models("regression", len(X_tr), X_tr.shape[1], use_boost=False)
    result = evaluate(
        candidates, "regression", "rmse",
        [f"f{i}" for i in range(X_tr.shape[1])],
        X_tr, y_tr, X_te, y_te,
    )
    assert result.baseline_name == "dummy_mean"
    assert "rmse" in result.baseline_metrics
