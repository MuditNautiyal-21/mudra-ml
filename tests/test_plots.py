"""Tests for the chart-rendering helpers in plots.py."""

from __future__ import annotations

import base64

import numpy as np

from mudra_ml import plots


def _decodable(value: str) -> bool:
    try:
        base64.b64decode(value)
        return True
    except (TypeError, ValueError):
        return False


def test_confusion_matrix_chart_returns_base64():
    matrix = [[10, 2], [1, 7]]
    encoded = plots.confusion_matrix_chart(matrix, [0, 1])
    assert encoded is not None
    assert _decodable(encoded)


def test_roc_curve_chart_returns_base64():
    fpr = [0.0, 0.1, 0.5, 1.0]
    tpr = [0.0, 0.6, 0.9, 1.0]
    encoded = plots.roc_curve_chart(fpr, tpr, 0.91)
    assert encoded is not None


def test_pr_curve_chart_returns_base64():
    encoded = plots.precision_recall_chart(
        recall=[1.0, 0.8, 0.5, 0.0],
        precision=[0.3, 0.6, 0.8, 1.0],
        average_precision=0.7,
    )
    assert encoded is not None


def test_feature_importance_chart_returns_base64():
    importances = {"a": 0.4, "b": 0.3, "c": 0.2, "d": 0.1}
    stds = {"a": 0.02, "b": 0.03, "c": 0.01, "d": 0.01}
    encoded = plots.feature_importance_chart(importances, stds)
    assert encoded is not None


def test_target_distribution_classification():
    encoded = plots.target_distribution_chart(["a", "b", "a", "c", "b"], "classification")
    assert encoded is not None


def test_target_distribution_regression():
    rng = np.random.default_rng(0)
    values = rng.normal(0, 1, 200).tolist()
    encoded = plots.target_distribution_chart(values, "regression")
    assert encoded is not None


def test_correlation_heatmap_returns_base64():
    matrix = [[1.0, 0.3, -0.2], [0.3, 1.0, 0.1], [-0.2, 0.1, 1.0]]
    encoded = plots.correlation_heatmap(matrix, ["a", "b", "c"])
    assert encoded is not None


def test_residual_chart_returns_base64():
    encoded = plots.residual_chart([1.0, 2.0, 3.0, 4.0], [1.1, 2.1, 2.9, 3.8])
    assert encoded is not None


def test_predicted_vs_actual_chart_returns_base64():
    encoded = plots.predicted_vs_actual_chart([1.0, 2.0, 3.0, 4.0], [1.1, 2.1, 2.9, 3.8])
    assert encoded is not None


def test_graceful_degradation_on_empty_inputs():
    assert plots.confusion_matrix_chart([], []) is None
    assert plots.roc_curve_chart([], [], 0.0) is None
    assert plots.precision_recall_chart([], [], 0.0) is None
    assert plots.feature_importance_chart({}) is None
    assert plots.correlation_heatmap([], []) is None
    assert plots.residual_chart([], []) is None
    assert plots.predicted_vs_actual_chart([], []) is None


def test_graceful_degradation_on_bad_inputs():
    """Charts must return None rather than raise on malformed input."""
    bad_matrix = [[1, 2], [3]]
    assert plots.correlation_heatmap(bad_matrix, ["a", "b"]) is None


def test_render_all_skips_unknown_kinds():
    spec = {"unknown": {"kind": "unknown", "kwargs": {}}}
    out = plots.render_all(spec)
    assert out == {}


def test_render_all_includes_good_charts():
    spec = {
        "confusion_matrix": {
            "kind": "confusion_matrix",
            "kwargs": {"matrix": [[5, 1], [0, 4]], "labels": [0, 1]},
        },
        "roc": {"kind": "roc", "kwargs": {"fpr": [0.0, 1.0], "tpr": [0.0, 1.0], "auc": 0.5}},
    }
    out = plots.render_all(spec)
    assert "confusion_matrix" in out
    assert "roc" in out
