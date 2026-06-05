"""Tests for the new report sections and chart embedding."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris, make_regression

from mudra_ml import Mudra


def _binary_frame() -> pd.DataFrame:
    return load_breast_cancer(as_frame=True).frame


def _multiclass_frame() -> pd.DataFrame:
    return load_iris(as_frame=True).frame


def _regression_frame() -> pd.DataFrame:
    X, y = make_regression(n_samples=300, n_features=8, noise=10.0, random_state=0)
    frame = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    frame["price"] = y
    return frame


def test_html_report_contains_chart_images(tmp_path):
    Mudra().run(_binary_frame(), target="target", report_path=tmp_path / "r")
    html_path = tmp_path / "r.html"
    text = html_path.read_text(encoding="utf-8")
    assert "data:image/png;base64," in text
    assert text.count("data:image/png;base64,") >= 3


def test_markdown_has_trust_summary_and_baseline(tmp_path):
    result = Mudra().run(
        _binary_frame(), target="target", report_path=tmp_path / "r"
    )
    md = result.report_path.read_text(encoding="utf-8")
    assert "Trust summary" in md
    assert "Baseline" in md
    assert "dummy_most_frequent" in md


def test_markdown_has_per_class_table(tmp_path):
    result = Mudra().run(
        _multiclass_frame(), target="target", report_path=tmp_path / "r"
    )
    md = result.report_path.read_text(encoding="utf-8")
    assert "Per-class report" in md


def test_markdown_has_data_quality_and_limitations(tmp_path):
    result = Mudra().run(
        _binary_frame(), target="target", report_path=tmp_path / "r"
    )
    md = result.report_path.read_text(encoding="utf-8")
    assert "Data quality" in md
    assert "Limitations and next steps" in md


def test_markdown_has_cv_mean_and_std(tmp_path):
    result = Mudra().run(
        _binary_frame(), target="target", report_path=tmp_path / "r"
    )
    md = result.report_path.read_text(encoding="utf-8")
    assert "Cross-validation score" in md
    assert "+/-" in md


def test_markdown_has_permutation_importance_note(tmp_path):
    result = Mudra().run(
        _binary_frame(), target="target", report_path=tmp_path / "r"
    )
    md = result.report_path.read_text(encoding="utf-8")
    assert "permutation" in md.lower()
    assert "biased toward high-cardinality" in md


def test_regression_report_has_residual_summary(tmp_path):
    result = Mudra().run(
        _regression_frame(), target="price", report_path=tmp_path / "r"
    )
    md = result.report_path.read_text(encoding="utf-8")
    assert "Regression diagnostics" in md
    html = (tmp_path / "r.html").read_text(encoding="utf-8")
    assert "Residual" in html
    assert "Predicted vs actual" in html


def test_html_report_has_leakage_warning(tmp_path):
    rng = np.random.default_rng(0)
    target = rng.integers(0, 2, 300)
    frame = pd.DataFrame(
        {
            "x": rng.normal(0, 1, 300),
            "z": rng.normal(0, 1, 300),
            "leak": target.copy(),
            "label": target,
        }
    )
    Mudra().run(frame, target="label", report_path=tmp_path / "r")
    html = (tmp_path / "r.html").read_text(encoding="utf-8")
    assert "leakage" in html.lower()


def test_small_sample_warning_appears_in_report(tmp_path):
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "x1": rng.normal(0, 1, 40),
            "x2": rng.normal(0, 1, 40),
            "label": rng.integers(0, 2, 40),
        }
    )
    result = Mudra().run(frame, target="label", report_path=tmp_path / "r")
    md = result.report_path.read_text(encoding="utf-8")
    assert "indicative only" in md


def test_baseline_difference_in_report(tmp_path):
    result = Mudra().run(
        _binary_frame(), target="target", report_path=tmp_path / "r"
    )
    md = result.report_path.read_text(encoding="utf-8")
    assert "Difference" in md or "difference" in md.lower()


def test_correlation_heatmap_in_html(tmp_path):
    Mudra().run(_multiclass_frame(), target="target", report_path=tmp_path / "r")
    html = (tmp_path / "r.html").read_text(encoding="utf-8")
    assert "Feature correlation" in html


def test_clustering_report_still_renders(tmp_path):
    iris = load_iris(as_frame=True).frame.drop(columns=["target"])
    result = Mudra().run(iris, task="clustering", report_path=tmp_path / "r")
    md = result.report_path.read_text(encoding="utf-8")
    assert "MudraML run report" in md
    assert "Data quality" in md
