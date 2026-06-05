from __future__ import annotations

import numpy as np
import pandas as pd

from mudra_ml import Mudra
from mudra_ml.report import ReportContext, render_html, render_markdown


def test_end_to_end_classification(tmp_path, classification_frame):
    path = tmp_path / "clf.csv"
    classification_frame.to_csv(path, index=False)
    result = Mudra().run(path, target="target", report_path=tmp_path / "report")
    assert result.task == "classification"
    assert result.report_path.exists()
    assert (tmp_path / "report.html").exists()
    preds = result.predict(classification_frame.drop(columns=["target"]).head(10))
    assert len(preds) == 10


def test_end_to_end_regression(tmp_path, regression_frame):
    path = tmp_path / "reg.csv"
    regression_frame.to_csv(path, index=False)
    result = Mudra().run(path, target="price", report_path=tmp_path / "report")
    assert result.task == "regression"
    assert "rmse" in result.evaluation["candidates"][0]["test_metrics"]


def test_end_to_end_clustering(tmp_path, clustering_frame):
    path = tmp_path / "clu.csv"
    clustering_frame.to_csv(path, index=False)
    result = Mudra().run(path, task="clustering", report_path=tmp_path / "report")
    assert result.task == "clustering"
    assert result.best_model is not None


def test_run_accepts_dataframe(regression_frame, tmp_path):
    result = Mudra().run(regression_frame, target="price", report_path=tmp_path / "r")
    assert result.task == "regression"


def test_save_load_predict_roundtrip(tmp_path, classification_frame):
    result = Mudra().run(classification_frame, target="target", report_path=tmp_path / "r")
    artifact = result.save(tmp_path / "artifact")
    assert artifact.exists()
    loaded = Mudra.load(artifact)
    original = result.predict(classification_frame.drop(columns=["target"]).head(20))
    reloaded = loaded.predict(classification_frame.drop(columns=["target"]).head(20))
    assert np.array_equal(original, reloaded)


def test_operator_constraints_in_report(tmp_path, classification_frame):
    result = Mudra().run(
        classification_frame,
        target="target",
        task="classification",
        constraints={"interpretable": True},
        report_path=tmp_path / "r",
    )
    text = result.report_path.read_text(encoding="utf-8")
    assert "operator" in text
    names = [c["name"] for c in result.evaluation["candidates"]]
    assert all(n in ("logistic_regression", "decision_tree") for n in names)


def test_report_marks_inferred_vs_operator(tmp_path):
    frame = pd.DataFrame({"f1": range(100), "f2": range(100, 200), "churn": [0, 1] * 50})
    result = Mudra().run(frame, report_path=tmp_path / "r")
    text = result.report_path.read_text(encoding="utf-8")
    assert "inferred" in text


def test_determinism_end_to_end(tmp_path, classification_frame):
    a = Mudra().run(classification_frame, target="target", report_path=tmp_path / "a")
    b = Mudra().run(classification_frame, target="target", report_path=tmp_path / "b")
    assert a.evaluation["best_name"] == b.evaluation["best_name"]


def _sample_context() -> ReportContext:
    return ReportContext(
        dataset_name="demo.csv",
        n_rows=100,
        n_columns=5,
        goal={"task": "classification", "target": "y", "metric": "f1", "constraints": {}},
        operator_set_fields=["target"],
        decisions=[
            {"stage": "profile", "decision": "typed col", "rule": "type-inference", "detail": {}},
            {"stage": "evaluate", "decision": "picked model", "rule": "best-model-selection", "detail": {}},
        ],
        candidates=[{"name": "m", "cv_score": 0.9, "test_metrics": {"f1": 0.91}}],
        best_name="m",
        metric="f1",
        test_metrics={"f1": 0.91, "accuracy": 0.9},
        feature_importance={"a": 0.5},
    )


def test_render_markdown_contains_sections():
    md = render_markdown(_sample_context())
    assert "# MudraML run report" in md
    assert "Decision log" in md
    assert "best-model-selection" in md


def test_render_html_is_document():
    html = render_html(_sample_context())
    assert html.startswith("<!DOCTYPE html>")
    assert "Decision log" in html
