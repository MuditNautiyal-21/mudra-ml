"""Human-readable run report rendered from the decision log and metrics.

The markdown report stays text and tables. The HTML report carries the same
content plus the diagnostic charts produced by the plots module. Charts are
embedded as base64 PNGs so the file is self-contained.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from jinja2 import Template

from . import plots
from .decisions import DecisionLog
from .profile import BOOLEAN, NUMERIC, DataProfile
from .quality import QualityReport

_STAGE_TITLES = {
    "profile": "Data profiling",
    "goal": "Goal definition",
    "quality": "Data quality",
    "preprocess": "Cleaning and preprocessing",
    "recommend": "Algorithm recommendation",
    "evaluate": "Training and evaluation",
}


@dataclass
class ReportContext:
    """Everything the report needs, gathered from a run."""

    dataset_name: str
    n_rows: int
    n_columns: int
    goal: dict[str, Any]
    operator_set_fields: list[str]
    decisions: list[dict[str, Any]]
    candidates: list[dict[str, Any]]
    best_name: str
    metric: str
    test_metrics: dict[str, Any]
    train_metrics: dict[str, Any] = field(default_factory=dict)
    feature_importance: dict[str, float] = field(default_factory=dict)
    permutation_importance: dict[str, float] = field(default_factory=dict)
    permutation_importance_std: dict[str, float] = field(default_factory=dict)
    baseline_metrics: dict[str, float] = field(default_factory=dict)
    baseline_name: str = ""
    per_class_report: dict[str, dict[str, float]] = field(default_factory=dict)
    cv_mean: float = 0.0
    cv_std: float = 0.0
    small_sample_warning: bool = False
    test_set_size: int = 0
    train_set_size: int = 0
    task: str = ""
    quality: dict[str, Any] = field(default_factory=dict)
    charts: dict[str, str] = field(default_factory=dict)
    overfitting_gap: dict[str, float] = field(default_factory=dict)
    regression_diag: dict[str, Any] = field(default_factory=dict)


def _format_value(value: Any, fmt: str = "%.4f") -> str:
    """Format a value for the report. Numbers go through `fmt`, others string."""
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return fmt % float(value)
    return str(value)


_MARKDOWN_TEMPLATE = """# MudraML run report

Dataset: {{ ctx.dataset_name }}
Rows: {{ ctx.n_rows }}  Columns: {{ ctx.n_columns }}

## Goal

| Field | Value | Source |
| --- | --- | --- |
| Task | {{ ctx.goal.task }} | {{ "operator" if "task" in ctx.operator_set_fields else "inferred" }} |
| Target | {{ ctx.goal.target if ctx.goal.target is not none else "none (unsupervised)" }} | {{ "operator" if "target" in ctx.operator_set_fields else "inferred" }} |
| Metric | {{ ctx.goal.metric }} | {{ "operator" if "metric" in ctx.operator_set_fields else "inferred" }} |
{% if ctx.goal.constraints %}| Constraints | {{ ctx.goal.constraints }} | operator |
{% endif %}

## Trust summary

{% if ctx.small_sample_warning %}Held-out test size: {{ ctx.test_set_size }} rows. Metrics are indicative only. Treat the numbers as a rough guide, not a verdict.
{% else %}Held-out test size: {{ ctx.test_set_size }} rows. Training size: {{ ctx.train_set_size }} rows.
{% endif %}
{% if ctx.baseline_metrics %}
Baseline: {{ ctx.baseline_name }} (no learning, predicts the most frequent class or the mean).

| Metric | Best model | Baseline | Difference |
| --- | --- | --- | --- |
{% for key, value in ctx.test_metrics.items() %}{% if key != "confusion_matrix" %}| {{ key }} | {{ fmt(value) }} | {{ fmt(ctx.baseline_metrics.get(key)) }} | {{ fmt(diff(value, ctx.baseline_metrics.get(key))) }} |
{% endif %}{% endfor %}
{% endif %}
{% if ctx.overfitting_gap %}
Train vs test gap on selected metrics (positive means train is better than test).

| Metric | Train | Test | Gap |
| --- | --- | --- | --- |
{% for key, value in ctx.overfitting_gap.items() %}| {{ key }} | {{ fmt(ctx.train_metrics.get(key)) }} | {{ fmt(ctx.test_metrics.get(key)) }} | {{ fmt(value) }} |
{% endfor %}
{% endif %}
## Result

Selected model: {{ ctx.best_name }}
Cross-validation score: {{ fmt(ctx.cv_mean) }} +/- {{ fmt(ctx.cv_std) }}

Held-out metrics:

{% for name, value in ctx.test_metrics.items() %}{% if name != "confusion_matrix" %}- {{ name }}: {{ fmt(value) }}
{% endif %}{% endfor %}
{% if ctx.per_class_report %}
### Per-class report

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
{% for label, values in ctx.per_class_report.items() %}{% if label not in ("accuracy", "macro avg", "weighted avg") %}| {{ label }} | {{ fmt(values.precision) }} | {{ fmt(values.recall) }} | {{ fmt(values.f1) }} | {{ "%d"|format(values.support|int) }} |
{% endif %}{% endfor %}{% endif %}
{% if ctx.candidates %}
## Candidates compared

Ranked by cross-validation {{ ctx.metric }} (mean across folds). The held-out test set was scored only for the selected model.

| Model | CV mean | CV std | Selected |
| --- | --- | --- | --- |
{% for cand in ctx.candidates %}| {{ cand.name }} | {{ fmt(cand.cv_mean) }} | {{ fmt(cand.cv_std) }} | {{ "yes" if cand.name == ctx.best_name else "no" }} |
{% endfor %}{% endif %}
{% if ctx.permutation_importance %}
## Feature importance (permutation, mean across {{ perm_repeats }} repeats)

Impurity importance is biased toward high-cardinality features. The permutation view is more reliable because it scores each feature by how much shuffling it hurts the model.

{% for name, score in ctx.permutation_importance.items() %}- {{ name }}: {{ fmt(score) }} (+/- {{ fmt(ctx.permutation_importance_std.get(name)) }})
{% endfor %}{% elif ctx.feature_importance %}
## Feature importance

Note: this is impurity or coefficient importance, which can be biased toward high-cardinality features.

{% for name, score in ctx.feature_importance.items() %}- {{ name }}: {{ fmt(score) }}
{% endfor %}{% endif %}
{% if ctx.task == "regression" and regression_diag %}
## Regression diagnostics

Residual summary (actual minus predicted):

- Mean: {{ fmt(regression_diag.residual_mean) }}
- Std: {{ fmt(regression_diag.residual_std) }}
- Mean absolute error: {{ fmt(regression_diag.residual_abs_mean) }}
- Max absolute residual: {{ fmt(regression_diag.residual_max) }}
{% endif %}

## Data quality

{% if ctx.quality.warnings %}{% for w in ctx.quality.warnings %}- [{{ w.severity }}] {{ w.message }} _(rule: {{ w.code }})_
{% endfor %}{% else %}No quality issues raised.
{% endif %}
{% if ctx.quality.class_balance %}
### Class balance

| Class | Count |
| --- | --- |
{% for label, count in ctx.quality.class_balance.items() %}| {{ label }} | {{ count }} |
{% endfor %}{% endif %}
{% if ctx.quality.missingness %}
### Missingness

| Column | Missing | Fraction |
| --- | --- | --- |
{% for row in ctx.quality.missingness %}| {{ row.column }} | {{ row.missing_count }} | {{ "%.2f%%"|format(row.missing_fraction * 100) }} |
{% endfor %}{% endif %}
{% if ctx.quality.outliers %}
### Outlier counts (IQR rule)

| Column | Outliers |
| --- | --- |
{% for row in ctx.quality.outliers %}| {{ row.column }} | {{ row.outlier_count }} |
{% endfor %}{% endif %}
{% if ctx.quality.leakage_suspects %}
### Possible leakage

| Feature | Reason | Score |
| --- | --- | --- |
{% for row in ctx.quality.leakage_suspects %}| {{ row.feature }} | {{ row.reason }} | {{ fmt(row.score) }} |
{% endfor %}{% endif %}
{% if ctx.quality.next_steps %}
## Limitations and next steps

{% for step in ctx.quality.next_steps %}- {{ step }}
{% endfor %}{% endif %}

## Decision log

Every automated choice and the rule that produced it.

{% for stage in stages %}### {{ stage_titles.get(stage, stage) }}

{% for d in ctx.decisions if d.stage == stage %}- {{ d.decision }} _(rule: {{ d.rule }})_
{% endfor %}
{% endfor %}"""


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MudraML run report</title>
<style>
body { font-family: system-ui, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; line-height: 1.5; }
h1 { border-bottom: 2px solid #333; padding-bottom: 0.3rem; }
h2 { margin-top: 2rem; color: #222; }
h3 { margin-top: 1.4rem; color: #444; }
table { border-collapse: collapse; width: 100%; margin: 1rem 0; }
th, td { border: 1px solid #ccc; padding: 0.4rem 0.6rem; text-align: left; }
th { background: #f3f3f3; }
code { background: #f3f3f3; padding: 0.1rem 0.3rem; border-radius: 3px; }
.rule { color: #777; font-size: 0.85em; }
.metric { font-weight: 600; }
.warn { border-left: 4px solid #ffa500; padding: 0.4rem 0.8rem; background: #fff7e6; margin: 0.4rem 0; }
.critical { border-left: 4px solid #c0392b; padding: 0.4rem 0.8rem; background: #fdecea; margin: 0.4rem 0; }
.info { border-left: 4px solid #4c78a8; padding: 0.4rem 0.8rem; background: #eef4fb; margin: 0.4rem 0; }
.note { color: #555; font-size: 0.9em; }
ul { padding-left: 1.2rem; }
img.chart { max-width: 100%; height: auto; display: block; margin: 1rem 0; border: 1px solid #eee; }
.chart-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 1rem; }
</style>
</head>
<body>
<h1>MudraML run report</h1>
<p>Dataset: <strong>{{ ctx.dataset_name }}</strong><br>
Rows: {{ ctx.n_rows }} &nbsp; Columns: {{ ctx.n_columns }}</p>

<h2>Goal</h2>
<table>
<tr><th>Field</th><th>Value</th><th>Source</th></tr>
<tr><td>Task</td><td>{{ ctx.goal.task }}</td><td>{{ "operator" if "task" in ctx.operator_set_fields else "inferred" }}</td></tr>
<tr><td>Target</td><td>{{ ctx.goal.target if ctx.goal.target is not none else "none (unsupervised)" }}</td><td>{{ "operator" if "target" in ctx.operator_set_fields else "inferred" }}</td></tr>
<tr><td>Metric</td><td>{{ ctx.goal.metric }}</td><td>{{ "operator" if "metric" in ctx.operator_set_fields else "inferred" }}</td></tr>
{% if ctx.goal.constraints %}<tr><td>Constraints</td><td>{{ ctx.goal.constraints }}</td><td>operator</td></tr>{% endif %}
</table>

<h2>Trust summary</h2>
{% if ctx.small_sample_warning %}<div class="critical">Held-out test size: {{ ctx.test_set_size }} rows. Metrics are indicative only.</div>
{% else %}<p>Held-out test size: {{ ctx.test_set_size }} rows. Training size: {{ ctx.train_set_size }} rows.</p>
{% endif %}
{% if ctx.baseline_metrics %}
<p>Baseline: <code>{{ ctx.baseline_name }}</code> (no learning, predicts the most frequent class or the mean).</p>
<table>
<tr><th>Metric</th><th>Best model</th><th>Baseline</th><th>Difference</th></tr>
{% for key, value in ctx.test_metrics.items() %}{% if key != "confusion_matrix" %}<tr><td>{{ key }}</td><td>{{ fmt(value) }}</td><td>{{ fmt(ctx.baseline_metrics.get(key)) }}</td><td>{{ fmt(diff(value, ctx.baseline_metrics.get(key))) }}</td></tr>{% endif %}{% endfor %}
</table>
{% endif %}
{% if ctx.overfitting_gap %}
<h3>Train vs test gap</h3>
<table>
<tr><th>Metric</th><th>Train</th><th>Test</th><th>Gap</th></tr>
{% for key, value in ctx.overfitting_gap.items() %}<tr><td>{{ key }}</td><td>{{ fmt(ctx.train_metrics.get(key)) }}</td><td>{{ fmt(ctx.test_metrics.get(key)) }}</td><td>{{ fmt(value) }}</td></tr>{% endfor %}
</table>
{% endif %}

<h2>Result</h2>
<p>Selected model: <span class="metric">{{ ctx.best_name }}</span><br>
Cross-validation score: {{ fmt(ctx.cv_mean) }} &plusmn; {{ fmt(ctx.cv_std) }}</p>
<ul>
{% for name, value in ctx.test_metrics.items() %}{% if name != "confusion_matrix" %}<li>{{ name }}: {{ fmt(value) }}</li>{% endif %}{% endfor %}
</ul>
{% if ctx.charts.target_distribution %}<img class="chart" alt="Target distribution" src="data:image/png;base64,{{ ctx.charts.target_distribution }}">{% endif %}

{% if ctx.per_class_report %}
<h3>Per-class report</h3>
<table>
<tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr>
{% for label, values in ctx.per_class_report.items() %}{% if label not in ("accuracy", "macro avg", "weighted avg") %}<tr><td>{{ label }}</td><td>{{ fmt(values.precision) }}</td><td>{{ fmt(values.recall) }}</td><td>{{ fmt(values.f1) }}</td><td>{{ values.support|int }}</td></tr>{% endif %}{% endfor %}
</table>
{% endif %}

{% if ctx.charts.confusion_matrix or ctx.charts.roc or ctx.charts.pr %}
<h3>Diagnostic charts</h3>
<div class="chart-grid">
{% if ctx.charts.confusion_matrix %}<img class="chart" alt="Confusion matrix" src="data:image/png;base64,{{ ctx.charts.confusion_matrix }}">{% endif %}
{% if ctx.charts.roc %}<img class="chart" alt="ROC curve" src="data:image/png;base64,{{ ctx.charts.roc }}">{% endif %}
{% if ctx.charts.pr %}<img class="chart" alt="Precision-recall curve" src="data:image/png;base64,{{ ctx.charts.pr }}">{% endif %}
</div>
{% endif %}

{% if ctx.charts.residual or ctx.charts.predicted_vs_actual %}
<h3>Regression diagnostics</h3>
{% if regression_diag %}<ul>
<li>Residual mean: {{ fmt(regression_diag.residual_mean) }}</li>
<li>Residual std: {{ fmt(regression_diag.residual_std) }}</li>
<li>Mean absolute error: {{ fmt(regression_diag.residual_abs_mean) }}</li>
<li>Max absolute residual: {{ fmt(regression_diag.residual_max) }}</li>
</ul>{% endif %}
<div class="chart-grid">
{% if ctx.charts.residual %}<img class="chart" alt="Residual plot" src="data:image/png;base64,{{ ctx.charts.residual }}">{% endif %}
{% if ctx.charts.predicted_vs_actual %}<img class="chart" alt="Predicted vs actual" src="data:image/png;base64,{{ ctx.charts.predicted_vs_actual }}">{% endif %}
</div>
{% endif %}

{% if ctx.candidates %}<h2>Candidates compared</h2>
<p class="note">Ranked by cross-validation {{ ctx.metric }} (mean across folds). The held-out test set was scored only for the selected model.</p>
<table>
<tr><th>Model</th><th>CV mean</th><th>CV std</th><th>Selected</th></tr>
{% for cand in ctx.candidates %}<tr><td>{{ cand.name }}</td><td>{{ fmt(cand.cv_mean) }}</td><td>{{ fmt(cand.cv_std) }}</td><td>{{ "yes" if cand.name == ctx.best_name else "no" }}</td></tr>{% endfor %}
</table>{% endif %}

{% if ctx.permutation_importance %}<h2>Feature importance (permutation, mean across {{ perm_repeats }} repeats)</h2>
<p class="note">Impurity importance is biased toward high-cardinality features. The permutation view is more reliable.</p>
<ul>
{% for name, score in ctx.permutation_importance.items() %}<li>{{ name }}: {{ fmt(score) }} (&plusmn; {{ fmt(ctx.permutation_importance_std.get(name)) }})</li>{% endfor %}
</ul>
{% if ctx.charts.feature_importance %}<img class="chart" alt="Feature importance" src="data:image/png;base64,{{ ctx.charts.feature_importance }}">{% endif %}
{% elif ctx.feature_importance %}<h2>Feature importance</h2>
<p class="note">Impurity or coefficient importance. Can be biased toward high-cardinality features.</p>
<ul>
{% for name, score in ctx.feature_importance.items() %}<li>{{ name }}: {{ fmt(score) }}</li>{% endfor %}
</ul>
{% if ctx.charts.feature_importance %}<img class="chart" alt="Feature importance" src="data:image/png;base64,{{ ctx.charts.feature_importance }}">{% endif %}
{% endif %}

<h2>Data quality</h2>
{% if ctx.quality.warnings %}{% for w in ctx.quality.warnings %}<div class="{{ w.severity }}"><strong>{{ w.severity|upper }}:</strong> {{ w.message }} <span class="rule">(rule: {{ w.code }})</span></div>{% endfor %}{% else %}<p>No quality issues raised.</p>{% endif %}

{% if ctx.quality.class_balance %}
<h3>Class balance</h3>
<table>
<tr><th>Class</th><th>Count</th></tr>
{% for label, count in ctx.quality.class_balance.items() %}<tr><td>{{ label }}</td><td>{{ count }}</td></tr>{% endfor %}
</table>{% endif %}

{% if ctx.quality.missingness %}
<h3>Missingness</h3>
<table>
<tr><th>Column</th><th>Missing</th><th>Fraction</th></tr>
{% for row in ctx.quality.missingness %}<tr><td>{{ row.column }}</td><td>{{ row.missing_count }}</td><td>{{ "%.2f%%"|format(row.missing_fraction * 100) }}</td></tr>{% endfor %}
</table>{% endif %}

{% if ctx.quality.outliers %}
<h3>Outlier counts (IQR rule)</h3>
<table>
<tr><th>Column</th><th>Outliers</th></tr>
{% for row in ctx.quality.outliers %}<tr><td>{{ row.column }}</td><td>{{ row.outlier_count }}</td></tr>{% endfor %}
</table>{% endif %}

{% if ctx.charts.correlation %}<h3>Feature correlation</h3>
<img class="chart" alt="Feature correlation" src="data:image/png;base64,{{ ctx.charts.correlation }}">{% endif %}

{% if ctx.quality.leakage_suspects %}
<h3>Possible leakage</h3>
<table>
<tr><th>Feature</th><th>Reason</th><th>Score</th></tr>
{% for row in ctx.quality.leakage_suspects %}<tr><td>{{ row.feature }}</td><td>{{ row.reason }}</td><td>{{ fmt(row.score) }}</td></tr>{% endfor %}
</table>{% endif %}

{% if ctx.quality.next_steps %}
<h2>Limitations and next steps</h2>
<ul>
{% for step in ctx.quality.next_steps %}<li>{{ step }}</li>{% endfor %}
</ul>{% endif %}

<h2>Decision log</h2>
<p>Every automated choice and the rule that produced it.</p>
{% for stage in stages %}<h3>{{ stage_titles.get(stage, stage) }}</h3>
<ul>
{% for d in ctx.decisions if d.stage == stage %}<li>{{ d.decision }} <span class="rule">(rule: {{ d.rule }})</span></li>{% endfor %}
</ul>
{% endfor %}
</body>
</html>"""


def _stages_present(decisions: list[dict[str, Any]]) -> list[str]:
    ordered = [s for s in _STAGE_TITLES if any(d["stage"] == s for d in decisions)]
    extras = sorted({d["stage"] for d in decisions} - set(_STAGE_TITLES))
    return ordered + extras


def _filters() -> dict[str, Any]:
    def diff(a: Any, b: Any) -> Any:
        try:
            return float(a) - float(b)
        except (TypeError, ValueError):
            return None

    return {"fmt": _format_value, "diff": diff}


def render_markdown(ctx: ReportContext) -> str:
    """Render the report as markdown text."""
    stages = _stages_present(ctx.decisions)
    template = Template(_MARKDOWN_TEMPLATE, trim_blocks=True, lstrip_blocks=True)
    return template.render(
        ctx=ctx,
        stages=stages,
        stage_titles=_STAGE_TITLES,
        perm_repeats=10,
        regression_diag=ctx.regression_diag,
        **_filters(),
    )


def render_html(ctx: ReportContext) -> str:
    """Render the report as a standalone HTML document."""
    stages = _stages_present(ctx.decisions)
    template = Template(_HTML_TEMPLATE, trim_blocks=True, lstrip_blocks=True)
    return template.render(
        ctx=ctx,
        stages=stages,
        stage_titles=_STAGE_TITLES,
        perm_repeats=10,
        regression_diag=ctx.regression_diag,
        **_filters(),
    )


def write_report(
    ctx: ReportContext,
    path: str | Path,
    html: bool = True,
) -> Path:
    """Write the report to disk as markdown, plus HTML when requested."""
    path = Path(path)
    md_path = path.with_suffix(".md")
    md_path.write_text(render_markdown(ctx), encoding="utf-8")
    if html:
        html_path = path.with_suffix(".html")
        html_path.write_text(render_html(ctx), encoding="utf-8")
    return md_path


def _overfitting_gap(
    train_metrics: dict[str, float], test_metrics: dict[str, float], task: str
) -> dict[str, float]:
    """Train minus test for headline metrics. Positive means train is better."""
    if task == "classification":
        keys = ["accuracy", "f1", "precision", "recall"]
    elif task == "regression":
        keys = ["rmse", "mae", "r2"]
    else:
        return {}
    gaps: dict[str, float] = {}
    for key in keys:
        if key in train_metrics and key in test_metrics:
            train_value = float(train_metrics[key])
            test_value = float(test_metrics[key])
            if key in ("rmse", "mae"):
                gaps[key] = test_value - train_value
            else:
                gaps[key] = train_value - test_value
    return gaps


def _numeric_columns(frame: pd.DataFrame, profile: DataProfile, limit: int = 20) -> list[str]:
    """Pick numeric/boolean columns from the profile that exist in the frame."""
    cols = [
        col.name
        for col in profile.columns.values()
        if col.inferred_type in (NUMERIC, BOOLEAN) and col.name in frame.columns
    ]
    if len(cols) > limit:
        cols = cols[:limit]
    return cols


def _correlation_matrix(
    frame: pd.DataFrame, columns: list[str]
) -> tuple[list[list[float]], list[str]]:
    if len(columns) < 2:
        return [], []
    try:
        sub = frame[columns].apply(pd.to_numeric, errors="coerce")
        corr = sub.corr().fillna(0.0)
    except (TypeError, ValueError):
        return [], []
    return corr.to_numpy().tolist(), list(corr.columns)


def _build_charts(
    evaluation: dict[str, Any],
    frame: pd.DataFrame | None,
    profile: DataProfile | None,
    task: str,
) -> dict[str, str]:
    """Produce every chart for the HTML report, skipping any that fail."""
    spec: dict[str, dict[str, Any]] = {}

    best = next(
        (c for c in evaluation.get("candidates", []) if c.get("name") == evaluation.get("best_name")),
        None,
    )
    test_metrics = best["test_metrics"] if best else {}
    confusion = test_metrics.get("confusion_matrix")
    class_labels = evaluation.get("class_labels") or []

    if task == "classification" and confusion:
        labels_for_chart = class_labels if class_labels else list(range(len(confusion)))
        spec["confusion_matrix"] = {
            "kind": "confusion_matrix",
            "kwargs": {"matrix": confusion, "labels": labels_for_chart},
        }

    roc = evaluation.get("roc_curve") or {}
    if roc.get("fpr"):
        spec["roc"] = {
            "kind": "roc",
            "kwargs": {"fpr": roc["fpr"], "tpr": roc["tpr"], "auc": roc.get("auc", 0.0)},
        }

    pr = evaluation.get("pr_curve") or {}
    if pr.get("recall"):
        spec["pr"] = {
            "kind": "pr",
            "kwargs": {
                "recall": pr["recall"],
                "precision": pr["precision"],
                "average_precision": pr.get("average_precision", 0.0),
            },
        }

    perm = evaluation.get("permutation_importance") or {}
    perm_std = evaluation.get("permutation_importance_std") or {}
    if perm:
        spec["feature_importance"] = {
            "kind": "feature_importance",
            "kwargs": {
                "importances": perm,
                "stds": perm_std,
                "title": "Permutation importance",
            },
        }
    elif evaluation.get("feature_importance"):
        spec["feature_importance"] = {
            "kind": "feature_importance",
            "kwargs": {
                "importances": evaluation["feature_importance"],
                "title": "Impurity importance",
            },
        }

    target_values = evaluation.get("target_values") or []
    if target_values and task in ("classification", "regression"):
        spec["target_distribution"] = {
            "kind": "target_distribution",
            "kwargs": {"values": target_values, "task": task},
        }

    if frame is not None and profile is not None:
        numeric_cols = _numeric_columns(frame, profile)
        matrix, columns = _correlation_matrix(frame, numeric_cols)
        if matrix:
            spec["correlation"] = {
                "kind": "correlation",
                "kwargs": {"matrix": matrix, "columns": columns},
            }

    diag = evaluation.get("regression_diagnostics") or {}
    if task == "regression" and diag.get("y_true"):
        spec["residual"] = {
            "kind": "residual",
            "kwargs": {"y_true": diag["y_true"], "y_pred": diag["y_pred"]},
        }
        spec["predicted_vs_actual"] = {
            "kind": "predicted_vs_actual",
            "kwargs": {"y_true": diag["y_true"], "y_pred": diag["y_pred"]},
        }

    return plots.render_all(spec)


def build_context(
    dataset_name: str,
    n_rows: int,
    n_columns: int,
    goal: dict[str, Any],
    operator_set_fields: list[str],
    log: DecisionLog,
    evaluation: dict[str, Any],
    profile: DataProfile | None = None,
    quality: QualityReport | None = None,
    frame: pd.DataFrame | None = None,
) -> ReportContext:
    """Assemble a ReportContext from run pieces."""
    best: dict[str, Any] = next(
        (c for c in evaluation["candidates"] if c["name"] == evaluation["best_name"]),
        {"test_metrics": {}, "train_metrics": {}, "cv_mean": 0.0, "cv_std": 0.0},
    )
    test_metrics = best.get("test_metrics", {})
    train_metrics = best.get("train_metrics", {})
    task = evaluation.get("task", goal.get("task", ""))

    overfitting_gap = _overfitting_gap(train_metrics, test_metrics, task)
    charts = _build_charts(evaluation, frame, profile, task)
    quality_dict = quality.as_dict() if quality is not None else {
        "warnings": [],
        "class_balance": {},
        "missingness": [],
        "cardinality": [],
        "outliers": [],
        "leakage_suspects": [],
        "next_steps": [],
    }

    return ReportContext(
        dataset_name=dataset_name,
        n_rows=n_rows,
        n_columns=n_columns,
        goal=goal,
        operator_set_fields=operator_set_fields,
        decisions=log.as_list(),
        candidates=evaluation["candidates"],
        best_name=evaluation["best_name"],
        metric=evaluation["metric"],
        test_metrics=test_metrics,
        train_metrics=train_metrics,
        feature_importance=evaluation.get("feature_importance", {}),
        permutation_importance=evaluation.get("permutation_importance", {}),
        permutation_importance_std=evaluation.get("permutation_importance_std", {}),
        baseline_metrics=evaluation.get("baseline_metrics", {}),
        baseline_name=evaluation.get("baseline_name", ""),
        per_class_report=evaluation.get("per_class_report", {}),
        cv_mean=float(best.get("cv_mean", 0.0)),
        cv_std=float(best.get("cv_std", 0.0)),
        small_sample_warning=bool(evaluation.get("small_sample_warning", False)),
        test_set_size=int(evaluation.get("test_set_size", 0)),
        train_set_size=int(evaluation.get("train_set_size", 0)),
        task=task,
        quality=quality_dict,
        charts=charts,
        overfitting_gap=overfitting_gap,
        regression_diag=evaluation.get("regression_diagnostics", {}) or {},
    )
