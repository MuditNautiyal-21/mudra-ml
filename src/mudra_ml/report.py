"""Human-readable run report rendered from the decision log."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jinja2 import Template

from .decisions import DecisionLog

_STAGE_TITLES = {
    "profile": "Data profiling",
    "goal": "Goal definition",
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
    feature_importance: dict[str, float]


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

## Result

Selected model: {{ ctx.best_name }}

Held-out metrics:

{% for name, value in ctx.test_metrics.items() %}{% if name != "confusion_matrix" %}- {{ name }}: {{ "%.4f"|format(value) if value is number else value }}
{% endif %}{% endfor %}
{% if ctx.candidates %}
## Candidates compared

| Model | CV score | Test {{ ctx.metric }} |
| --- | --- | --- |
{% for cand in ctx.candidates %}| {{ cand.name }} | {{ "%.4f"|format(cand.cv_score) }} | {{ "%.4f"|format(cand.test_metrics.get(ctx.metric, 0.0)) if cand.test_metrics.get(ctx.metric) is not none else "n/a" }} |
{% endfor %}{% endif %}
{% if ctx.feature_importance %}
## Feature importance (top)

{% for name, score in ctx.feature_importance.items() %}- {{ name }}: {{ "%.4f"|format(score) }}
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
body { font-family: system-ui, sans-serif; max-width: 860px; margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; line-height: 1.5; }
h1 { border-bottom: 2px solid #333; padding-bottom: 0.3rem; }
h2 { margin-top: 2rem; color: #222; }
h3 { margin-top: 1.4rem; color: #444; }
table { border-collapse: collapse; width: 100%; margin: 1rem 0; }
th, td { border: 1px solid #ccc; padding: 0.4rem 0.6rem; text-align: left; }
th { background: #f3f3f3; }
code { background: #f3f3f3; padding: 0.1rem 0.3rem; border-radius: 3px; }
.rule { color: #777; font-size: 0.85em; }
.metric { font-weight: 600; }
ul { padding-left: 1.2rem; }
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

<h2>Result</h2>
<p>Selected model: <span class="metric">{{ ctx.best_name }}</span></p>
<ul>
{% for name, value in ctx.test_metrics.items() %}{% if name != "confusion_matrix" %}<li>{{ name }}: {{ "%.4f"|format(value) if value is number else value }}</li>{% endif %}{% endfor %}
</ul>

{% if ctx.candidates %}<h2>Candidates compared</h2>
<table>
<tr><th>Model</th><th>CV score</th><th>Test {{ ctx.metric }}</th></tr>
{% for cand in ctx.candidates %}<tr><td>{{ cand.name }}</td><td>{{ "%.4f"|format(cand.cv_score) }}</td><td>{{ "%.4f"|format(cand.test_metrics.get(ctx.metric, 0.0)) if cand.test_metrics.get(ctx.metric) is not none else "n/a" }}</td></tr>{% endfor %}
</table>{% endif %}

{% if ctx.feature_importance %}<h2>Feature importance (top)</h2>
<ul>
{% for name, score in ctx.feature_importance.items() %}<li>{{ name }}: {{ "%.4f"|format(score) }}</li>{% endfor %}
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


def render_markdown(ctx: ReportContext) -> str:
    """Render the report as markdown text."""
    stages = [s for s in _STAGE_TITLES if any(d["stage"] == s for d in ctx.decisions)]
    template = Template(_MARKDOWN_TEMPLATE, trim_blocks=True, lstrip_blocks=True)
    return template.render(ctx=ctx, stages=stages, stage_titles=_STAGE_TITLES)


def render_html(ctx: ReportContext) -> str:
    """Render the report as a standalone HTML document."""
    stages = [s for s in _STAGE_TITLES if any(d["stage"] == s for d in ctx.decisions)]
    template = Template(_HTML_TEMPLATE, trim_blocks=True, lstrip_blocks=True)
    return template.render(ctx=ctx, stages=stages, stage_titles=_STAGE_TITLES)


def write_report(
    ctx: ReportContext,
    path: str | Path,
    html: bool = True,
) -> Path:
    """Write the report to disk as markdown, plus HTML when requested.

    Args:
        ctx: The report context.
        path: Output path. A .md suffix is used for markdown; the HTML file
            sits beside it with a .html suffix.
        html: Whether to also write the HTML version.

    Returns:
        Path to the markdown report.
    """
    path = Path(path)
    md_path = path.with_suffix(".md")
    md_path.write_text(render_markdown(ctx), encoding="utf-8")
    if html:
        html_path = path.with_suffix(".html")
        html_path.write_text(render_html(ctx), encoding="utf-8")
    return md_path


def build_context(
    dataset_name: str,
    n_rows: int,
    n_columns: int,
    goal: dict[str, Any],
    operator_set_fields: list[str],
    log: DecisionLog,
    evaluation: dict[str, Any],
) -> ReportContext:
    """Assemble a ReportContext from run pieces."""
    best: dict[str, Any] = next(
        (c for c in evaluation["candidates"] if c["name"] == evaluation["best_name"]),
        {"test_metrics": {}},
    )
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
        test_metrics=best["test_metrics"],
        feature_importance=evaluation.get("feature_importance", {}),
    )
