"""The report reads in pipeline order, top to bottom.

The sections follow the sequence of what actually happened: run summary,
data profile, data quality, preprocessing, split, model shortlist, tuning
and selection, evaluation, limitations, and the full decision log last.
Each HTML chart stays with its section.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mudra_ml import Mudra

MD_HEADINGS = [
    "## Run summary",
    "## Data profile",
    "## Data quality",
    "## Preprocessing",
    "## Split",
    "## Model shortlist",
    "## Tuning and selection",
    "## Evaluation",
    "## Limitations and next steps",
    "## Decision log",
]

HTML_HEADINGS = [
    "<h2>Run summary</h2>",
    "<h2>Data profile</h2>",
    "<h2>Data quality</h2>",
    "<h2>Preprocessing</h2>",
    "<h2>Split</h2>",
    "<h2>Model shortlist</h2>",
    "<h2>Tuning and selection</h2>",
    "<h2>Evaluation</h2>",
    "<h2>Limitations and next steps</h2>",
    "<h2>Decision log</h2>",
]


def _classification_frame() -> pd.DataFrame:
    rng = np.random.default_rng(4)
    n = 150
    frame = pd.DataFrame(
        {
            "age": rng.uniform(18, 80, n),
            "income": rng.uniform(15000, 90000, n),
            "region": rng.choice(["north", "south", "east"], size=n),
        }
    )
    frame["target"] = ((frame["income"] / 1000 + frame["age"]) > 90).astype(int)
    return frame


def _regression_frame() -> pd.DataFrame:
    rng = np.random.default_rng(5)
    n = 150
    frame = pd.DataFrame(
        {
            "size": rng.uniform(30, 200, n),
            "rooms": rng.integers(1, 6, n).astype(float),
            "area": rng.choice(["city", "suburb"], size=n),
        }
    )
    frame["target"] = 900.0 * frame["size"] + rng.normal(scale=2000.0, size=n)
    return frame


def _positions(text: str, markers: list[str]) -> list[int]:
    missing = [m for m in markers if m not in text]
    assert not missing, f"missing sections: {missing}"
    return [text.index(m) for m in markers]


@pytest.fixture(scope="module")
def classification_reports(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("order_clf")
    Mudra(random_state=42).run(
        _classification_frame(), target="target", report_path=tmp / "r", use_boost=False
    )
    return (tmp / "r.md").read_text(encoding="utf-8"), (tmp / "r.html").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def regression_reports(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("order_reg")
    Mudra(random_state=42).run(
        _regression_frame(), target="target", report_path=tmp / "r", use_boost=False
    )
    return (tmp / "r.md").read_text(encoding="utf-8"), (tmp / "r.html").read_text(encoding="utf-8")


def test_classification_markdown_sections_in_pipeline_order(classification_reports):
    md, _ = classification_reports
    positions = _positions(md, MD_HEADINGS)
    assert positions == sorted(positions)


def test_classification_html_sections_in_pipeline_order(classification_reports):
    _, html = classification_reports
    positions = _positions(html, HTML_HEADINGS)
    assert positions == sorted(positions)


def test_regression_markdown_sections_in_pipeline_order(regression_reports):
    md, _ = regression_reports
    positions = _positions(md, MD_HEADINGS)
    assert positions == sorted(positions)


def test_regression_html_sections_in_pipeline_order(regression_reports):
    _, html = regression_reports
    positions = _positions(html, HTML_HEADINGS)
    assert positions == sorted(positions)


def test_classification_charts_stay_with_their_sections(classification_reports):
    _, html = classification_reports
    profile_at = html.index("<h2>Data profile</h2>")
    quality_at = html.index("<h2>Data quality</h2>")
    evaluation_at = html.index("<h2>Evaluation</h2>")
    limitations_at = html.index("<h2>Limitations and next steps</h2>")

    target_chart = html.index('alt="Target distribution"')
    assert profile_at < target_chart < quality_at

    correlation_chart = html.index('alt="Feature correlation"')
    assert profile_at < correlation_chart < quality_at

    confusion_chart = html.index('alt="Confusion matrix"')
    assert evaluation_at < confusion_chart < limitations_at


def test_regression_charts_stay_with_their_sections(regression_reports):
    _, html = regression_reports
    evaluation_at = html.index("<h2>Evaluation</h2>")
    limitations_at = html.index("<h2>Limitations and next steps</h2>")
    for alt in ('alt="Residual plot"', 'alt="Predicted vs actual"'):
        chart = html.index(alt)
        assert evaluation_at < chart < limitations_at


def test_markdown_keeps_trust_summary_and_decision_log(classification_reports):
    md, _ = classification_reports
    assert "Trust summary" in md
    assert "Decision log" in md
    assert "(rule:" in md
