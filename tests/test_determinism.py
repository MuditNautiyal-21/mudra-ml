"""Two runs on the same input give the same result and the same report.

One seed is threaded through the split, the search, the estimators, and any
sampling, so the whole run is reproducible. The check covers a dirty mixed
dataset on purpose: the repair and coercion paths must be just as
deterministic as the happy path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from battery_datasets import make_dataset
from mudra_ml import Mudra


def _dirty_mixed_frame() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    n = 180
    frame = pd.DataFrame(
        {
            "amount": [f"${v:,.2f}" for v in rng.uniform(100, 9000, n)],
            "rate": [f"{v:.1f}%" for v in rng.uniform(0, 30, n)],
            "balance": rng.normal(loc=2000, scale=600, size=n),
            "city": rng.choice(["york", "leeds", "bath", "hull"], size=n),
            "active": rng.choice([True, False], size=n),
            "joined": [
                (pd.Timestamp("2023-06-01") + pd.Timedelta(days=int(d))).strftime("%Y-%m-%d")
                for d in rng.integers(0, 300, size=n)
            ],
        }
    )
    score = (
        pd.to_numeric(frame["balance"], errors="coerce") / 1000
        + frame["active"].astype(int)
        + rng.normal(scale=0.4, size=n)
    )
    frame["target"] = (score > score.median()).astype(int)
    frame.loc[frame.index[::13], "amount"] = "--"
    frame.loc[frame.index[::17], "rate"] = np.nan
    frame.loc[frame.index[::19], "city"] = np.nan
    return frame


def _run(frame: pd.DataFrame, path, **kwargs):
    return Mudra(random_state=42).run(
        frame.copy(), report_path=path, html=True, use_boost=False, **kwargs
    )


def test_two_runs_on_dirty_mixed_data_are_identical(tmp_path):
    frame = _dirty_mixed_frame()
    first = _run(frame, tmp_path / "first", target="target")
    second = _run(frame, tmp_path / "second", target="target")

    assert first.evaluation == second.evaluation
    assert first.metrics == second.metrics
    assert first.positive_label == second.positive_label
    assert first.input_schema == second.input_schema

    new_rows = frame.drop(columns=["target"]).iloc[:40]
    assert np.array_equal(first.predict(new_rows), second.predict(new_rows))
    assert np.array_equal(first.predict_proba(new_rows), second.predict_proba(new_rows))

    first_md = (tmp_path / "first.md").read_text(encoding="utf-8")
    second_md = (tmp_path / "second.md").read_text(encoding="utf-8")
    assert first_md == second_md

    first_html = (tmp_path / "first.html").read_text(encoding="utf-8")
    second_html = (tmp_path / "second.html").read_text(encoding="utf-8")
    assert first_html == second_html


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("name", ["random_forest", "extra_trees"])
def test_repeated_forest_predictions_are_bit_identical(task, name):
    from mudra_ml.recommend import _classification_candidates, _regression_candidates

    pool = (
        _classification_candidates(42) if task == "classification" else _regression_candidates(42)
    )
    estimator = pool[name].estimator
    rng = np.random.default_rng(2)
    X = rng.normal(size=(200, 7))
    if task == "classification":
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
    else:
        y = 3.0 * X[:, 0] - X[:, 1] + rng.normal(scale=0.1, size=200)
    estimator.fit(X, y)
    base = estimator.predict(X)
    for _ in range(20):
        assert np.array_equal(base, estimator.predict(X))


@pytest.mark.parametrize(
    ("task", "kwargs"),
    [
        ("regression", {"target": "target"}),
        ("clustering", {"task": "clustering"}),
    ],
)
def test_two_runs_per_task_are_identical(task, kwargs, tmp_path):
    frame = make_dataset(task, "clean")
    first = _run(frame, tmp_path / "first", **kwargs)
    second = _run(frame, tmp_path / "second", **kwargs)

    assert first.evaluation == second.evaluation
    assert first.metrics == second.metrics
    first_md = (tmp_path / "first.md").read_text(encoding="utf-8")
    second_md = (tmp_path / "second.md").read_text(encoding="utf-8")
    assert first_md == second_md
