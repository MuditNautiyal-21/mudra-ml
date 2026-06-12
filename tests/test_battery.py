"""The evaluation battery: every task type crossed with every data condition.

Each cell runs the full pipeline end to end. A cell passes when the run
completes and returns a usable result, or when it fails with a deliberate,
clearly worded MudraError. A raw pandas, numpy, or scikit-learn traceback is
always a failure.
"""

from __future__ import annotations

import pytest

from battery_datasets import CELLS, make_dataset
from mudra_ml import Mudra, MudraError

_EXPECTED_TASK = {
    "binary": "classification",
    "multiclass": "classification",
    "regression": "regression",
    "clustering": "clustering",
}


@pytest.mark.parametrize(("task", "condition"), CELLS)
def test_battery_cell(task, condition, tmp_path):
    frame = make_dataset(task, condition)
    m = Mudra(random_state=42)
    kwargs = {"report_path": tmp_path / "report", "html": False, "use_boost": False}
    if task == "clustering":
        kwargs["task"] = "clustering"
    else:
        kwargs["target"] = "target"

    result = m.run(frame, **kwargs)

    assert result.task == _EXPECTED_TASK[task]
    assert result.metrics, "the selected model carries held-out metrics"
    assert result.report_path.exists()
    assert result.input_schema["feature_columns"], "the schema records features"

    preds = result.predict(frame.head(5))
    assert len(preds) == 5

    if condition == "leakage":
        rules = {entry.rule for entry in m.log}
        assert "leakage-suspect" in rules, "the planted leak is flagged"


def test_battery_covers_every_applicable_cell():
    tasks = {t for t, _ in CELLS}
    conditions = {c for _, c in CELLS}
    assert tasks == set(_EXPECTED_TASK)
    assert len(conditions) == 14
    assert len(CELLS) == 53


def test_single_class_target_is_a_deliberate_error(tmp_path):
    frame = make_dataset("binary", "clean")
    frame["target"] = 1
    with pytest.raises(MudraError, match="one class"):
        Mudra(random_state=42).run(
            frame, target="target", report_path=tmp_path / "report", html=False
        )
