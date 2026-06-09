"""Dirty-data robustness battery for the 0.3.0 upgrade.

Every test asserts that a run either produces a fitted model or fails with a
specific MudraError. No raw pandas or scikit-learn traceback should escape.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mudra_ml import Mudra, MudraError
from mudra_ml.ingest import coerce_numeric_like


def _signal_frame(labels, n=240, seed=0):
    """A binary frame whose target is predictable from x1, labelled by `labels`."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0.0, 1.0, n)
    x2 = rng.normal(0.0, 1.0, n)
    decided = x1 + rng.normal(0.0, 0.3, n) > 0
    target = np.where(decided, labels[1], labels[0])
    return pd.DataFrame({"x1": x1, "x2": x2, "target": target})


def _run(frame, **kwargs):
    kwargs.setdefault("report_path", "_t_report")
    return Mudra(random_state=42).run(frame, html=False, **kwargs)


def _best_metrics(result):
    ev = result.evaluation
    best = next(c for c in ev["candidates"] if c["name"] == ev["best_name"])
    return best["test_metrics"]


def test_binary_string_labels_yes_no(tmp_path):
    frame = _signal_frame(("no", "yes"))
    result = _run(frame, target="target", report_path=str(tmp_path / "r"))
    assert result.task == "classification"
    assert result.best_model is not None
    assert result.evaluation["best_name"]
    # The chosen positive class is recorded in the report and the evaluation.
    assert result.evaluation["positive_label"] in {"yes", "no"}
    md = result.report_path.read_text(encoding="utf-8")
    assert "Positive class" in md


def test_binary_labels_one_two(tmp_path):
    frame = _signal_frame((1, 2))
    result = _run(frame, target="target", report_path=str(tmp_path / "r"))
    assert result.task == "classification"
    assert result.best_model is not None
    assert "f1" in _best_metrics(result)


def test_binary_labels_zero_one(tmp_path):
    frame = _signal_frame((0, 1))
    result = _run(frame, target="target", report_path=str(tmp_path / "r"))
    assert result.task == "classification"
    assert result.best_model is not None
    assert "f1" in _best_metrics(result)


def test_multiclass_still_works(tmp_path):
    rng = np.random.default_rng(3)
    n = 300
    x1 = rng.normal(0.0, 1.0, n)
    x2 = rng.normal(0.0, 1.0, n)
    target = np.select(
        [x1 + x2 > 1.0, x1 + x2 < -1.0], ["high", "low"], default="mid"
    )
    frame = pd.DataFrame({"x1": x1, "x2": x2, "target": target})
    result = _run(frame, target="target", report_path=str(tmp_path / "r"))
    assert result.task == "classification"
    assert result.best_model is not None
    assert result.evaluation["positive_label"] is None
    assert "f1" in _best_metrics(result)


def test_boolean_feature_columns_do_not_crash(tmp_path):
    rng = np.random.default_rng(5)
    n = 240
    flag_a = rng.random(n) > 0.5
    flag_b = rng.random(n) > 0.7
    target = np.where(flag_a, "yes", "no")
    frame = pd.DataFrame({"flag_a": flag_a, "flag_b": flag_b, "score": rng.normal(0, 1, n),
                          "target": target})
    result = _run(frame, target="target", report_path=str(tmp_path / "r"))
    assert result.task == "classification"
    assert result.best_model is not None


def test_coerce_thousands_separator():
    frame = pd.DataFrame({"income": ["1,200", "3,400", "2,100", "5,600"]})
    out = coerce_numeric_like(frame)
    assert pd.api.types.is_numeric_dtype(out["income"])
    assert out["income"].tolist() == [1200.0, 3400.0, 2100.0, 5600.0]


def test_coerce_currency_and_percent():
    frame = pd.DataFrame({"price": ["$10.5", "$20.1", "$30.0"], "rate": ["10%", "20%", "30%"]})
    out = coerce_numeric_like(frame)
    assert out["price"].tolist() == [10.5, 20.1, 30.0]
    assert out["rate"].tolist() == [10.0, 20.0, 30.0]


def test_coerce_double_dash_and_missing_tokens():
    frame = pd.DataFrame({"bmi": ["28.1", "--", "30.5", "missing", "19.9", "?"]})
    out = coerce_numeric_like(frame)
    assert pd.api.types.is_numeric_dtype(out["bmi"])
    assert out["bmi"].isna().sum() == 3
    assert out["bmi"].dropna().tolist() == [28.1, 30.5, 19.9]


def test_protected_category_not_coerced_to_missing():
    # A column dominated by numbers but with a real 'Unknown' category must NOT
    # be coerced, so 'Unknown' is never silently turned into missing.
    frame = pd.DataFrame({"col": ["1.0", "2.0", "Unknown", "3.0", "Unknown", "4.0"]})
    out = coerce_numeric_like(frame)
    assert out["col"].tolist() == frame["col"].tolist()


def test_pandas_default_tokens_not_re_added():
    # 'NA' is a pandas default already handled at read time; an in-memory frame
    # with literal 'NA' strings should not be coerced by our extra-token rule.
    frame = pd.DataFrame({"col": ["red", "blue", "NA", "green", "red"]})
    out = coerce_numeric_like(frame)
    assert out["col"].tolist() == frame["col"].tolist()


def test_dirty_numeric_runs_end_to_end(tmp_path):
    rng = np.random.default_rng(7)
    n = 240
    raw = rng.normal(50, 10, n)
    income = [f"{v:,.0f}" if i % 9 else "--" for i, v in enumerate(raw)]
    target = np.where(raw > 50, "high", "low")
    frame = pd.DataFrame({"income": income, "x": rng.normal(0, 1, n), "target": target})
    result = _run(frame, target="target", report_path=str(tmp_path / "r"))
    assert result.task == "classification"
    assert result.best_model is not None
    md = result.report_path.read_text(encoding="utf-8")
    assert "dirty-numeric-coercion" in md


def test_single_class_target_raises_mudra_error(tmp_path):
    rng = np.random.default_rng(8)
    n = 120
    frame = pd.DataFrame(
        {"x1": rng.normal(0, 1, n), "x2": rng.normal(0, 1, n), "target": ["yes"] * n}
    )
    with pytest.raises(MudraError, match="target"):
        _run(frame, target="target", report_path=str(tmp_path / "r"))


def test_class_too_small_raises_mudra_error(tmp_path):
    rng = np.random.default_rng(9)
    n = 200
    y = ["no"] * (n - 1) + ["yes"]
    frame = pd.DataFrame(
        {"x1": rng.normal(0, 1, n), "x2": rng.normal(0, 1, n), "target": y}
    )
    with pytest.raises(MudraError):
        _run(frame, target="target", report_path=str(tmp_path / "r"))


def test_severe_imbalance_still_trains(tmp_path):
    rng = np.random.default_rng(10)
    n = 400
    x1 = rng.normal(0, 1, n)
    positive = rng.random(n) < 0.08
    y = np.where(positive, "pos", "neg")
    frame = pd.DataFrame({"x1": x1, "x2": rng.normal(0, 1, n), "target": y})
    result = _run(frame, target="target", report_path=str(tmp_path / "r"))
    assert result.task == "classification"
    assert result.best_model is not None


def test_two_runs_produce_identical_reports(tmp_path):
    rng = np.random.default_rng(11)
    n = 260
    income = [f"{v:,.0f}" if i % 7 else "--" for i, v in enumerate(rng.normal(50, 10, n))]
    frame = pd.DataFrame(
        {
            "income": income,
            "flag": rng.random(n) > 0.5,
            "x": rng.normal(0, 1, n),
            "target": np.where(rng.random(n) > 0.5, "yes", "no"),
        }
    )
    first = Mudra(random_state=42).run(frame, target="target", html=False,
                                       report_path=str(tmp_path / "a"))
    second = Mudra(random_state=42).run(frame, target="target", html=False,
                                        report_path=str(tmp_path / "b"))
    assert first.evaluation["best_name"] == second.evaluation["best_name"]
    assert _best_metrics(first) == _best_metrics(second)


def test_all_missing_column_does_not_crash(tmp_path):
    rng = np.random.default_rng(13)
    n = 200
    frame = pd.DataFrame(
        {
            "x": rng.normal(0, 1, n),
            "empty": [np.nan] * n,
            "target": np.where(rng.random(n) > 0.5, "yes", "no"),
        }
    )
    result = _run(frame, target="target", report_path=str(tmp_path / "r"))
    assert result.best_model is not None


def test_near_constant_column_does_not_crash(tmp_path):
    rng = np.random.default_rng(14)
    n = 200
    near = np.zeros(n)
    near[0] = 1.0
    frame = pd.DataFrame(
        {
            "x": rng.normal(0, 1, n),
            "near": near,
            "target": np.where(rng.random(n) > 0.5, "yes", "no"),
        }
    )
    result = _run(frame, target="target", report_path=str(tmp_path / "r"))
    assert result.best_model is not None


def test_high_cardinality_text_column_does_not_crash(tmp_path):
    rng = np.random.default_rng(15)
    n = 200
    notes = [f"free text comment number {i} with several words" for i in range(n)]
    frame = pd.DataFrame(
        {
            "x": rng.normal(0, 1, n),
            "notes": notes,
            "target": np.where(rng.random(n) > 0.5, "yes", "no"),
        }
    )
    result = _run(frame, target="target", report_path=str(tmp_path / "r"))
    assert result.best_model is not None


def test_stroke_like_replica_trains(tmp_path):
    """A faithful stroke replica: numeric bmi polluted with 'N/A', a unique id,
    ~6 percent positive imbalance, and an 'Unknown' smoking category."""
    rng = np.random.default_rng(12)
    n = 1500
    bmi_num = np.round(rng.normal(28.9, 7.7, n), 1)
    bmi = np.array([f"{v:.1f}" for v in bmi_num], dtype=object)
    bmi[rng.random(n) < 0.04] = "N/A"
    frame = pd.DataFrame(
        {
            "id": rng.permutation(np.arange(10000, 10000 + n)),
            "age": np.round(rng.uniform(1.0, 82.0, n), 1),
            "bmi": bmi,
            "smoking_status": rng.choice(
                ["never smoked", "Unknown", "formerly smoked", "smokes"], n
            ),
            "stroke": (rng.random(n) < 0.06).astype(int),
        }
    )
    result = _run(frame, target="stroke", report_path=str(tmp_path / "r"))
    assert result.task == "classification"
    assert result.best_model is not None
    # bmi must survive as a usable numeric feature, and Unknown stays a category.
    cols = result.profile["columns"]
    assert cols["bmi"]["inferred_type"] == "numeric"
    assert cols["smoking_status"]["inferred_type"] == "categorical"


def test_data_eval_stroke_csv_smoke(tmp_path):
    csv = Path(__file__).resolve().parent.parent / "data_eval" / (
        "healthcare-dataset-stroke-data.csv"
    )
    if not csv.exists():
        pytest.skip("operator stroke CSV not present in data_eval/")
    result = Mudra(random_state=42).run(
        str(csv), target="stroke", html=False, report_path=str(tmp_path / "r")
    )
    assert result.task == "classification"
    assert result.best_model is not None
