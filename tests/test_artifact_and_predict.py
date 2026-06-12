"""The production surface: stable result fields, save, load, and prediction.

A saved then loaded model must give identical predictions. New data is
validated against the training schema before prediction: missing columns,
unexpected columns, changed types, and unseen categories all raise a clear
MudraError instead of a raw traceback.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from mudra_ml import Mudra, MudraError, __version__


def _clean_frame(seed: int = 3, n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    income = rng.uniform(20000, 90000, n)
    age = rng.uniform(18, 75, n)
    return pd.DataFrame(
        {
            "income": income,
            "age": age,
            "region": rng.choice(["north", "south", "east"], size=n),
            "approved": ((income / 1000 + age) > 80).astype(int),
        }
    )


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("artifact")
    frame = _clean_frame()
    result = Mudra(random_state=42).run(
        frame,
        target="approved",
        report_path=tmp / "report",
        html=False,
        use_boost=False,
    )
    return result, frame


def test_result_has_the_stable_fields(trained):
    result, _ = trained
    assert result.task == "classification"
    assert result.target == "approved"
    assert isinstance(result.metrics, dict)
    assert "accuracy" in result.metrics
    assert result.positive_label is not None
    assert result.model_path is None
    assert isinstance(result.feature_names, list)
    assert result.report_path.exists()
    assert result.best_model is not None
    assert result.pipeline is not None


def test_save_writes_model_and_metadata_files(trained, tmp_path):
    result, _ = trained
    written = result.save(tmp_path / "model")
    assert written.suffix == ".joblib"
    assert written.exists()
    assert result.model_path == written

    sidecar = written.with_suffix(".json")
    assert sidecar.exists()
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    for key in (
        "library_version",
        "python_version",
        "created",
        "task",
        "target",
        "metric",
        "selected_model",
        "seed",
        "positive_label",
    ):
        assert key in meta, key
    assert meta["library_version"] == __version__
    assert meta["task"] == "classification"
    assert meta["target"] == "approved"
    assert meta["seed"] == 42

    schema = meta["input_schema"]
    assert schema["target"] == "approved"
    assert set(schema["feature_columns"]) == {"income", "age", "region"}


def test_saved_then_loaded_model_gives_identical_predictions(trained, tmp_path):
    result, frame = trained
    new_rows = frame.drop(columns=["approved"]).iloc[:25]
    before = result.predict(new_rows)

    written = result.save(tmp_path / "roundtrip")
    loaded = Mudra.load(written)
    after = loaded.predict(new_rows)

    assert np.array_equal(before, after)
    assert loaded.task == result.task
    assert loaded.target == result.target
    assert loaded.positive_label == result.positive_label
    assert loaded.model_path == written


def test_loaded_model_predict_proba_is_identical(trained, tmp_path):
    result, frame = trained
    new_rows = frame.drop(columns=["approved"]).iloc[:25]
    before = result.predict_proba(new_rows)

    written = result.save(tmp_path / "proba_roundtrip")
    loaded = Mudra.load(written)
    after = loaded.predict_proba(new_rows)

    assert np.array_equal(before, after)
    assert np.allclose(after.sum(axis=1), 1.0)


def test_predict_with_missing_column_raises(trained):
    result, frame = trained
    broken = frame.drop(columns=["approved", "income"]).iloc[:5]
    with pytest.raises(MudraError, match="income"):
        result.predict(broken)


def test_predict_with_unexpected_column_raises(trained):
    result, frame = trained
    broken = frame.drop(columns=["approved"]).iloc[:5].copy()
    broken["mystery"] = 1
    with pytest.raises(MudraError, match="mystery"):
        result.predict(broken)


def test_predict_with_changed_type_raises(trained):
    result, frame = trained
    broken = frame.drop(columns=["approved"]).iloc[:5].copy()
    broken["income"] = ["a", "b", "c", "d", "e"]
    with pytest.raises(MudraError, match="income"):
        result.predict(broken)


def test_predict_with_unseen_category_raises(trained):
    result, frame = trained
    broken = frame.drop(columns=["approved"]).iloc[:5].copy()
    broken["region"] = "atlantis"
    with pytest.raises(MudraError, match="region"):
        result.predict(broken)


def test_predict_accepts_the_target_column_if_present(trained):
    result, frame = trained
    preds = result.predict(frame.iloc[:10])
    assert len(preds) == 10


def test_predict_proba_on_regression_raises(tmp_path):
    rng = np.random.default_rng(5)
    frame = pd.DataFrame(
        {
            "a": rng.normal(size=120),
            "b": rng.normal(size=120),
        }
    )
    frame["y"] = 2.0 * frame["a"] - frame["b"] + rng.normal(scale=0.1, size=120)
    result = Mudra(random_state=42).run(
        frame,
        target="y",
        task="regression",
        report_path=tmp_path / "report",
        html=False,
        use_boost=False,
    )
    with pytest.raises(MudraError, match="proba"):
        result.predict_proba(frame.drop(columns=["y"]).iloc[:5])


def test_predict_messages_never_show_a_raw_traceback_type(trained):
    result, frame = trained
    broken = frame.drop(columns=["approved", "income"]).iloc[:5]
    try:
        result.predict(broken)
    except MudraError as exc:
        text = str(exc)
        assert "Traceback" not in text
        assert "KeyError" not in text
    else:
        raise AssertionError("expected a MudraError")
