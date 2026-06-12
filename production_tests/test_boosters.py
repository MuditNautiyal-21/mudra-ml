"""The real boosters: each one builds, trains, and can be selected.

These tests exercise the actual xgboost, lightgbm, and catboost code paths
on a real dataset, not the lazy-import skip path the offline suite covers.
"""

from __future__ import annotations

import time

import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from conftest import peak_memory_mb
from mudra_ml import Mudra
from mudra_ml.decisions import DecisionLog
from mudra_ml.evaluate import evaluate
from mudra_ml.recommend import _boost_candidates

BOOSTERS = ["xgboost", "lightgbm", "catboost"]


def _real_arrays():
    data = load_breast_cancer()
    return train_test_split(data.data, data.target, test_size=0.2, random_state=42)


@pytest.mark.parametrize("name", BOOSTERS)
def test_booster_trains_and_is_selected_on_real_data(name, record):
    pytest.importorskip(name)
    candidate = _boost_candidates("classification", 42, DecisionLog())[name]
    X_train, X_test, y_train, y_test = _real_arrays()
    start = time.perf_counter()
    result = evaluate(
        candidates=[candidate],
        task="classification",
        metric="f1",
        feature_names=[f"f{i}" for i in range(X_train.shape[1])],
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    elapsed = time.perf_counter() - start
    assert result.best_name == name
    accuracy = float(result.best.test_metrics["accuracy"])
    assert accuracy > 0.9
    record(
        {
            "suite": "boosters",
            "booster": name,
            "dataset": "breast-cancer",
            "accuracy": round(accuracy, 4),
            "seconds": round(elapsed, 1),
            "status": "pass",
        }
    )


def test_full_run_offers_all_boosters(record, tmp_path):
    for name in BOOSTERS:
        pytest.importorskip(name)
    frame = load_breast_cancer(as_frame=True).frame
    start = time.perf_counter()
    result = Mudra(random_state=42).run(
        frame, target="target", report_path=tmp_path / "r", html=False
    )
    elapsed = time.perf_counter() - start
    names = {c["name"] for c in result.evaluation["candidates"]}
    assert {"xgboost", "lightgbm", "catboost"} <= names
    preds = result.predict(frame.drop(columns=["target"]).iloc[:20])
    assert len(preds) == 20
    record(
        {
            "suite": "boosters",
            "booster": "all-in-shortlist",
            "dataset": "breast-cancer",
            "selected": result.evaluation["best_name"],
            "seconds": round(elapsed, 1),
            "peak_mb": peak_memory_mb(),
            "status": "pass",
        }
    )
