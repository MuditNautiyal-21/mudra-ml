"""The full pipeline against real public datasets.

Each dataset runs end to end with its true target: the run must complete,
return sensible held-out metrics, write a report, and round-trip through
save, load, and predict. The dataset, task, selected model, headline
metric, and run time are recorded for the results table.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import (
    fetch_california_housing,
    fetch_openml,
    load_breast_cancer,
    load_diabetes,
    load_wine,
)

from conftest import peak_memory_mb
from mudra_ml import Mudra

# name, loader kind, loader spec, task, floor for the sanity metric
DATASETS = [
    ("titanic", "openml", {"name": "titanic", "version": 1}, "classification", 0.7),
    ("adult", "openml", {"name": "adult", "version": 2}, "classification", 0.7),
    ("credit-g", "openml", {"name": "credit-g", "version": 1}, "classification", 0.65),
    ("bank-marketing", "openml", {"name": "bank-marketing", "version": 1}, "classification", 0.8),
    ("breast-cancer", "sklearn", "breast_cancer", "classification", 0.9),
    ("wine", "sklearn", "wine", "classification", 0.85),
    ("california-housing", "sklearn", "california", "regression", 0.6),
    ("diabetes", "sklearn", "diabetes", "regression", 0.3),
]


def _load(kind: str, spec) -> tuple[pd.DataFrame, str]:
    if kind == "openml":
        bunch = fetch_openml(**spec, as_frame=True)
        target = bunch.target.name if hasattr(bunch.target, "name") else "target"
        return bunch.frame, str(target)
    loaders = {
        "breast_cancer": load_breast_cancer,
        "wine": load_wine,
        "diabetes": load_diabetes,
    }
    if spec == "california":
        frame = fetch_california_housing(as_frame=True).frame
        return frame, "MedHouseVal"
    frame = loaders[spec](as_frame=True).frame
    return frame, "target"


@pytest.mark.parametrize(
    ("name", "kind", "spec", "task", "floor"),
    DATASETS,
    ids=[d[0] for d in DATASETS],
)
def test_real_dataset_end_to_end(name, kind, spec, task, floor, record, tmp_path):
    frame, target = _load(kind, spec)
    row = {
        "suite": "real_datasets",
        "dataset": name,
        "n_rows": int(len(frame)),
        "n_columns": int(frame.shape[1]),
    }
    start = time.perf_counter()
    try:
        result = Mudra(random_state=42).run(
            frame, target=target, report_path=tmp_path / "r", html=True
        )
    except Exception as exc:
        row.update(
            {
                "status": "fail",
                "error": f"{type(exc).__name__}: {exc}",
                "seconds": round(time.perf_counter() - start, 1),
            }
        )
        record(row)
        raise
    elapsed = time.perf_counter() - start

    assert result.task == task
    assert result.report_path.exists()
    headline = result.metrics.get(result.metric)
    assert headline is not None and np.isfinite(float(headline))

    sanity_key = "accuracy" if task == "classification" else "r2"
    sanity = float(result.metrics[sanity_key])
    assert sanity >= floor, f"{sanity_key}={sanity:.3f} is below the floor {floor}"

    holdout = frame.drop(columns=[target]).iloc[:50]
    written = result.save(tmp_path / "model")
    loaded = Mudra.load(written)
    before = result.predict(holdout)
    after = loaded.predict(holdout)
    assert np.array_equal(before, after)

    row.update(
        {
            "status": "pass",
            "task": result.task,
            "model": result.evaluation["best_name"],
            "metric": result.metric,
            "value": round(float(headline), 4),
            sanity_key: round(sanity, 4),
            "seconds": round(elapsed, 1),
            "peak_mb": peak_memory_mb(),
        }
    )
    record(row)
