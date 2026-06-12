"""Scale checks: the pipeline on progressively larger synthetic data.

Each size runs the full pipeline and records the wall-clock time and the
process peak memory, so slowdowns and memory growth are visible in the
results table rather than guessed at.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from conftest import peak_memory_mb
from mudra_ml import Mudra

SIZES = [1000, 10000, 100000]


def _synthetic(n: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "balance": rng.normal(2000, 700, n),
            "age": rng.uniform(18, 80, n),
            "visits": rng.poisson(4, n).astype(float),
            "rate": rng.uniform(0, 30, n),
            "region": rng.choice(["north", "south", "east", "west"], size=n),
            "plan": rng.choice(["basic", "plus", "pro"], size=n),
        }
    )
    score = (
        0.0005 * frame["balance"]
        + 0.02 * frame["age"]
        + 0.1 * frame["visits"]
        + 0.4 * (frame["plan"] == "pro").astype(float)
        + rng.normal(scale=0.5, size=n)
    )
    frame["target"] = (score > score.median()).astype(int)
    return frame


@pytest.mark.parametrize("n", SIZES)
def test_scale_run_completes_and_is_timed(n, record, tmp_path):
    frame = _synthetic(n)
    start = time.perf_counter()
    result = Mudra(random_state=42).run(
        frame, target="target", report_path=tmp_path / "r", html=True
    )
    elapsed = time.perf_counter() - start
    assert result.report_path.exists()
    assert result.metrics
    record(
        {
            "suite": "scale",
            "n_rows": n,
            "model": result.evaluation["best_name"],
            "metric": result.metric,
            "value": round(float(result.metrics[result.metric]), 4),
            "seconds": round(elapsed, 1),
            "peak_mb": peak_memory_mb(),
            "status": "pass",
        }
    )
