"""Shared fixtures: small deterministic datasets for each task type."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer, load_iris, make_regression


@pytest.fixture
def classification_frame() -> pd.DataFrame:
    return load_breast_cancer(as_frame=True).frame


@pytest.fixture
def regression_frame() -> pd.DataFrame:
    X, y = make_regression(n_samples=300, n_features=8, noise=12.0, random_state=0)
    frame = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    frame["price"] = y
    return frame


@pytest.fixture
def clustering_frame() -> pd.DataFrame:
    return load_iris(as_frame=True).frame.drop(columns=["target"])


@pytest.fixture
def mixed_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 200
    return pd.DataFrame(
        {
            "user_id": [f"u{i}" for i in range(n)],
            "age": rng.integers(18, 80, n).astype(float),
            "city": rng.choice(["paris", "berlin", "rome", "madrid"], n),
            "signup": pd.date_range("2020-01-01", periods=n, freq="D").astype(str),
            "note": rng.choice(
                ["short note here", "a slightly longer descriptive comment about usage"], n
            ),
            "spend": rng.normal(100, 25, n),
            "churn": rng.integers(0, 2, n),
        }
    )


@pytest.fixture
def frame_with_missing() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    n = 150
    age = rng.normal(40, 10, n)
    age[:30] = np.nan
    return pd.DataFrame(
        {
            "age": age,
            "mostly_empty": [np.nan] * 130 + list(range(20)),
            "category": rng.choice(["a", "b", "c"], n),
            "label": rng.integers(0, 2, n),
        }
    )
