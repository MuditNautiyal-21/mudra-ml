"""Deterministic dataset generators for the evaluation battery.

The battery crosses task types with data conditions. Every generator is
seeded, so two runs build byte-identical frames. The datasets are synthetic
on purpose: the battery must run offline and reproduce exactly, and every
condition (planted leakage, dirty tokens, constant columns) must be present
by construction, not by luck.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TASKS = ("binary", "multiclass", "regression", "clustering")

CONDITIONS = (
    "clean",
    "dirty_numeric",
    "missing_heavy",
    "imbalanced",
    "high_cardinality",
    "wide",
    "tiny",
    "mixed_types",
    "boolean",
    "datetime",
    "id_column",
    "duplicates",
    "constant",
    "leakage",
)

# Conditions that only make sense for some tasks.
_ONLY_FOR = {
    "imbalanced": ("binary", "multiclass"),
    "leakage": ("binary", "multiclass", "regression"),
}

CELLS = [
    (task, condition)
    for task in TASKS
    for condition in CONDITIONS
    if task in _ONLY_FOR.get(condition, TASKS)
]


def _base_features(rng: np.random.Generator, n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x1": rng.normal(loc=50, scale=12, size=n),
            "x2": rng.normal(loc=0, scale=1, size=n),
            "x3": rng.uniform(0, 100, size=n),
            "x4": rng.normal(loc=-5, scale=3, size=n),
            "colour": rng.choice(["red", "green", "blue"], size=n),
        }
    )


def _score(frame: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    return (
        0.04 * (frame["x1"].to_numpy() - 50)
        + 0.8 * frame["x2"].to_numpy()
        - 0.01 * (frame["x3"].to_numpy() - 50)
        + 0.6 * (frame["colour"] == "red").to_numpy()
        + rng.normal(scale=0.25, size=len(frame))
    )


def _supervised_base(task: str, rng: np.random.Generator, n: int) -> pd.DataFrame:
    frame = _base_features(rng, n)
    score = _score(frame, rng)
    if task == "binary":
        frame["target"] = (score > np.median(score)).astype(int)
    elif task == "multiclass":
        low, high = np.quantile(score, [0.33, 0.66])
        frame["target"] = np.select(
            [score <= low, score <= high], ["low", "mid"], default="high"
        )
    elif task == "regression":
        frame["target"] = 100.0 + 40.0 * score + rng.normal(scale=2.0, size=n)
    return frame


def _clustering_base(rng: np.random.Generator, n: int) -> pd.DataFrame:
    centers = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [6.0, 6.0, 0.0, -4.0],
            [-6.0, 4.0, 5.0, 3.0],
        ]
    )
    assignments = rng.integers(0, 3, size=n)
    points = centers[assignments] + rng.normal(scale=1.0, size=(n, 4))
    return pd.DataFrame(points, columns=["x1", "x2", "x3", "x4"])


def _wide(task: str, rng: np.random.Generator) -> pd.DataFrame:
    n, p = 40, 60
    X = rng.normal(size=(n, p))
    frame = pd.DataFrame(X, columns=[f"f{i:02d}" for i in range(p)])
    score = X[:, 0] + 0.8 * X[:, 1] - 0.6 * X[:, 2]
    if task == "binary":
        frame["target"] = (score > np.median(score)).astype(int)
    elif task == "multiclass":
        low, high = np.quantile(score, [0.33, 0.66])
        frame["target"] = np.select(
            [score <= low, score <= high], ["low", "mid"], default="high"
        )
    elif task == "regression":
        frame["target"] = 10.0 * score + rng.normal(scale=0.5, size=n)
    return frame


def _dates(rng: np.random.Generator, n: int) -> list[str]:
    days = rng.integers(0, 365, size=n)
    base = pd.Timestamp("2024-01-01")
    return [(base + pd.Timedelta(days=int(d))).strftime("%Y-%m-%d") for d in days]


def _sentences(rng: np.random.Generator, n: int) -> list[str]:
    items = ["lamps", "chairs", "tables", "shelves", "stools"]
    notes = ["deliver to the rear entrance", "fragile contents inside", "standard handling applies"]
    return [
        f"order of {rng.integers(1, 9)} {items[int(rng.integers(0, len(items)))]}, "
        f"{notes[int(rng.integers(0, len(notes)))]}"
        for _ in range(n)
    ]


def _apply_condition(
    frame: pd.DataFrame, condition: str, task: str, rng: np.random.Generator
) -> pd.DataFrame:
    frame = frame.copy()
    n = len(frame)
    if condition == "clean":
        return frame

    if condition == "dirty_numeric":
        frame["x1"] = [f"${v:,.2f}" for v in frame["x1"]]
        frame["x3"] = [f"{v:.1f}%" for v in frame["x3"]]
        dirty_rows = rng.choice(n, size=max(2, n // 15), replace=False)
        frame.loc[frame.index[dirty_rows], "x1"] = "--"
        token_rows = rng.choice(n, size=max(2, n // 20), replace=False)
        frame.loc[frame.index[token_rows], "x3"] = "?"
        return frame

    if condition == "missing_heavy":
        for col in ("x1", "x2", "colour"):
            holes = rng.choice(n, size=int(n * 0.35), replace=False)
            frame.loc[frame.index[holes], col] = np.nan
        return frame

    if condition == "imbalanced":
        score = pd.to_numeric(frame["x2"], errors="coerce").fillna(0.0)
        if task == "binary":
            cut = score.quantile(0.9)
            frame["target"] = (score > cut).astype(int)
        else:
            low, high = score.quantile(0.80), score.quantile(0.95)
            frame["target"] = np.select(
                [score <= low, score <= high], ["common", "uncommon"], default="rare"
            )
        return frame

    if condition == "high_cardinality":
        frame["customer"] = [f"cust_{int(v):03d}" for v in rng.integers(0, 60, size=n)]
        return frame

    if condition == "tiny":
        return frame

    if condition == "mixed_types":
        frame["active"] = rng.choice([True, False], size=n)
        frame["signup_date"] = _dates(rng, n)
        frame["note"] = _sentences(rng, n)
        return frame

    if condition == "boolean":
        frame["active"] = rng.choice([True, False], size=n)
        frame["subscribed"] = rng.choice(["yes", "no"], size=n)
        return frame

    if condition == "datetime":
        frame["signup_date"] = _dates(rng, n)
        return frame

    if condition == "id_column":
        frame["record_id"] = np.arange(1000, 1000 + n)
        return frame

    if condition == "duplicates":
        extra = frame.iloc[: max(2, n // 7)]
        return pd.concat([frame, extra], ignore_index=True)

    if condition == "constant":
        frame["site"] = "main"
        return frame

    if condition == "leakage":
        frame["leak"] = frame["target"].to_numpy()
        return frame

    raise ValueError(f"Unknown condition '{condition}'.")


def make_dataset(task: str, condition: str, seed: int = 42) -> pd.DataFrame:
    """Build the battery dataset for one cell of the matrix.

    Args:
        task: binary, multiclass, regression, or clustering.
        condition: One of CONDITIONS.
        seed: Seed for the generator. The same seed gives the same frame.

    Returns:
        A DataFrame. Supervised tasks carry a ``target`` column.
    """
    rng = np.random.default_rng(seed)
    if condition == "wide":
        if task == "clustering":
            n, p = 40, 60
            X = rng.normal(size=(n, p))
            X[: n // 2] += 5.0
            return pd.DataFrame(X, columns=[f"f{i:02d}" for i in range(p)])
        return _wide(task, rng)

    n = 30 if condition == "tiny" else 200
    if task == "clustering":
        base = _clustering_base(rng, n)
    else:
        base = _supervised_base(task, rng, n)
    return _apply_condition(base, condition, task, rng)
