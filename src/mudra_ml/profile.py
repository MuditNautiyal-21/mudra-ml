"""Data profiling: per-column type inference and dataset statistics."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .constants import (
    CATEGORICAL_MAX_RATIO,
    CATEGORICAL_MAX_UNIQUE,
    DISCRETE_NUMERIC_MAX_UNIQUE,
    ID_UNIQUE_RATIO,
    TEXT_MIN_AVG_LENGTH,
    TEXT_MIN_WORD_COUNT,
)
from .decisions import DecisionLog

# Inferred semantic types for a column.
NUMERIC = "numeric"
CATEGORICAL = "categorical"
DATETIME = "datetime"
BOOLEAN = "boolean"
ID = "id"
TEXT = "text"


@dataclass
class ColumnProfile:
    """Profile for one column."""

    name: str
    inferred_type: str
    dtype: str
    missing_count: int
    missing_fraction: float
    n_unique: int
    unique_ratio: float
    sample_values: list[Any] = field(default_factory=list)
    stats: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DataProfile:
    """Profile for a full dataset."""

    n_rows: int
    n_columns: int
    columns: dict[str, ColumnProfile]
    candidate_targets: list[str]
    duplicate_rows: int

    def column(self, name: str) -> ColumnProfile:
        return self.columns[name]

    def columns_of_type(self, inferred_type: str) -> list[str]:
        return [c.name for c in self.columns.values() if c.inferred_type == inferred_type]

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "duplicate_rows": self.duplicate_rows,
            "candidate_targets": self.candidate_targets,
            "columns": {name: col.as_dict() for name, col in self.columns.items()},
        }


def _is_textual(series: pd.Series) -> bool:
    """True for object or string dtype, the carriers of free text and labels."""
    return pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)


def _looks_like_datetime(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if not _is_textual(series):
        return False
    sample = series.dropna().head(50)
    if sample.empty:
        return False
    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return parsed.notna().mean() >= 0.8


def _is_boolean(series: pd.Series, n_unique: int) -> bool:
    if pd.api.types.is_bool_dtype(series):
        return True
    if n_unique == 0 or n_unique > 2:
        return False
    non_null = series.dropna()
    if pd.api.types.is_numeric_dtype(series):
        try:
            arr = np.asarray(non_null, dtype=float)
        except (TypeError, ValueError):
            return False
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            return False
        if not np.allclose(arr, np.round(arr)):
            return False
        ints = {int(round(v)) for v in arr}
        return ints.issubset({0, 1})
    values = {str(v).strip().lower() for v in non_null.unique()}
    boolean_sets = [
        {"true", "false"},
        {"yes", "no"},
        {"t", "f"},
        {"y", "n"},
    ]
    return any(values.issubset(s) for s in boolean_sets) and len(values) > 0


def _is_integer_like(series: pd.Series) -> bool:
    """True when every non-null value equals its rounded form."""
    non_null = series.dropna()
    if non_null.empty:
        return False
    try:
        arr = np.asarray(non_null, dtype=float)
    except (TypeError, ValueError):
        return False
    if not np.all(np.isfinite(arr)):
        return False
    return bool(np.allclose(arr, np.round(arr)))


def _avg_text_length(series: pd.Series) -> float:
    sample = series.dropna().astype(str).head(200)
    if sample.empty:
        return 0.0
    return float(sample.str.len().mean())


def _avg_word_count(series: pd.Series) -> float:
    sample = series.dropna().astype(str).head(200)
    if sample.empty:
        return 0.0
    return float(sample.str.split().apply(len).mean())


def _infer_type(series: pd.Series, n_unique: int, n_rows: int) -> str:
    non_null = series.dropna()
    unique_ratio = n_unique / max(len(non_null), 1)

    if _is_boolean(series, n_unique):
        return BOOLEAN
    if _looks_like_datetime(series):
        return DATETIME
    if pd.api.types.is_numeric_dtype(series):
        if unique_ratio >= ID_UNIQUE_RATIO and n_unique > CATEGORICAL_MAX_UNIQUE:
            if _is_integer_like(series):
                return ID
        # Low-cardinality integer columns are discrete labels. IQR clipping
        # and continuous scaling would distort them. Route to categorical
        # so they get one-hot encoded instead.
        if (
            3 <= n_unique <= DISCRETE_NUMERIC_MAX_UNIQUE
            and _is_integer_like(series)
        ):
            return CATEGORICAL
        return NUMERIC

    if unique_ratio >= ID_UNIQUE_RATIO and n_unique > CATEGORICAL_MAX_UNIQUE:
        if _avg_word_count(series) >= TEXT_MIN_WORD_COUNT:
            return TEXT
        return ID

    long_enough = _avg_text_length(series) >= TEXT_MIN_AVG_LENGTH
    wordy_enough = _avg_word_count(series) >= TEXT_MIN_WORD_COUNT
    if long_enough and wordy_enough:
        return TEXT

    if n_unique <= CATEGORICAL_MAX_UNIQUE or unique_ratio <= CATEGORICAL_MAX_RATIO:
        return CATEGORICAL
    return TEXT


def _numeric_stats(series: pd.Series) -> dict[str, float]:
    non_null = series.dropna()
    if non_null.empty:
        return {}
    return {
        "min": float(non_null.min()),
        "max": float(non_null.max()),
        "mean": float(non_null.mean()),
        "median": float(non_null.median()),
        "std": float(non_null.std(ddof=0)),
        "skew": float(non_null.skew()) if len(non_null) > 2 else 0.0,
    }


class DataProfiler:
    """Build a structured profile of a DataFrame.

    The profile drives goal inference and preprocessing. It records each
    column type decision and the rule behind it in the decision log.
    """

    def __init__(self, log: DecisionLog | None = None) -> None:
        self.log = log if log is not None else DecisionLog()

    def profile(self, frame: pd.DataFrame) -> DataProfile:
        """Profile every column and detect candidate target columns.

        Args:
            frame: The dataset to profile.

        Returns:
            A DataProfile describing the dataset.
        """
        n_rows = len(frame)
        columns: dict[str, ColumnProfile] = {}

        for name in frame.columns:
            series = frame[name]
            missing = int(series.isna().sum())
            n_unique = int(series.nunique(dropna=True))
            non_null_count = n_rows - missing
            unique_ratio = n_unique / max(non_null_count, 1)
            inferred = _infer_type(series, n_unique, n_rows)

            stats = _numeric_stats(series) if inferred == NUMERIC else {}
            sample = [
                self._safe(v) for v in series.dropna().unique()[:5].tolist()
            ]

            columns[name] = ColumnProfile(
                name=str(name),
                inferred_type=inferred,
                dtype=str(series.dtype),
                missing_count=missing,
                missing_fraction=round(missing / max(n_rows, 1), 4),
                n_unique=n_unique,
                unique_ratio=round(unique_ratio, 4),
                sample_values=sample,
                stats=stats,
            )
            self.log.record(
                "profile",
                f"Column '{name}' typed as {inferred}.",
                "type-inference",
                {
                    "dtype": str(series.dtype),
                    "n_unique": n_unique,
                    "unique_ratio": round(unique_ratio, 4),
                    "missing_fraction": round(missing / max(n_rows, 1), 4),
                },
            )
            if (
                inferred == CATEGORICAL
                and pd.api.types.is_numeric_dtype(series)
                and 3 <= n_unique <= DISCRETE_NUMERIC_MAX_UNIQUE
            ):
                self.log.record(
                    "profile",
                    f"Column '{name}' is an integer column with {n_unique} "
                    f"distinct values. Treated as a discrete label, not a "
                    f"continuous measurement, to keep outlier clipping and "
                    f"scaling from distorting the signal.",
                    "discrete-low-cardinality-integer",
                    {
                        "n_unique": n_unique,
                        "threshold": DISCRETE_NUMERIC_MAX_UNIQUE,
                    },
                )

        candidates = self._candidate_targets(columns, n_rows)
        duplicates = int(frame.duplicated().sum())

        return DataProfile(
            n_rows=n_rows,
            n_columns=frame.shape[1],
            columns=columns,
            candidate_targets=candidates,
            duplicate_rows=duplicates,
        )

    def _candidate_targets(
        self, columns: dict[str, ColumnProfile], n_rows: int
    ) -> list[str]:
        """Rank columns by how plausible they are as a supervised target."""
        scored: list[tuple[float, str]] = []
        for col in columns.values():
            if col.inferred_type in (ID, TEXT, DATETIME):
                continue
            if col.missing_fraction > 0.2:
                continue
            score = 0.0
            name_lower = col.name.lower()
            if any(k in name_lower for k in ("target", "label", "class", "outcome", "y")):
                score += 3.0
            if any(k in name_lower for k in ("churn", "price", "sales", "default", "fraud")):
                score += 2.0
            if col.inferred_type == BOOLEAN:
                score += 1.5
            if col.inferred_type == CATEGORICAL and col.n_unique <= 10:
                score += 1.0
            if col.inferred_type == NUMERIC:
                score += 0.5
            scored.append((score, col.name))

        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        ranked = [name for score, name in scored if score > 0]
        if not ranked and scored:
            ranked = [scored[-1][1]]
        if ranked:
            self.log.record(
                "profile",
                f"Candidate target columns ranked: {ranked[:3]}.",
                "candidate-target-detection",
                {"top": ranked[:3]},
            )
        return ranked

    @staticmethod
    def _safe(value: Any) -> Any:
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        return value
