"""The input schema captured at training time and enforced at prediction time.

The schema records which columns the model expects, the role each column
played in preprocessing, and the categories seen during training. Prediction
validates new data against it, so a missing column, an unexpected column, a
changed type, or an unseen category raises a clear SchemaError instead of a
raw traceback from deep inside a transformer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .errors import SchemaError

ROLE_NUMERIC = "numeric"
ROLE_BOOLEAN = "boolean"
ROLE_CATEGORICAL = "categorical"
ROLE_CATEGORICAL_HIGH = "categorical_high"
ROLE_DATETIME = "datetime"
ROLE_TEXT = "text"

_BOOLEAN_TOKENS = frozenset(
    {"true", "false", "yes", "no", "t", "f", "y", "n", "0", "1", "0.0", "1.0"}
)

# Symbols stripped before a numeric parse, mirroring training-time coercion.
_NUMERIC_STRIP = str.maketrans("", "", ",$%£€ \t")

# Missing tokens accepted in numeric columns, mirroring training-time coercion.
_MISSING_TOKENS = frozenset({"--", "?", "missing"})


def _is_missing(value: Any) -> bool:
    return value is None or (not isinstance(value, str) and pd.isna(value))


def _parse_numeric(value: Any) -> float | None:
    """Parse one cell the way training-time coercion would, or return None."""
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip()
    if text.lower() in _MISSING_TOKENS:
        return float("nan")
    try:
        return float(text.translate(_NUMERIC_STRIP))
    except ValueError:
        return None


@dataclass
class InputSchema:
    """What the model expects from new data.

    Args:
        target: The training target column, or None for clustering.
        feature_columns: Columns the model consumes, in training order.
        dropped_columns: Columns present during training but unused, such as
            identifiers. They may appear in new data and are ignored.
        columns: Per-column detail: role, training dtype, and the categories
            seen during training for one-hot encoded columns.
    """

    target: str | None
    feature_columns: list[str]
    dropped_columns: list[str] = field(default_factory=list)
    columns: dict[str, dict[str, Any]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "feature_columns": list(self.feature_columns),
            "dropped_columns": list(self.dropped_columns),
            "columns": {name: dict(info) for name, info in self.columns.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InputSchema:
        return cls(
            target=data.get("target"),
            feature_columns=list(data.get("feature_columns", [])),
            dropped_columns=list(data.get("dropped_columns", [])),
            columns={name: dict(info) for name, info in data.get("columns", {}).items()},
        )

    @classmethod
    def from_training(
        cls, X: pd.DataFrame, plan: Any, target: str | None
    ) -> InputSchema:
        """Capture the schema from the training features and the preprocess plan.

        Args:
            X: The training feature frame, before transformation.
            plan: The PreprocessPlan that assigned each column a strategy.
            target: The target column name, or None.

        Returns:
            An InputSchema ready to validate new data.
        """
        roles: dict[str, str] = {}
        for name in plan.numeric:
            roles[name] = ROLE_NUMERIC
        for name in plan.boolean:
            roles[name] = ROLE_BOOLEAN
        for name in plan.categorical_low:
            roles[name] = ROLE_CATEGORICAL
        for name in plan.categorical_high:
            roles[name] = ROLE_CATEGORICAL_HIGH
        for name in plan.datetime:
            roles[name] = ROLE_DATETIME
        for name in plan.text:
            roles[name] = ROLE_TEXT

        feature_columns = [c for c in X.columns if c in roles]
        columns: dict[str, dict[str, Any]] = {}
        for name in feature_columns:
            info: dict[str, Any] = {
                "role": roles[name],
                "dtype": str(X[name].dtype),
                "categories": None,
            }
            if roles[name] == ROLE_CATEGORICAL:
                seen = X[name].dropna().unique().tolist()
                info["categories"] = sorted({str(v) for v in seen})
            columns[name] = info

        return cls(
            target=target,
            feature_columns=feature_columns,
            dropped_columns=list(plan.dropped),
            columns=columns,
        )

    def validate(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Check new data against the schema and return the usable columns.

        Args:
            frame: New rows to predict on.

        Returns:
            A copy holding exactly the feature columns, with numeric-like text
            repaired the same way training-time ingestion repaired it.

        Raises:
            SchemaError: If a column is missing or unexpected, a column type
                changed, or a one-hot encoded column carries unseen categories.
        """
        if not isinstance(frame, pd.DataFrame):
            raise SchemaError(
                "Prediction needs a pandas DataFrame with the same feature "
                "columns the model was trained on."
            )

        present = list(frame.columns)
        missing = [c for c in self.feature_columns if c not in present]
        if missing:
            raise SchemaError(
                f"The data is missing column(s) {missing} that the model was "
                f"trained on. Add them, or retrain without them."
            )

        allowed = set(self.feature_columns) | set(self.dropped_columns)
        if self.target is not None:
            allowed.add(self.target)
        extra = [c for c in present if c not in allowed]
        if extra:
            raise SchemaError(
                f"The data carries unexpected column(s) {extra} the model was "
                f"not trained on. Remove them, or retrain with them included."
            )

        out = frame[self.feature_columns].copy()
        for name in self.feature_columns:
            info = self.columns.get(name, {})
            role = info.get("role")
            if role == ROLE_NUMERIC:
                out[name] = self._validate_numeric(out[name], name)
            elif role == ROLE_BOOLEAN:
                self._validate_boolean(out[name], name)
            elif role == ROLE_CATEGORICAL:
                self._validate_categories(out[name], name, info.get("categories"))
        return out

    @staticmethod
    def _validate_numeric(series: pd.Series, name: str) -> pd.Series:
        if pd.api.types.is_numeric_dtype(series):
            return series
        parsed = []
        bad: list[Any] = []
        for value in series:
            if _is_missing(value):
                parsed.append(np.nan)
                continue
            number = _parse_numeric(value)
            if number is None:
                if len(bad) < 3:
                    bad.append(value)
                parsed.append(np.nan)
            else:
                parsed.append(number)
        if bad:
            raise SchemaError(
                f"Column '{name}' was numeric during training but now holds "
                f"values like {bad} that do not parse as numbers. Fix the "
                f"column type before predicting."
            )
        return pd.Series(parsed, index=series.index, dtype=float)

    @staticmethod
    def _validate_boolean(series: pd.Series, name: str) -> None:
        bad: list[Any] = []
        for value in series:
            if _is_missing(value) or isinstance(value, (bool, np.bool_)):
                continue
            if isinstance(value, (int, float, np.integer, np.floating)):
                if float(value) in (0.0, 1.0):
                    continue
                if len(bad) < 3:
                    bad.append(value)
                continue
            if str(value).strip().lower() in _BOOLEAN_TOKENS:
                continue
            if len(bad) < 3:
                bad.append(value)
        if bad:
            raise SchemaError(
                f"Column '{name}' was a yes-or-no column during training but "
                f"now holds values like {bad}. Use true/false, yes/no, or 0/1."
            )

    @staticmethod
    def _validate_categories(
        series: pd.Series, name: str, categories: list[str] | None
    ) -> None:
        if not categories:
            return
        known = set(categories)
        unseen = sorted(
            {
                str(v)
                for v in series.dropna().unique()
                if str(v) not in known
            }
        )
        if unseen:
            raise SchemaError(
                f"Column '{name}' holds categories {unseen[:5]} the model "
                f"never saw during training (known: {sorted(known)[:10]}). "
                f"Map them to known categories or retrain with them present."
            )
