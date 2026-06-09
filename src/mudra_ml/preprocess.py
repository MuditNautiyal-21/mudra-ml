"""Leakage-safe cleaning and preprocessing assembled as a scikit-learn Pipeline.

Every transformer learns its parameters during fit. When the pipeline is fit on
the training split only, no statistic from the test split can reach the model.
The leakage test in the suite checks this property directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .constants import (
    DEFAULT_MISSING_DROP_THRESHOLD,
    HIGH_CARDINALITY_THRESHOLD,
    IQR_MULTIPLIER,
    ZSCORE_THRESHOLD,
)
from .decisions import DecisionLog
from .profile import (
    BOOLEAN,
    CATEGORICAL,
    DATETIME,
    ID,
    NUMERIC,
    TEXT,
    DataProfile,
)


class DatetimeFeatures(BaseEstimator, TransformerMixin):
    """Expand datetime columns into numeric parts (year, month, day, weekday)."""

    def __init__(self, columns: list[str]) -> None:
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: Any = None) -> DatetimeFeatures:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.columns:
            if col not in X.columns:
                continue
            parsed = pd.to_datetime(X[col], errors="coerce", format="mixed")
            X[f"{col}__year"] = parsed.dt.year
            X[f"{col}__month"] = parsed.dt.month
            X[f"{col}__day"] = parsed.dt.day
            X[f"{col}__weekday"] = parsed.dt.weekday
            X = X.drop(columns=[col])
        return X

    def get_feature_names_out(self, input_features: Any = None) -> Any:
        return input_features


class TextLengthFeatures(BaseEstimator, TransformerMixin):
    """Replace text columns with length and word-count features."""

    def __init__(self, columns: list[str]) -> None:
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: Any = None) -> TextLengthFeatures:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.columns:
            if col not in X.columns:
                continue
            text = X[col].fillna("").astype(str)
            X[f"{col}__char_len"] = text.str.len()
            X[f"{col}__word_count"] = text.str.split().apply(len)
            X = X.drop(columns=[col])
        return X

    def get_feature_names_out(self, input_features: Any = None) -> Any:
        return input_features


class OutlierClipper(BaseEstimator, TransformerMixin):
    """Clip numeric outliers to bounds learned on the training data.

    Bounds come from the IQR rule or a z-score rule. Because the bounds are
    learned during fit, clipping is leakage-safe.
    """

    def __init__(self, strategy: str = "iqr") -> None:
        self.strategy = strategy

    def fit(self, X: Any, y: Any = None) -> OutlierClipper:
        data = np.asarray(X, dtype=float)
        if self.strategy == "zscore":
            mean = np.nanmean(data, axis=0)
            std = np.nanstd(data, axis=0)
            std = np.where(std == 0, 1.0, std)
            self.lower_ = mean - ZSCORE_THRESHOLD * std
            self.upper_ = mean + ZSCORE_THRESHOLD * std
        else:
            q1 = np.nanpercentile(data, 25, axis=0)
            q3 = np.nanpercentile(data, 75, axis=0)
            iqr = q3 - q1
            self.lower_ = q1 - IQR_MULTIPLIER * iqr
            self.upper_ = q3 + IQR_MULTIPLIER * iqr
        return self

    def transform(self, X: Any) -> np.ndarray:
        data = np.asarray(X, dtype=float)
        return np.clip(data, self.lower_, self.upper_)

    def get_feature_names_out(self, input_features: Any = None) -> Any:
        return input_features


class BooleanToNumeric(BaseEstimator, TransformerMixin):
    """Cast a boolean column to a clean float 0/1 array.

    Accepts numeric 0/1 (int or float), Python and numpy bool, and the
    common string forms (true/false, yes/no, t/f, y/n). Anything else
    maps to 0. The transformer learns nothing during fit, so it is
    leakage-safe by construction.
    """

    _TRUE_TOKENS = frozenset({"true", "yes", "t", "y", "1", "1.0"})

    def fit(self, X: Any, y: Any = None) -> BooleanToNumeric:
        arr = np.asarray(X)
        # n_features_in_ records the fit shape, so scikit-learn's
        # check_is_fitted recognises this transformer as fitted.
        self.n_features_in_ = arr.shape[1] if arr.ndim > 1 else 1
        return self

    def transform(self, X: Any) -> np.ndarray:
        arr = np.asarray(X, dtype=object)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        out = np.zeros(arr.shape, dtype=float)
        for j in range(arr.shape[1]):
            column = arr[:, j]
            for i, value in enumerate(column):
                # Missing stays missing (NaN) so a following imputer can fill
                # it with the mode rather than silently forcing it to False.
                if value is None or (not isinstance(value, str) and pd.isna(value)):
                    out[i, j] = np.nan
                    continue
                if isinstance(value, (bool, np.bool_)):
                    out[i, j] = 1.0 if bool(value) else 0.0
                    continue
                if isinstance(value, str):
                    if value.strip().lower() in self._TRUE_TOKENS:
                        out[i, j] = 1.0
                    continue
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(numeric):
                    out[i, j] = np.nan
                    continue
                out[i, j] = 1.0 if numeric != 0.0 else 0.0
        return out

    def get_feature_names_out(self, input_features: Any = None) -> Any:
        return input_features


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Encode high-cardinality categories by their training-set frequency.

    Frequencies are learned during fit. Categories unseen at transform time map
    to zero. This keeps high-cardinality encoding leak-safe and bounded in width.
    """

    def fit(self, X: pd.DataFrame, y: Any = None) -> FrequencyEncoder:
        frame = pd.DataFrame(X)
        self.maps_: dict[Any, dict[Any, float]] = {}
        n = len(frame)
        for col in frame.columns:
            counts = frame[col].astype("object").value_counts(dropna=True)
            self.maps_[col] = (counts / max(n, 1)).to_dict()
        self.columns_ = list(frame.columns)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        frame = pd.DataFrame(X)
        out = np.zeros((len(frame), len(self.columns_)), dtype=float)
        for i, col in enumerate(self.columns_):
            mapping = self.maps_.get(col, {})
            out[:, i] = frame[col].astype("object").map(mapping).fillna(0.0).to_numpy()
        return out

    def get_feature_names_out(self, input_features: Any = None) -> Any:
        if input_features is None:
            return np.asarray([f"freq_{c}" for c in self.columns_])
        return np.asarray([f"freq_{c}" for c in input_features])


@dataclass
class PreprocessPlan:
    """The columns assigned to each handling strategy."""

    numeric: list[str] = field(default_factory=list)
    boolean: list[str] = field(default_factory=list)
    categorical_low: list[str] = field(default_factory=list)
    categorical_high: list[str] = field(default_factory=list)
    datetime: list[str] = field(default_factory=list)
    text: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    outlier_strategy: str = "iqr"

    def as_dict(self) -> dict[str, Any]:
        return {
            "numeric": self.numeric,
            "boolean": self.boolean,
            "categorical_low": self.categorical_low,
            "categorical_high": self.categorical_high,
            "datetime": self.datetime,
            "text": self.text,
            "dropped": self.dropped,
            "outlier_strategy": self.outlier_strategy,
        }


def _numeric_pipeline(outlier_strategy: str) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("clip", OutlierClipper(strategy=outlier_strategy)),
            ("scale", StandardScaler()),
        ]
    )


def _boolean_pipeline() -> Pipeline:
    """Discrete pipeline for binary columns.

    BooleanToNumeric casts bool, string, and numeric forms to a 0/1 float
    array first, leaving missing entries as NaN. It runs before the imputer so
    a raw boolean-dtype column is converted to numbers before SimpleImputer,
    which rejects bool dtype, ever sees it. Mode imputation then fills the
    missing entries with the more common value. No outlier clipping and no
    scaling: those steps collapse skewed binary columns to a constant.
    """
    return Pipeline(
        [
            ("cast", BooleanToNumeric()),
            ("impute", SimpleImputer(strategy="most_frequent")),
        ]
    )


def _low_cardinality_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "encode",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=1),
            ),
        ]
    )


def _high_cardinality_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="__missing__")),
            ("encode", FrequencyEncoder()),
        ]
    )


def plan_preprocess(
    profile: DataProfile,
    target: str | None,
    constraints: dict[str, Any] | None,
    log: DecisionLog,
) -> PreprocessPlan:
    """Decide how to handle each column based on its profile.

    Args:
        profile: Dataset profile.
        target: Target column to exclude from features, or None.
        constraints: Optional knobs (missing_threshold, outlier_strategy).
        log: Decision log.

    Returns:
        A PreprocessPlan mapping columns to strategies.
    """
    constraints = constraints or {}
    drop_threshold = float(
        constraints.get("missing_threshold", DEFAULT_MISSING_DROP_THRESHOLD)
    )
    outlier_strategy = str(constraints.get("outlier_strategy", "iqr"))
    plan = PreprocessPlan(outlier_strategy=outlier_strategy)
    dropped_ids: list[str] = []

    for col in profile.columns.values():
        if target is not None and col.name == target:
            continue

        if col.missing_fraction > drop_threshold:
            plan.dropped.append(col.name)
            log.record(
                "preprocess",
                f"Dropped '{col.name}': {col.missing_fraction:.0%} missing "
                f"(above {drop_threshold:.0%}).",
                "missing-drop-threshold",
                {"missing_fraction": col.missing_fraction, "threshold": drop_threshold},
            )
            continue

        if col.inferred_type == ID:
            dropped_ids.append(col.name)
            plan.dropped.append(col.name)
            log.record(
                "preprocess",
                f"Dropped '{col.name}': inferred as an identifier with no signal.",
                "drop-id-column",
                {"unique_ratio": col.unique_ratio},
            )
            continue

        if col.inferred_type == BOOLEAN:
            plan.boolean.append(col.name)
            log.record(
                "preprocess",
                f"'{col.name}': mode imputation, pass through as 0/1. "
                f"Discrete column, so no outlier clipping and no scaling.",
                "boolean-discrete-handling",
                {"type": col.inferred_type, "n_unique": col.n_unique},
            )
        elif col.inferred_type == NUMERIC:
            plan.numeric.append(col.name)
            log.record(
                "preprocess",
                f"'{col.name}': median imputation, outlier clip ({outlier_strategy}), scale.",
                "numeric-handling",
                {"type": col.inferred_type},
            )
        elif col.inferred_type == DATETIME:
            plan.datetime.append(col.name)
            log.record(
                "preprocess",
                f"'{col.name}': parse datetime, extract year/month/day/weekday.",
                "datetime-part-extraction",
                {},
            )
        elif col.inferred_type == TEXT:
            plan.text.append(col.name)
            log.record(
                "preprocess",
                f"'{col.name}': replace text with length and word-count features.",
                "text-length-features",
                {},
            )
        elif col.inferred_type == CATEGORICAL:
            if col.n_unique <= HIGH_CARDINALITY_THRESHOLD:
                plan.categorical_low.append(col.name)
                log.record(
                    "preprocess",
                    f"'{col.name}': mode imputation, one-hot encoding "
                    f"({col.n_unique} categories).",
                    "low-cardinality-onehot",
                    {"n_unique": col.n_unique, "threshold": HIGH_CARDINALITY_THRESHOLD},
                )
            else:
                plan.categorical_high.append(col.name)
                log.record(
                    "preprocess",
                    f"'{col.name}': frequency encoding "
                    f"({col.n_unique} categories, above {HIGH_CARDINALITY_THRESHOLD}).",
                    "high-cardinality-frequency",
                    {"n_unique": col.n_unique, "threshold": HIGH_CARDINALITY_THRESHOLD},
                )

    _recover_if_empty(plan, profile, dropped_ids, log)
    return plan


def _recover_if_empty(
    plan: PreprocessPlan,
    profile: DataProfile,
    dropped_ids: list[str],
    log: DecisionLog,
) -> None:
    """Keep id columns as features when nothing else would remain.

    Dropping identifier columns is the right default, but if every feature
    column was dropped that way the model would have nothing to learn from.
    In that case the id columns are reinstated by their underlying dtype.
    """
    has_features = any(
        [
            plan.numeric,
            plan.boolean,
            plan.categorical_low,
            plan.categorical_high,
            plan.datetime,
            plan.text,
        ]
    )
    if has_features or not dropped_ids:
        return

    for name in dropped_ids:
        col = profile.column(name)
        plan.dropped.remove(name)
        if col.dtype.startswith(("int", "float", "uint")):
            plan.numeric.append(name)
        else:
            plan.categorical_high.append(name)
    log.record(
        "preprocess",
        f"Kept id columns {dropped_ids} as features: nothing else remained.",
        "recover-id-columns",
        {"columns": dropped_ids},
    )


def build_pipeline(
    profile: DataProfile,
    target: str | None = None,
    constraints: dict[str, Any] | None = None,
    log: DecisionLog | None = None,
) -> tuple[Pipeline, PreprocessPlan]:
    """Build a leakage-safe preprocessing Pipeline from a data profile.

    The returned pipeline expands datetime and text columns first, then routes
    the remaining columns through a ColumnTransformer. Fit it on training data
    only.

    Args:
        profile: Dataset profile.
        target: Target column to exclude, or None for unsupervised.
        constraints: Optional preprocessing knobs.
        log: Decision log.

    Returns:
        A tuple of (pipeline, plan).
    """
    log = log if log is not None else DecisionLog()
    plan = plan_preprocess(profile, target, constraints, log)

    transformers = []
    if plan.numeric:
        transformers.append(("numeric", _numeric_pipeline(plan.outlier_strategy), plan.numeric))
    if plan.boolean:
        transformers.append(("boolean", _boolean_pipeline(), plan.boolean))
    if plan.categorical_low:
        transformers.append(("categorical_low", _low_cardinality_pipeline(), plan.categorical_low))
    if plan.categorical_high:
        transformers.append(
            ("categorical_high", _high_cardinality_pipeline(), plan.categorical_high)
        )
    # Datetime and text columns are expanded into numeric columns up front, so
    # the column transformer treats the expansions as numeric.
    expanded_numeric: list[str] = []
    for col in plan.datetime:
        expanded_numeric += [f"{col}__year", f"{col}__month", f"{col}__day", f"{col}__weekday"]
    for col in plan.text:
        expanded_numeric += [f"{col}__char_len", f"{col}__word_count"]
    if expanded_numeric:
        transformers.append(
            ("expanded", _numeric_pipeline(plan.outlier_strategy), expanded_numeric)
        )

    column_transformer = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )

    steps = []
    if plan.datetime:
        steps.append(("datetime", DatetimeFeatures(plan.datetime)))
    if plan.text:
        steps.append(("text", TextLengthFeatures(plan.text)))
    steps.append(("columns", column_transformer))

    pipeline = Pipeline(steps)
    return pipeline, plan
