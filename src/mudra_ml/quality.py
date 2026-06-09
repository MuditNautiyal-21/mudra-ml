"""Data-quality checks and the limitations and next-steps summary.

These checks read the data profile and the raw frame, then raise structured
warnings. Each warning carries a short message, the rule that produced it,
and any detail useful in the report. Nothing here changes the pipeline. The
warnings are surfaced in the report so the reader can judge the result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .constants import (
    HIGH_CARDINALITY_THRESHOLD,
    IQR_MULTIPLIER,
)
from .decisions import DecisionLog
from .profile import (
    CATEGORICAL,
    ID,
    NUMERIC,
    DataProfile,
    DataProfiler,
)

SEVERITY_INFO = "info"
SEVERITY_WARN = "warn"
SEVERITY_CRITICAL = "critical"

# A dataset below this row count is treated as too small for the metrics to
# be considered reliable. Used to flag the small-sample warning.
SMALL_DATASET_WARNING_ROWS = 100

# A held-out test set below this is treated as indicative only.
SMALL_TEST_SET_ROWS = 50

# Duplicate row fraction above this triggers a warning.
DUPLICATE_ROW_FRACTION = 0.05

# Correlation magnitude with the target above this triggers a leakage suspect.
LEAKAGE_CORR_THRESHOLD = 0.99

# Numeric target unique-value ratio above this is too sparse for a histogram.
HIGH_IMBALANCE_RATIO = 10.0


@dataclass
class QualityWarning:
    """A structured data-quality warning."""

    code: str
    severity: str
    message: str
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "detail": self.detail,
        }


@dataclass
class QualityReport:
    """Container for the data-quality section."""

    warnings: list[QualityWarning] = field(default_factory=list)
    class_balance: dict[str, int] = field(default_factory=dict)
    missingness: list[dict[str, Any]] = field(default_factory=list)
    cardinality: list[dict[str, Any]] = field(default_factory=list)
    outliers: list[dict[str, Any]] = field(default_factory=list)
    leakage_suspects: list[dict[str, Any]] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)

    def add(self, warning: QualityWarning) -> None:
        self.warnings.append(warning)

    def has(self, code: str) -> bool:
        return any(w.code == code for w in self.warnings)

    def as_dict(self) -> dict[str, Any]:
        return {
            "warnings": [w.as_dict() for w in self.warnings],
            "class_balance": self.class_balance,
            "missingness": self.missingness,
            "cardinality": self.cardinality,
            "outliers": self.outliers,
            "leakage_suspects": self.leakage_suspects,
            "next_steps": self.next_steps,
        }


def _outlier_counts(series: pd.Series) -> int:
    """Count IQR-based outliers in a numeric series."""
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return 0
    # Cast to float so the quantile interpolation is safe: numpy cannot
    # subtract booleans, and pd.to_numeric leaves a bool column as bool.
    values = values.astype(float)
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return 0
    lo = q1 - IQR_MULTIPLIER * iqr
    hi = q3 + IQR_MULTIPLIER * iqr
    return int(((values < lo) | (values > hi)).sum())


def _coerce_numeric(series: pd.Series) -> pd.Series:
    """Try numeric coercion, then category code fallback, dropping NaN."""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() >= max(2, int(0.5 * len(series))):
        return numeric.dropna()
    try:
        codes = pd.Categorical(series).codes
        codes = pd.Series(codes, index=series.index)
        return codes[codes >= 0]
    except (TypeError, ValueError):
        return pd.Series([], dtype=float)


def _safe_correlation(a: pd.Series, b: pd.Series) -> float:
    """Pearson correlation that returns 0.0 when not computable."""
    combined = pd.concat([a, b], axis=1, join="inner").dropna()
    if len(combined) < 3:
        return 0.0
    first = combined.iloc[:, 0]
    second = combined.iloc[:, 1]
    if first.nunique() <= 1 or second.nunique() <= 1:
        return 0.0
    try:
        with np.errstate(invalid="ignore"):
            value = float(np.corrcoef(first.to_numpy(), second.to_numpy())[0, 1])
    except (TypeError, ValueError):
        return 0.0
    if np.isnan(value) or np.isinf(value):
        return 0.0
    return value


def _detect_leakage(
    frame: pd.DataFrame,
    target: str,
    profile: DataProfile,
) -> list[dict[str, Any]]:
    """Flag features that are suspiciously predictive of the target."""
    suspects: list[dict[str, Any]] = []
    if target not in frame.columns:
        return suspects
    y_series = frame[target]
    y_numeric = _coerce_numeric(y_series)
    if y_numeric.empty:
        return suspects

    for col in profile.columns.values():
        if col.name == target:
            continue
        if col.inferred_type == ID:
            continue
        feature = frame[col.name]
        if feature.equals(y_series):
            suspects.append(
                {
                    "feature": col.name,
                    "reason": "feature equals target",
                    "score": 1.0,
                }
            )
            continue
        feature_numeric = _coerce_numeric(feature)
        if feature_numeric.empty:
            continue
        corr = _safe_correlation(feature_numeric, y_numeric)
        if abs(corr) >= LEAKAGE_CORR_THRESHOLD:
            suspects.append(
                {
                    "feature": col.name,
                    "reason": f"correlation with target = {corr:.3f}",
                    "score": round(abs(corr), 4),
                }
            )
    return suspects


def _class_balance(target: pd.Series) -> dict[str, int]:
    counts = target.dropna().value_counts()
    return {str(label): int(count) for label, count in counts.items()}


def _missingness_table(profile: DataProfile) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {
            "column": col.name,
            "missing_count": col.missing_count,
            "missing_fraction": col.missing_fraction,
        }
        for col in profile.columns.values()
        if col.missing_count > 0
    ]

    def _fraction(row: dict[str, Any]) -> float:
        return float(row["missing_fraction"])

    rows.sort(key=_fraction, reverse=True)
    return rows


def _cardinality_table(profile: DataProfile) -> list[dict[str, Any]]:
    return [
        {
            "column": col.name,
            "n_unique": col.n_unique,
            "type": col.inferred_type,
        }
        for col in profile.columns.values()
    ]


def _outlier_table(
    frame: pd.DataFrame, profile: DataProfile
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for col in profile.columns.values():
        # Boolean columns have only two values and no meaningful outliers, and
        # a raw bool series breaks the quantile computation. Skip them.
        if col.inferred_type != NUMERIC:
            continue
        if col.name not in frame.columns:
            continue
        count = _outlier_counts(frame[col.name])
        if count > 0:
            rows.append({"column": col.name, "outlier_count": count})
    rows.sort(key=lambda row: -row["outlier_count"])
    return rows


def check_quality(
    frame: pd.DataFrame,
    profile: DataProfile,
    target: str | None,
    task: str,
    log: DecisionLog,
) -> QualityReport:
    """Run every data-quality check, raise warnings, record them in the log.

    Args:
        frame: Raw input frame (before train-test split).
        profile: Output of the DataProfiler.
        target: Target column for supervised runs, or None.
        task: classification, regression, or clustering.
        log: Decision log to receive each warning as a decision entry.

    Returns:
        A QualityReport that the renderer can consume directly.
    """
    report = QualityReport()
    n_rows = profile.n_rows

    if n_rows < SMALL_DATASET_WARNING_ROWS:
        report.add(
            QualityWarning(
                code="small-dataset",
                severity=SEVERITY_CRITICAL,
                message=(
                    f"Only {n_rows} rows. Metrics are indicative only. "
                    f"Collect more data before relying on this model."
                ),
                detail={"n_rows": n_rows, "threshold": SMALL_DATASET_WARNING_ROWS},
            )
        )
        log.record(
            "quality",
            f"Dataset has {n_rows} rows, below {SMALL_DATASET_WARNING_ROWS}. "
            "Metrics flagged as indicative only.",
            "small-dataset",
            {"n_rows": n_rows},
        )

    constant_cols = [
        col.name
        for col in profile.columns.values()
        if col.n_unique <= 1 and col.name != target
    ]
    for name in constant_cols:
        report.add(
            QualityWarning(
                code="constant-column",
                severity=SEVERITY_WARN,
                message=f"Column '{name}' is constant or empty. It carries no signal.",
                detail={"column": name},
            )
        )
        log.record(
            "quality",
            f"Column '{name}' is constant.",
            "constant-column",
            {"column": name},
        )

    if profile.duplicate_rows > 0:
        fraction = profile.duplicate_rows / max(n_rows, 1)
        if fraction >= DUPLICATE_ROW_FRACTION:
            report.add(
                QualityWarning(
                    code="duplicate-rows",
                    severity=SEVERITY_WARN,
                    message=(
                        f"{profile.duplicate_rows} duplicate rows "
                        f"({fraction:.0%}). Consider deduplicating before training."
                    ),
                    detail={"duplicate_rows": profile.duplicate_rows, "fraction": fraction},
                )
            )
            log.record(
                "quality",
                f"Found {profile.duplicate_rows} duplicate rows ({fraction:.0%}).",
                "duplicate-rows",
                {"duplicate_rows": profile.duplicate_rows, "fraction": round(fraction, 4)},
            )

    high_card = [
        (col.name, col.n_unique)
        for col in profile.columns.values()
        if col.inferred_type == CATEGORICAL
        and col.n_unique > HIGH_CARDINALITY_THRESHOLD
        and col.name != target
    ]
    for name, n_unique in high_card:
        report.add(
            QualityWarning(
                code="high-cardinality",
                severity=SEVERITY_INFO,
                message=(
                    f"Categorical column '{name}' has {n_unique} levels. "
                    f"Frequency encoding will be used."
                ),
                detail={"column": name, "n_unique": n_unique},
            )
        )
        log.record(
            "quality",
            f"High-cardinality categorical '{name}' ({n_unique} levels).",
            "high-cardinality",
            {"column": name, "n_unique": n_unique},
        )

    all_missing = [
        col.name for col in profile.columns.values() if col.missing_fraction >= 0.99
    ]
    for name in all_missing:
        report.add(
            QualityWarning(
                code="all-missing-column",
                severity=SEVERITY_WARN,
                message=f"Column '{name}' is almost entirely missing.",
                detail={"column": name},
            )
        )
        log.record(
            "quality",
            f"Column '{name}' is almost entirely missing.",
            "all-missing-column",
            {"column": name},
        )

    report.missingness = _missingness_table(profile)
    report.cardinality = _cardinality_table(profile)
    report.outliers = _outlier_table(frame, profile)

    if target is not None and target in frame.columns:
        target_missing = int(frame[target].isna().sum())
        if target_missing > 0:
            report.add(
                QualityWarning(
                    code="target-missing",
                    severity=SEVERITY_WARN,
                    message=(
                        f"Target '{target}' has {target_missing} missing values. "
                        f"Those rows are dropped before training."
                    ),
                    detail={"target": target, "missing": target_missing},
                )
            )
            log.record(
                "quality",
                f"Target '{target}' has {target_missing} missing values.",
                "target-missing",
                {"target": target, "missing": target_missing},
            )

        if task == "classification":
            balance = _class_balance(frame[target])
            report.class_balance = balance
            counts = sorted(balance.values())
            if counts:
                if len(counts) == 1:
                    report.add(
                        QualityWarning(
                            code="single-class",
                            severity=SEVERITY_CRITICAL,
                            message=(
                                "Target has only one class. A classifier cannot be "
                                "trained on this data."
                            ),
                            detail={"classes": list(balance.keys())},
                        )
                    )
                    log.record(
                        "quality",
                        "Target has only one class.",
                        "single-class",
                        {"classes": list(balance.keys())},
                    )
                else:
                    ratio = counts[-1] / max(counts[0], 1)
                    if ratio >= HIGH_IMBALANCE_RATIO:
                        report.add(
                            QualityWarning(
                                code="imbalanced-classes",
                                severity=SEVERITY_WARN,
                                message=(
                                    f"Class imbalance ratio {ratio:.1f}:1. "
                                    f"Plain accuracy will overstate quality."
                                ),
                                detail={"ratio": round(ratio, 2), "counts": balance},
                            )
                        )
                        log.record(
                            "quality",
                            f"Class imbalance ratio {ratio:.1f}:1.",
                            "imbalanced-classes",
                            {"ratio": round(ratio, 2)},
                        )

        report.leakage_suspects = _detect_leakage(frame, target, profile)
        for suspect in report.leakage_suspects:
            report.add(
                QualityWarning(
                    code="leakage-suspect",
                    severity=SEVERITY_CRITICAL,
                    message=(
                        f"Feature '{suspect['feature']}' is suspiciously "
                        f"predictive of the target ({suspect['reason']}). "
                        f"Check for leakage before trusting the metrics."
                    ),
                    detail=suspect,
                )
            )
            log.record(
                "quality",
                f"Leakage suspect: '{suspect['feature']}' ({suspect['reason']}).",
                "leakage-suspect",
                suspect,
            )

    report.next_steps = _next_steps(report, n_rows, target, task)
    return report


def _next_steps(
    report: QualityReport,
    n_rows: int,
    target: str | None,
    task: str,
) -> list[str]:
    """Concrete suggestions based on the warnings raised."""
    steps: list[str] = []
    if report.has("small-dataset"):
        steps.append(
            f"Row count is {n_rows}. Aim for at least a few hundred examples per class "
            f"before treating the metrics as reliable."
        )
    if report.has("single-class"):
        steps.append(
            "Collect rows for the missing class or reframe the problem. A classifier "
            "cannot be trained on one class."
        )
    if report.has("imbalanced-classes"):
        steps.append(
            "Report macro-averaged metrics and a per-class breakdown. Consider class "
            "weights or resampling if the minority class matters."
        )
    if report.has("leakage-suspect"):
        steps.append(
            "Inspect the leakage suspects. If a feature carries the target, drop it and "
            "rerun before drawing conclusions."
        )
    if report.has("duplicate-rows"):
        steps.append(
            "Deduplicate before splitting so the held-out set does not contain rows "
            "already seen during training."
        )
    if report.has("constant-column"):
        steps.append(
            "Drop the constant columns from the source data. They add noise to "
            "feature-importance views without contributing signal."
        )
    if report.has("high-cardinality"):
        steps.append(
            "For the high-cardinality features, consider domain-specific grouping "
            "(for example by region or tier) instead of relying only on frequency encoding."
        )
    if report.has("target-missing"):
        steps.append(
            "Investigate the rows with a missing target. Dropping them is a default, not "
            "always the right choice."
        )
    if not steps:
        steps.append(
            "No critical quality issues were detected. Validate the model on a fresh "
            "sample from the same source before deploying."
        )
    return steps


def assess_dataset(
    frame: pd.DataFrame,
    target: str | None,
    task: str,
    log: DecisionLog | None = None,
) -> tuple[DataProfile, QualityReport]:
    """Profile a frame and run quality checks against it.

    Convenience wrapper used by the stress-test battery and by ad-hoc callers.

    Args:
        frame: The input frame.
        target: Optional target column.
        task: classification, regression, or clustering.
        log: Optional decision log to record into.

    Returns:
        A tuple of (DataProfile, QualityReport).
    """
    log = log if log is not None else DecisionLog()
    profile = DataProfiler(log).profile(frame)
    quality = check_quality(frame, profile, target, task, log)
    return profile, quality
