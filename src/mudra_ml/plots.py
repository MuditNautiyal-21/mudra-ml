"""Chart rendering for the HTML report.

The Agg backend is forced at import so rendering works headless with no
display. Each chart function returns a base64-encoded PNG suitable for an
inline `<img>` tag, or None if the chart could not be produced for the
given inputs. Callers should treat None as a skip, not an error.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

logger = logging.getLogger("mudra_ml")

_FIGSIZE_SMALL = (5.5, 4.0)
_FIGSIZE_WIDE = (7.0, 4.0)


def _encode(fig: plt.Figure) -> str:
    """Render a figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _safe(name: str):
    """Decorator that turns chart failures into a logged skip returning None."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                logger.info("Skipped %s chart: %s", name, exc)
                plt.close("all")
                return None

        return wrapper

    return decorator


@_safe("confusion-matrix")
def confusion_matrix_chart(matrix: list[list[int]], labels: list[Any]) -> str | None:
    """Heatmap of the confusion matrix with cell counts annotated."""
    array = np.asarray(matrix, dtype=int)
    if array.size == 0 or array.ndim != 2:
        return None
    fig, ax = plt.subplots(figsize=_FIGSIZE_SMALL)
    im = ax.imshow(array, cmap="Blues", aspect="auto")
    ax.set_xticks(range(array.shape[1]))
    ax.set_yticks(range(array.shape[0]))
    ax.set_xticklabels([str(label) for label in labels])
    ax.set_yticklabels([str(label) for label in labels])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix")
    threshold = array.max() / 2.0 if array.max() > 0 else 0.5
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            color = "white" if array[i, j] > threshold else "black"
            ax.text(j, i, str(array[i, j]), ha="center", va="center", color=color)
    fig.colorbar(im, ax=ax)
    return _encode(fig)


@_safe("roc-curve")
def roc_curve_chart(fpr: list[float], tpr: list[float], auc: float) -> str | None:
    """ROC curve for a binary classifier."""
    if not fpr or not tpr:
        return None
    fig, ax = plt.subplots(figsize=_FIGSIZE_SMALL)
    ax.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], color="#999", linestyle="--", lw=1)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right")
    return _encode(fig)


@_safe("pr-curve")
def precision_recall_chart(
    recall: list[float], precision: list[float], average_precision: float
) -> str | None:
    """Precision-recall curve for a binary classifier."""
    if not recall or not precision:
        return None
    fig, ax = plt.subplots(figsize=_FIGSIZE_SMALL)
    ax.plot(recall, precision, color="#2ca02c", lw=2, label=f"AP = {average_precision:.3f}")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curve")
    ax.legend(loc="lower left")
    return _encode(fig)


@_safe("feature-importance")
def feature_importance_chart(
    importances: dict[str, float],
    stds: dict[str, float] | None = None,
    top_k: int = 15,
    title: str = "Feature importance",
) -> str | None:
    """Horizontal bar chart of the top features by importance."""
    if not importances:
        return None
    items = list(importances.items())[:top_k]
    names = [name for name, _ in items][::-1]
    values = [value for _, value in items][::-1]
    errors = None
    if stds:
        errors = [stds.get(name, 0.0) for name in names]
    fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE)
    ax.barh(names, values, xerr=errors, color="#4c78a8", ecolor="#333")
    ax.set_xlabel("Importance")
    ax.set_title(title)
    fig.tight_layout()
    return _encode(fig)


@_safe("target-distribution")
def target_distribution_chart(
    values: list[Any],
    task: str,
    title: str = "Target distribution",
) -> str | None:
    """Class counts for classification, histogram for regression."""
    if not values:
        return None
    fig, ax = plt.subplots(figsize=_FIGSIZE_SMALL)
    if task == "classification":
        unique, counts = np.unique(np.asarray(values), return_counts=True)
        ax.bar([str(u) for u in unique], counts, color="#4c78a8")
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
    else:
        ax.hist(np.asarray(values, dtype=float), bins=30, color="#4c78a8")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
    ax.set_title(title)
    fig.tight_layout()
    return _encode(fig)


@_safe("correlation-heatmap")
def correlation_heatmap(matrix: list[list[float]], columns: list[str]) -> str | None:
    """Pearson correlation heatmap for the supplied numeric columns."""
    array = np.asarray(matrix, dtype=float)
    if array.size == 0 or array.shape[0] != array.shape[1]:
        return None
    n = array.shape[0]
    fig_w = max(4.5, min(10.0, 0.45 * n + 3.0))
    fig, ax = plt.subplots(figsize=(fig_w, fig_w * 0.9))
    im = ax.imshow(array, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(columns, rotation=45, ha="right")
    ax.set_yticklabels(columns)
    ax.set_title("Feature correlation")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return _encode(fig)


@_safe("residual")
def residual_chart(y_true: list[float], y_pred: list[float]) -> str | None:
    """Residual vs predicted scatter for a regression model."""
    if not y_true or not y_pred:
        return None
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    residuals = actual - predicted
    fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE)
    ax.scatter(predicted, residuals, alpha=0.6, color="#4c78a8", edgecolor="none")
    ax.axhline(0.0, color="#999", linestyle="--", lw=1)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (actual minus predicted)")
    ax.set_title("Residuals")
    fig.tight_layout()
    return _encode(fig)


@_safe("predicted-vs-actual")
def predicted_vs_actual_chart(y_true: list[float], y_pred: list[float]) -> str | None:
    """Scatter of predicted against actual values with the y=x reference."""
    if not y_true or not y_pred:
        return None
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    fig, ax = plt.subplots(figsize=_FIGSIZE_SMALL)
    ax.scatter(actual, predicted, alpha=0.6, color="#4c78a8", edgecolor="none")
    lo = float(min(actual.min(), predicted.min()))
    hi = float(max(actual.max(), predicted.max()))
    ax.plot([lo, hi], [lo, hi], color="#999", linestyle="--", lw=1)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Predicted vs actual")
    fig.tight_layout()
    return _encode(fig)


def render_all(spec: dict[str, Any]) -> dict[str, str]:
    """Render every chart described in the spec, dropping any that fail.

    Args:
        spec: A dict mapping a chart key to its inputs. Each entry is a
            tuple of (kind, kwargs).

    Returns:
        Dict of chart key to base64 PNG string. Failed charts are omitted.
    """
    handlers = {
        "confusion_matrix": confusion_matrix_chart,
        "roc": roc_curve_chart,
        "pr": precision_recall_chart,
        "feature_importance": feature_importance_chart,
        "target_distribution": target_distribution_chart,
        "correlation": correlation_heatmap,
        "residual": residual_chart,
        "predicted_vs_actual": predicted_vs_actual_chart,
    }
    out: dict[str, str] = {}
    for key, payload in spec.items():
        kind = payload.get("kind", key)
        kwargs = payload.get("kwargs", {})
        handler = handlers.get(kind)
        if handler is None:
            continue
        encoded = handler(**kwargs)
        if encoded:
            out[key] = encoded
    return out
