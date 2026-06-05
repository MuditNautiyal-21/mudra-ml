"""Training, tuning, and task-appropriate evaluation."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import RandomizedSearchCV, cross_val_score

from .constants import DEFAULT_CV_FOLDS, DEFAULT_SEARCH_ITER
from .decisions import DecisionLog
from .recommend import Candidate

# Metrics where a larger value is better.
_HIGHER_IS_BETTER = {
    "accuracy",
    "f1",
    "f1_macro",
    "roc_auc",
    "precision",
    "recall",
    "r2",
    "silhouette",
}
_LOWER_IS_BETTER = {"rmse", "mae", "mse", "davies_bouldin"}

_SKLEARN_SCORING = {
    "accuracy": "accuracy",
    "f1": "f1_weighted",
    "f1_macro": "f1_macro",
    "roc_auc": "roc_auc_ovr_weighted",
    "precision": "precision_weighted",
    "recall": "recall_weighted",
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "mse": "neg_mean_squared_error",
    "r2": "r2",
}


@dataclass
class CandidateResult:
    """Cross-validation and held-out result for one candidate."""

    name: str
    cv_score: float
    test_metrics: dict[str, float]
    best_params: dict[str, Any] = field(default_factory=dict)
    estimator: Any = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "cv_score": self.cv_score,
            "test_metrics": self.test_metrics,
            "best_params": self.best_params,
        }


@dataclass
class EvaluationResult:
    """Outcome of evaluating the shortlist."""

    task: str
    metric: str
    candidates: list[CandidateResult]
    best_name: str
    best_estimator: Any
    feature_importance: dict[str, float] = field(default_factory=dict)

    @property
    def best(self) -> CandidateResult:
        return next(c for c in self.candidates if c.name == self.best_name)

    def as_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "metric": self.metric,
            "best_name": self.best_name,
            "candidates": [c.as_dict() for c in self.candidates],
            "feature_importance": self.feature_importance,
        }


def _classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, estimator: Any, X: Any
) -> dict[str, float]:
    average = "binary" if len(np.unique(y_true)) == 2 else "weighted"
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
    }
    metrics["confusion_matrix"] = _confusion_matrix(y_true, y_pred)
    try:
        if hasattr(estimator, "predict_proba"):
            proba = estimator.predict_proba(X)
            if proba.shape[1] == 2:
                metrics["roc_auc"] = float(roc_auc_score(y_true, proba[:, 1]))
            else:
                metrics["roc_auc"] = float(
                    roc_auc_score(y_true, proba, multi_class="ovr", average="weighted")
                )
    except (ValueError, AttributeError):
        pass
    return metrics


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> list[list[int]]:
    from sklearn.metrics import confusion_matrix

    return confusion_matrix(y_true, y_pred).tolist()


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": mse,
        "r2": float(r2_score(y_true, y_pred)),
    }


def _score_value(metrics: dict[str, float], metric: str) -> float:
    value = metrics.get(metric)
    if value is None:
        return float("-inf")
    return value if metric in _HIGHER_IS_BETTER else -value


def _tune_one(
    candidate: Candidate,
    X: np.ndarray,
    y: np.ndarray,
    scoring: str,
    cv: int,
    random_state: int,
    log: DecisionLog,
) -> tuple[Any, dict[str, Any], float]:
    """Fit one candidate, tuning with RandomizedSearchCV when it has a grid."""
    grid = candidate.param_grid
    if not grid:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_val_score(candidate.estimator, X, y, scoring=scoring, cv=cv)
            candidate.estimator.fit(X, y)
        cv_score = float(np.mean(scores))
        log.record(
            "evaluate",
            f"{candidate.name}: no grid, cross-validated as-is (cv {scoring}={cv_score:.4f}).",
            "cv-no-tuning",
            {"cv_score": round(cv_score, 4)},
        )
        return candidate.estimator, {}, cv_score

    n_combos = 1
    for values in grid.values():
        n_combos *= len(values)
    n_iter = min(DEFAULT_SEARCH_ITER, n_combos)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search = RandomizedSearchCV(
            candidate.estimator,
            grid,
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            random_state=random_state,
            n_jobs=-1,
        )
        search.fit(X, y)

    cv_score = float(search.best_score_)
    log.record(
        "evaluate",
        f"{candidate.name}: tuned over {n_iter} configs "
        f"(cv {scoring}={cv_score:.4f}).",
        "randomized-search-cv",
        {"best_params": search.best_params_, "cv_score": round(cv_score, 4), "n_iter": n_iter},
    )
    return search.best_estimator_, search.best_params_, cv_score


def _feature_importance(estimator: Any, feature_names: list[str]) -> dict[str, float]:
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_, dtype=float)
        values = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
    else:
        return {}
    if len(values) != len(feature_names):
        feature_names = [f"feature_{i}" for i in range(len(values))]
    pairs = sorted(zip(feature_names, values, strict=False), key=lambda p: -p[1])
    return {name: float(score) for name, score in pairs[:20]}


def evaluate_supervised(
    candidates: list[Candidate],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task: str,
    metric: str,
    feature_names: list[str],
    random_state: int = 42,
    cv: int = DEFAULT_CV_FOLDS,
    log: DecisionLog | None = None,
) -> EvaluationResult:
    """Tune and evaluate supervised candidates, then select the best.

    Args:
        candidates: Shortlist to evaluate.
        X_train: Transformed training features.
        y_train: Training target.
        X_test: Transformed test features.
        y_test: Test target.
        task: classification or regression.
        metric: Metric used for selection.
        feature_names: Names of the transformed features.
        random_state: Seed.
        cv: Number of cross-validation folds.
        log: Decision log.

    Returns:
        An EvaluationResult with the selected model.
    """
    log = log or DecisionLog()
    scoring = _SKLEARN_SCORING.get(metric, "accuracy" if task == "classification" else "r2")
    effective_cv = max(2, min(cv, _min_class_count(y_train, task)))

    results: list[CandidateResult] = []
    for candidate in candidates:
        estimator, best_params, cv_score = _tune_one(
            candidate, X_train, y_train, scoring, effective_cv, random_state, log
        )
        y_pred = estimator.predict(X_test)
        if task == "classification":
            test_metrics = _classification_metrics(y_test, y_pred, estimator, X_test)
        else:
            test_metrics = _regression_metrics(y_test, y_pred)
        results.append(
            CandidateResult(
                name=candidate.name,
                cv_score=round(cv_score, 6),
                test_metrics=test_metrics,
                best_params=best_params,
                estimator=estimator,
            )
        )

    best = max(results, key=lambda r: _score_value(r.test_metrics, metric))
    log.record(
        "evaluate",
        f"Best model: {best.name} ({metric}={best.test_metrics.get(metric):.4f} on held-out test).",
        "best-model-selection",
        {"metric": metric, "value": round(best.test_metrics.get(metric, 0.0), 4)},
    )

    importance = _feature_importance(best.estimator, feature_names)
    return EvaluationResult(
        task=task,
        metric=metric,
        candidates=results,
        best_name=best.name,
        best_estimator=best.estimator,
        feature_importance=importance,
    )


def _min_class_count(y: np.ndarray, task: str) -> int:
    if task != "classification":
        return DEFAULT_CV_FOLDS
    _, counts = np.unique(y, return_counts=True)
    return int(counts.min())


def evaluate_clustering(
    candidate: Candidate,
    X: np.ndarray,
    metric: str,
    random_state: int = 42,
    log: DecisionLog | None = None,
) -> EvaluationResult:
    """Sweep cluster counts and select the best by an internal index.

    Args:
        candidate: The clustering candidate (KMeans) with an n_clusters grid.
        X: Transformed features.
        metric: silhouette or davies_bouldin.
        random_state: Seed.
        log: Decision log.

    Returns:
        An EvaluationResult with the fitted clusterer.
    """
    log = log or DecisionLog()
    grid = candidate.param_grid.get("n_clusters", [2, 3, 4, 5])
    results: list[CandidateResult] = []
    best_estimator = None
    best_value = float("-inf")
    best_name = ""

    for k in grid:
        if k >= len(X):
            continue
        estimator = candidate.estimator.set_params(n_clusters=k)
        labels = estimator.fit_predict(X)
        if len(np.unique(labels)) < 2:
            continue
        sil = float(silhouette_score(X, labels))
        db = float(davies_bouldin_score(X, labels))
        metrics = {"silhouette": sil, "davies_bouldin": db, "n_clusters": float(k)}
        value = _score_value(metrics, metric)
        name = f"kmeans_k{k}"
        results.append(
            CandidateResult(
                name=name, cv_score=sil, test_metrics=metrics, best_params={"n_clusters": k}
            )
        )
        log.record(
            "evaluate",
            f"KMeans k={k}: silhouette={sil:.4f}, davies_bouldin={db:.4f}.",
            "clustering-sweep",
            {"k": k, "silhouette": round(sil, 4)},
        )
        if value > best_value:
            best_value = value
            best_estimator = estimator
            best_name = name

    if best_estimator is None:
        raise ValueError("Clustering produced no valid partition. Check the data.")

    # Refit the chosen estimator so it is the returned model.
    best_k = int(best_name.split("k")[-1])
    best_estimator = candidate.estimator.set_params(n_clusters=best_k)
    best_estimator.fit(X)
    for result in results:
        if result.name == best_name:
            result.estimator = best_estimator

    log.record(
        "evaluate",
        f"Selected {best_name} by {metric}.",
        "clustering-selection",
        {"metric": metric, "k": best_k},
    )
    return EvaluationResult(
        task="clustering",
        metric=metric,
        candidates=results,
        best_name=best_name,
        best_estimator=best_estimator,
    )


def evaluate(
    candidates: list[Candidate],
    task: str,
    metric: str,
    feature_names: list[str],
    X_train: np.ndarray,
    y_train: np.ndarray | None = None,
    X_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    random_state: int = 42,
    cv: int = DEFAULT_CV_FOLDS,
    log: DecisionLog | None = None,
) -> EvaluationResult:
    """Dispatch to supervised or clustering evaluation based on the task."""
    if task == "clustering":
        return evaluate_clustering(candidates[0], X_train, metric, random_state, log)
    if y_train is None or X_test is None or y_test is None:
        raise ValueError("Supervised evaluation needs y_train, X_test, and y_test.")
    return evaluate_supervised(
        candidates,
        X_train,
        np.asarray(y_train),
        X_test,
        np.asarray(y_test),
        task,
        metric,
        feature_names,
        random_state,
        cv,
        log,
    )
