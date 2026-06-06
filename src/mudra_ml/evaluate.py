"""Training, tuning, and task-appropriate evaluation.

This module owns metric production for the report. Every number it produces
carries a name the report can render. New in this release: baselines, CV
mean and standard deviation across folds, train-versus-test metrics,
per-class breakdowns, ROC and precision-recall data for binary tasks,
regression diagnostics, and permutation importance.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
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

# Below this held-out size the metrics are flagged as indicative only.
SMALL_TEST_SET_ROWS = 50

# Permutation importance settings. Capped on the large stress dataset for time.
PERM_REPEATS = 10
PERM_MAX_SAMPLES = 2000


@dataclass
class CandidateResult:
    """Cross-validation and held-out result for one candidate.

    The cv_score remains the headline cross-validation score so existing
    callers see no change. cv_mean and cv_std expose the per-fold variation
    so the report can show mean +/- std.
    """

    name: str
    cv_score: float
    test_metrics: dict[str, float]
    best_params: dict[str, Any] = field(default_factory=dict)
    estimator: Any = None
    train_metrics: dict[str, float] = field(default_factory=dict)
    cv_mean: float = 0.0
    cv_std: float = 0.0
    cv_scores: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "cv_score": self.cv_score,
            "cv_mean": self.cv_mean,
            "cv_std": self.cv_std,
            "cv_scores": self.cv_scores,
            "test_metrics": self.test_metrics,
            "train_metrics": self.train_metrics,
            "best_params": self.best_params,
        }


@dataclass
class EvaluationResult:
    """Outcome of evaluating the shortlist.

    The optional fields hold diagnostic data the renderer uses for charts and
    the trust section: baseline metrics, ROC and PR curves, permutation
    importance, residuals.
    """

    task: str
    metric: str
    candidates: list[CandidateResult]
    best_name: str
    best_estimator: Any
    feature_importance: dict[str, float] = field(default_factory=dict)
    permutation_importance: dict[str, float] = field(default_factory=dict)
    permutation_importance_std: dict[str, float] = field(default_factory=dict)
    baseline_metrics: dict[str, float] = field(default_factory=dict)
    baseline_name: str = ""
    per_class_report: dict[str, dict[str, float]] = field(default_factory=dict)
    roc_curve: dict[str, Any] = field(default_factory=dict)
    pr_curve: dict[str, Any] = field(default_factory=dict)
    regression_diagnostics: dict[str, Any] = field(default_factory=dict)
    small_sample_warning: bool = False
    test_set_size: int = 0
    train_set_size: int = 0
    class_labels: list[Any] = field(default_factory=list)
    target_values: list[Any] = field(default_factory=list)

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
            "permutation_importance": self.permutation_importance,
            "permutation_importance_std": self.permutation_importance_std,
            "baseline_metrics": self.baseline_metrics,
            "baseline_name": self.baseline_name,
            "per_class_report": self.per_class_report,
            "roc_curve": self.roc_curve,
            "pr_curve": self.pr_curve,
            "regression_diagnostics": self.regression_diagnostics,
            "small_sample_warning": self.small_sample_warning,
            "test_set_size": self.test_set_size,
            "train_set_size": self.train_set_size,
            "class_labels": [str(label) for label in self.class_labels],
            "target_values": list(self.target_values),
        }


def _classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, estimator: Any, X: Any
) -> dict[str, float]:
    average = "binary" if len(np.unique(y_true)) == 2 else "weighted"
    try:
        binary_f1 = float(f1_score(y_true, y_pred, average=average, zero_division=0))
    except ValueError:
        # Falls back when the binary case has only one class in y_true.
        binary_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": binary_f1,
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


def _cv_value_for_selection(cv_mean: float, metric: str) -> float:
    """Higher is better. For metrics where lower is better the value is negated.

    Cross-validation scores returned by _per_fold_scores are already in
    positive units (negative sklearn scorings are re-flipped), so the only
    direction adjustment needed is for lower-is-better metrics.
    """
    return cv_mean if metric in _HIGHER_IS_BETTER else -cv_mean


def _select_best_by_cv(
    results: list[CandidateResult], metric: str
) -> CandidateResult:
    """Pick the candidate with the best cross-validation score for the metric.

    Selection uses only cross-validation. Test-set metrics are never used to
    rank candidates: the held-out set is scored once, for the selected model,
    for reporting only.
    """
    if not results:
        raise ValueError("No candidates to rank by cross-validation.")
    return max(results, key=lambda r: _cv_value_for_selection(r.cv_mean, metric))


def _per_fold_scores(
    estimator: Any, X: np.ndarray, y: np.ndarray, scoring: str, cv: int
) -> list[float]:
    """Cross-validated scores per fold. Negated scores are flipped to positive."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = cross_val_score(estimator, X, y, scoring=scoring, cv=cv)
    arr = np.asarray(scores, dtype=float)
    if scoring.startswith("neg_"):
        arr = -arr
    return [float(v) for v in arr]


def _tune_one(
    candidate: Candidate,
    X: np.ndarray,
    y: np.ndarray,
    scoring: str,
    cv: int,
    random_state: int,
    log: DecisionLog,
) -> tuple[Any, dict[str, Any], float, list[float]]:
    """Fit one candidate. Return estimator, best_params, cv mean, per-fold scores."""
    grid = candidate.param_grid
    if not grid:
        scores = _per_fold_scores(candidate.estimator, X, y, scoring, cv)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            candidate.estimator.fit(X, y)
        cv_score = float(np.mean(scores)) if scores else 0.0
        log.record(
            "evaluate",
            f"{candidate.name}: no grid, cross-validated as-is "
            f"(cv {scoring}={cv_score:.4f} +/- {float(np.std(scores)) if scores else 0.0:.4f}).",
            "cv-no-tuning",
            {"cv_score": round(cv_score, 4)},
        )
        return candidate.estimator, {}, cv_score, scores

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
    if scoring.startswith("neg_"):
        cv_score = -cv_score
    fold_scores = _per_fold_scores(search.best_estimator_, X, y, scoring, cv)

    log.record(
        "evaluate",
        f"{candidate.name}: tuned over {n_iter} configs "
        f"(cv {scoring}={cv_score:.4f}).",
        "randomized-search-cv",
        {"best_params": search.best_params_, "cv_score": round(cv_score, 4), "n_iter": n_iter},
    )
    return search.best_estimator_, search.best_params_, cv_score, fold_scores


def _feature_importance(estimator: Any, feature_names: list[str]) -> dict[str, float]:
    """Pull the impurity or coefficient importance from the fitted estimator."""
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


def _permutation_importance(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    task: str,
    random_state: int,
    log: DecisionLog,
) -> tuple[dict[str, float], dict[str, float]]:
    """Permutation importance with the standard deviation across repeats."""
    if X.shape[0] == 0 or X.shape[1] == 0:
        return {}, {}
    n_samples = X.shape[0]
    indices: np.ndarray | None = None
    if n_samples > PERM_MAX_SAMPLES:
        rng = np.random.default_rng(random_state)
        indices = rng.choice(n_samples, size=PERM_MAX_SAMPLES, replace=False)
        X_eval = X[indices]
        y_eval = y[indices]
        log.record(
            "evaluate",
            f"Permutation importance computed on a {PERM_MAX_SAMPLES}-row subsample "
            "to keep runtime in check.",
            "permutation-subsample",
            {"n_samples": PERM_MAX_SAMPLES},
        )
    else:
        X_eval = X
        y_eval = y

    scoring = "accuracy" if task == "classification" else "r2"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = permutation_importance(
                estimator,
                X_eval,
                y_eval,
                n_repeats=PERM_REPEATS,
                random_state=random_state,
                scoring=scoring,
                n_jobs=1,
            )
    except (ValueError, TypeError) as exc:
        log.record(
            "evaluate",
            f"Permutation importance skipped: {exc}.",
            "permutation-skip",
            {"reason": str(exc)},
        )
        return {}, {}

    means = np.asarray(result.importances_mean, dtype=float)
    stds = np.asarray(result.importances_std, dtype=float)
    if len(means) != len(feature_names):
        feature_names = [f"feature_{i}" for i in range(len(means))]

    paired = sorted(
        zip(feature_names, means, stds, strict=False),
        key=lambda triple: -triple[1],
    )
    top = paired[:20]
    importance = {name: float(value) for name, value, _std in top}
    importance_std = {name: float(std) for name, _value, std in top}
    log.record(
        "evaluate",
        f"Permutation importance computed (n_repeats={PERM_REPEATS}). "
        "Impurity importance is biased toward high-cardinality features; "
        "the permutation view is more reliable.",
        "permutation-importance",
        {"n_repeats": PERM_REPEATS, "scoring": scoring},
    )
    return importance, importance_std


def _baseline(
    task: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    random_state: int,
    log: DecisionLog,
) -> tuple[str, dict[str, float]]:
    """Score a naive baseline so the headline metric has a reference."""
    if task == "classification":
        estimator = DummyClassifier(strategy="most_frequent", random_state=random_state)
        name = "dummy_most_frequent"
    else:
        estimator = DummyRegressor(strategy="mean")
        name = "dummy_mean"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)
    except (ValueError, TypeError) as exc:
        log.record(
            "evaluate",
            f"Baseline ({name}) skipped: {exc}.",
            "baseline-skip",
            {"reason": str(exc)},
        )
        return name, {}
    if task == "classification":
        metrics = _classification_metrics(y_test, y_pred, estimator, X_test)
    else:
        metrics = _regression_metrics(y_test, y_pred)
    log.record(
        "evaluate",
        f"Baseline {name} scored for reference (rule: naive-baseline).",
        "naive-baseline",
        {"name": name},
    )
    return name, metrics


def _per_class_report(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, dict[str, float]]:
    """Per-class precision, recall, f1, and support."""
    try:
        report = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )
    except ValueError:
        return {}
    out: dict[str, dict[str, float]] = {}
    for label, values in report.items():
        if not isinstance(values, dict):
            continue
        out[str(label)] = {
            "precision": float(values.get("precision", 0.0)),
            "recall": float(values.get("recall", 0.0)),
            "f1": float(values.get("f1-score", 0.0)),
            "support": float(values.get("support", 0.0)),
        }
    return out


def _roc_pr_curves(
    estimator: Any, X_test: np.ndarray, y_test: np.ndarray
) -> tuple[dict[str, Any], dict[str, Any]]:
    """ROC and PR curve points for binary classifiers with predict_proba."""
    labels = np.unique(y_test)
    if len(labels) != 2 or not hasattr(estimator, "predict_proba"):
        return {}, {}
    try:
        proba = estimator.predict_proba(X_test)
        if proba.shape[1] != 2:
            return {}, {}
        positive = proba[:, 1]
        fpr, tpr, _ = roc_curve(y_test, positive)
        auc_value = float(roc_auc_score(y_test, positive))
        precision, recall, _ = precision_recall_curve(y_test, positive)
        ap = float(average_precision_score(y_test, positive))
    except (ValueError, AttributeError):
        return {}, {}
    return (
        {"fpr": [float(v) for v in fpr], "tpr": [float(v) for v in tpr], "auc": auc_value},
        {
            "recall": [float(v) for v in recall],
            "precision": [float(v) for v in precision],
            "average_precision": ap,
        },
    )


def _regression_diagnostics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, Any]:
    """Residual stats and arrays for the regression diagnostic plots."""
    if len(y_true) == 0:
        return {}
    residuals = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return {
        "y_true": [float(v) for v in y_true],
        "y_pred": [float(v) for v in y_pred],
        "residual_mean": float(np.mean(residuals)),
        "residual_std": float(np.std(residuals)),
        "residual_abs_mean": float(np.mean(np.abs(residuals))),
        "residual_max": float(np.max(np.abs(residuals))) if residuals.size else 0.0,
    }


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
        An EvaluationResult with the selected model and the diagnostic block.
    """
    log = log if log is not None else DecisionLog()
    scoring = _SKLEARN_SCORING.get(metric, "accuracy" if task == "classification" else "r2")
    effective_cv = max(2, min(cv, _min_class_count(y_train, task)))

    # Phase 1: tune and cross-validate every candidate. No test-set scoring
    # happens here. The test set must not influence selection.
    results: list[CandidateResult] = []
    for candidate in candidates:
        estimator, best_params, cv_score, fold_scores = _tune_one(
            candidate, X_train, y_train, scoring, effective_cv, random_state, log
        )
        cv_mean = float(np.mean(fold_scores)) if fold_scores else cv_score
        cv_std = float(np.std(fold_scores)) if fold_scores else 0.0
        results.append(
            CandidateResult(
                name=candidate.name,
                cv_score=round(cv_score, 6),
                test_metrics={},
                train_metrics={},
                best_params=best_params,
                estimator=estimator,
                cv_mean=round(cv_mean, 6),
                cv_std=round(cv_std, 6),
                cv_scores=[round(v, 6) for v in fold_scores],
            )
        )

    # Phase 2: select by cross-validation only.
    best = _select_best_by_cv(results, metric)
    log.record(
        "evaluate",
        f"Best model: {best.name} (cv {metric}={best.cv_mean:.4f} "
        f"+/- {best.cv_std:.4f}). Selection uses cross-validation only; "
        f"the held-out test set is scored once, for this model, for "
        f"reporting only.",
        "best-model-by-cv",
        {"metric": metric, "cv_mean": best.cv_mean, "cv_std": best.cv_std},
    )

    # Phase 3: score the held-out test set once, only for the selected model.
    test_pred = best.estimator.predict(X_test)
    if task == "classification":
        test_metrics = _classification_metrics(y_test, test_pred, best.estimator, X_test)
        train_metrics = _classification_metrics(
            y_train, best.estimator.predict(X_train), best.estimator, X_train
        )
    else:
        test_metrics = _regression_metrics(y_test, test_pred)
        train_metrics = _regression_metrics(y_train, best.estimator.predict(X_train))
    best.test_metrics = test_metrics
    best.train_metrics = {
        k: v for k, v in train_metrics.items() if k != "confusion_matrix"
    }

    # Rank the candidates list so consumers (the report table) see the best
    # CV scorer first. Test metrics stay empty for the non-selected models.
    results.sort(key=lambda r: _cv_value_for_selection(r.cv_mean, metric), reverse=True)

    importance = _feature_importance(best.estimator, feature_names)
    perm_importance, perm_std = _permutation_importance(
        best.estimator, X_test, y_test, feature_names, task, random_state, log
    )

    baseline_name, baseline_metrics = _baseline(
        task, X_train, y_train, X_test, y_test, random_state, log
    )
    if baseline_metrics and metric in baseline_metrics:
        gap = float(best.test_metrics.get(metric, 0.0)) - float(baseline_metrics[metric])
        log.record(
            "evaluate",
            f"Best vs baseline gap on {metric}: {gap:+.4f}.",
            "baseline-comparison",
            {"metric": metric, "gap": round(gap, 4)},
        )

    test_size = int(len(y_test))
    train_size = int(len(y_train))
    small_sample = test_size < SMALL_TEST_SET_ROWS
    if small_sample:
        log.record(
            "evaluate",
            f"Held-out test set has {test_size} rows. Metrics are indicative only.",
            "small-test-set",
            {"test_size": test_size, "threshold": SMALL_TEST_SET_ROWS},
        )

    if task == "classification":
        per_class = _per_class_report(y_test, best.estimator.predict(X_test))
        roc, pr = _roc_pr_curves(best.estimator, X_test, y_test)
        diagnostics: dict[str, Any] = {}
        class_labels: list[Any] = list(np.unique(y_test).tolist())
        target_values: list[Any] = [str(v) for v in y_test.tolist()]
    else:
        per_class = {}
        roc, pr = ({}, {})
        diagnostics = _regression_diagnostics(y_test, best.estimator.predict(X_test))
        class_labels = []
        target_values = [float(v) for v in y_test.tolist()]

    return EvaluationResult(
        task=task,
        metric=metric,
        candidates=results,
        best_name=best.name,
        best_estimator=best.estimator,
        feature_importance=importance,
        permutation_importance=perm_importance,
        permutation_importance_std=perm_std,
        baseline_metrics={k: v for k, v in baseline_metrics.items() if k != "confusion_matrix"},
        baseline_name=baseline_name,
        per_class_report=per_class,
        roc_curve=roc,
        pr_curve=pr,
        regression_diagnostics=diagnostics,
        small_sample_warning=small_sample,
        test_set_size=test_size,
        train_set_size=train_size,
        class_labels=class_labels,
        target_values=target_values,
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
    log = log if log is not None else DecisionLog()
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
                name=name,
                cv_score=sil,
                cv_mean=sil,
                cv_std=0.0,
                cv_scores=[sil],
                test_metrics=metrics,
                best_params={"n_clusters": k},
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
        train_set_size=int(len(X)),
        test_set_size=0,
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
