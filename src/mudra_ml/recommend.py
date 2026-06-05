"""Rule-based algorithm recommendation.

The shortlist is produced by a documented rule set keyed on task type, dataset
size, feature count, cardinality, and the operator's interpretability and speed
preferences. No model is trained to make this choice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sklearn.cluster import KMeans
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .constants import LARGE_DATASET_ROWS, SMALL_DATASET_ROWS
from .decisions import DecisionLog


@dataclass
class Candidate:
    """A model to train, with the curated search grid for its tuning."""

    name: str
    estimator: Any
    param_grid: dict[str, list[Any]] = field(default_factory=dict)
    interpretable: bool = False
    reason: str = ""


def _classification_candidates(random_state: int) -> dict[str, Candidate]:
    return {
        "logistic_regression": Candidate(
            name="logistic_regression",
            estimator=LogisticRegression(max_iter=1000, random_state=random_state),
            param_grid={"C": [0.1, 1.0, 10.0], "penalty": ["l2"]},
            interpretable=True,
        ),
        "decision_tree": Candidate(
            name="decision_tree",
            estimator=DecisionTreeClassifier(random_state=random_state),
            param_grid={"max_depth": [3, 5, 8, None], "min_samples_leaf": [1, 5, 10]},
            interpretable=True,
        ),
        "random_forest": Candidate(
            name="random_forest",
            estimator=RandomForestClassifier(random_state=random_state, n_jobs=-1),
            param_grid={
                "n_estimators": [100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_leaf": [1, 2, 5],
            },
        ),
        "gradient_boosting": Candidate(
            name="gradient_boosting",
            estimator=GradientBoostingClassifier(random_state=random_state),
            param_grid={
                "n_estimators": [100, 200],
                "learning_rate": [0.05, 0.1],
                "max_depth": [2, 3],
            },
        ),
    }


def _regression_candidates(random_state: int) -> dict[str, Candidate]:
    return {
        "linear_regression": Candidate(
            name="linear_regression",
            estimator=LinearRegression(),
            param_grid={},
            interpretable=True,
        ),
        "ridge": Candidate(
            name="ridge",
            estimator=Ridge(random_state=random_state),
            param_grid={"alpha": [0.1, 1.0, 10.0]},
            interpretable=True,
        ),
        "decision_tree": Candidate(
            name="decision_tree",
            estimator=DecisionTreeRegressor(random_state=random_state),
            param_grid={"max_depth": [3, 5, 8, None], "min_samples_leaf": [1, 5, 10]},
            interpretable=True,
        ),
        "random_forest": Candidate(
            name="random_forest",
            estimator=RandomForestRegressor(random_state=random_state, n_jobs=-1),
            param_grid={
                "n_estimators": [100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_leaf": [1, 2, 5],
            },
        ),
        "gradient_boosting": Candidate(
            name="gradient_boosting",
            estimator=GradientBoostingRegressor(random_state=random_state),
            param_grid={
                "n_estimators": [100, 200],
                "learning_rate": [0.05, 0.1],
                "max_depth": [2, 3],
            },
        ),
    }


def _boost_candidates(task: str, random_state: int) -> dict[str, Candidate]:
    """Return xgboost and lightgbm candidates when the extra is installed."""
    candidates: dict[str, Candidate] = {}
    try:
        from xgboost import XGBClassifier, XGBRegressor

        estimator = (
            XGBClassifier(
                random_state=random_state,
                n_jobs=-1,
                eval_metric="logloss",
                tree_method="hist",
            )
            if task == "classification"
            else XGBRegressor(random_state=random_state, n_jobs=-1, tree_method="hist")
        )
        candidates["xgboost"] = Candidate(
            name="xgboost",
            estimator=estimator,
            param_grid={
                "n_estimators": [100, 200],
                "learning_rate": [0.05, 0.1],
                "max_depth": [3, 6],
            },
        )
    except ImportError:
        pass

    try:
        from lightgbm import LGBMClassifier, LGBMRegressor

        estimator = (
            LGBMClassifier(random_state=random_state, n_jobs=-1, verbosity=-1)
            if task == "classification"
            else LGBMRegressor(random_state=random_state, n_jobs=-1, verbosity=-1)
        )
        candidates["lightgbm"] = Candidate(
            name="lightgbm",
            estimator=estimator,
            param_grid={
                "n_estimators": [100, 200],
                "learning_rate": [0.05, 0.1],
                "num_leaves": [15, 31],
            },
        )
    except ImportError:
        pass

    return candidates


def _select_supervised(
    pool: dict[str, Candidate],
    n_rows: int,
    n_features: int,
    constraints: dict[str, Any],
    log: DecisionLog,
) -> list[Candidate]:
    interpretable = bool(constraints.get("interpretable", False))
    speed = bool(constraints.get("max_train_seconds")) and float(
        constraints.get("max_train_seconds", 1e9)
    ) <= 30

    chosen: list[Candidate] = []

    if interpretable:
        chosen = [c for c in pool.values() if c.interpretable]
        log.record(
            "recommend",
            "Constraint interpretable=True: shortlist limited to interpretable models.",
            "constraint-interpretable",
            {"models": [c.name for c in chosen]},
        )
        return chosen

    for cand in pool.values():
        linear_models = ("logistic_regression", "ridge", "linear_regression")
        if cand.interpretable and cand.name in linear_models:
            chosen.append(cand)

    if n_rows <= SMALL_DATASET_ROWS:
        for name in ("decision_tree", "random_forest", "gradient_boosting"):
            if name in pool:
                chosen.append(pool[name])
        log.record(
            "recommend",
            f"Small dataset ({n_rows} rows): include tree and ensemble models.",
            "size-small",
            {"n_rows": n_rows, "threshold": SMALL_DATASET_ROWS},
        )
    elif n_rows <= LARGE_DATASET_ROWS:
        for name in ("random_forest", "gradient_boosting"):
            if name in pool:
                chosen.append(pool[name])
        log.record(
            "recommend",
            f"Medium dataset ({n_rows} rows): prefer ensemble models.",
            "size-medium",
            {"n_rows": n_rows},
        )
    else:
        for name in ("random_forest",):
            if name in pool:
                chosen.append(pool[name])
        log.record(
            "recommend",
            f"Large dataset ({n_rows} rows): cap ensemble breadth for training cost.",
            "size-large",
            {"n_rows": n_rows, "threshold": LARGE_DATASET_ROWS},
        )

    if speed:
        chosen = [c for c in chosen if c.name != "gradient_boosting"]
        log.record(
            "recommend",
            "Tight time budget: drop gradient boosting from the shortlist.",
            "constraint-speed",
            {},
        )

    # Deduplicate while preserving order.
    seen: set[str] = set()
    unique = []
    for cand in chosen:
        if cand.name not in seen:
            seen.add(cand.name)
            unique.append(cand)
    return unique


def recommend_models(
    task: str,
    n_rows: int,
    n_features: int,
    constraints: dict[str, Any] | None = None,
    random_state: int = 42,
    log: DecisionLog | None = None,
    use_boost: bool = True,
) -> list[Candidate]:
    """Return a candidate shortlist for the task using documented rules.

    Args:
        task: classification, regression, or clustering.
        n_rows: Number of training rows.
        n_features: Number of features after preprocessing.
        constraints: Optional operator constraints.
        random_state: Seed for the estimators.
        log: Decision log.
        use_boost: Whether to add xgboost and lightgbm candidates if present.

    Returns:
        An ordered list of Candidate objects.
    """
    log = log if log is not None else DecisionLog()
    constraints = constraints or {}

    if task == "clustering":
        candidate = Candidate(
            name="kmeans",
            estimator=KMeans(random_state=random_state, n_init=10),
            param_grid={"n_clusters": [2, 3, 4, 5, 6, 8]},
        )
        log.record(
            "recommend",
            "Clustering task: KMeans with a sweep over cluster counts.",
            "clustering-kmeans",
            {"n_clusters_grid": candidate.param_grid["n_clusters"]},
        )
        return [candidate]

    if task == "classification":
        pool = _classification_candidates(random_state)
    elif task == "regression":
        pool = _regression_candidates(random_state)
    else:
        raise ValueError(f"Unknown task '{task}'.")

    chosen = _select_supervised(pool, n_rows, n_features, constraints, log)

    if use_boost and not constraints.get("interpretable", False):
        boost = _boost_candidates(task, random_state)
        if boost:
            chosen.extend(boost.values())
            log.record(
                "recommend",
                f"Optional boosters available: added {list(boost)}.",
                "boost-extra-present",
                {"models": list(boost)},
            )

    log.record(
        "recommend",
        f"Final shortlist: {[c.name for c in chosen]}.",
        "shortlist",
        {"models": [c.name for c in chosen]},
    )
    return chosen
