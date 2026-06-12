"""Rule-based algorithm recommendation.

The shortlist is produced by a documented rule set keyed on task type, dataset
size, dimensionality, sparsity, and the operator's interpretability and speed
preferences. No model is trained to make this choice.

The size and shape rules:

- Linear models (logistic regression, linear regression, ridge) are always on
  the shortlist. They are cheap, stable, and a sanity reference.
- Small datasets (rows at or below SMALL_DATASET_ROWS) add the decision tree,
  random forest, gradient boosting, and the kernel model (SVC or SVR), whose
  training cost grows roughly quadratically with rows and is affordable here.
- K-nearest neighbors joins the small-data shortlist only when the feature
  count is at or below KNN_MAX_FEATURES and the transformed matrix is not
  sparse (zero fraction below KNN_MAX_SPARSITY). Distances lose meaning in
  high dimensions and on mostly-zero one-hot matrices.
- Medium datasets (up to LARGE_DATASET_ROWS) shortlist the ensembles: random
  forest, extra trees, and gradient boosting.
- Large datasets cap the shortlist at random forest to bound training cost.
- Wide data (more features than rows) adds gaussian naive bayes for
  classification and elastic net for regression. Both stay usable when the
  feature count outruns the row count.
- A tight time budget removes gradient boosting and the kernel models.
- The optional boosters (xgboost, lightgbm, catboost) are appended when their
  libraries are installed. A missing library is skipped with a logged note.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any

from sklearn.cluster import KMeans
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, LinearRegression, LogisticRegression, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .constants import (
    KNN_MAX_FEATURES,
    KNN_MAX_SPARSITY,
    LARGE_DATASET_ROWS,
    SMALL_DATASET_ROWS,
)
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
        # The forests run single-threaded. Their parallel predict adds tree
        # outputs in thread completion order, and float addition is not
        # associative, so repeated predictions differ in the last bits. The
        # search already parallelizes across configurations.
        "random_forest": Candidate(
            name="random_forest",
            estimator=RandomForestClassifier(random_state=random_state, n_jobs=1),
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
        "extra_trees": Candidate(
            name="extra_trees",
            estimator=ExtraTreesClassifier(random_state=random_state, n_jobs=1),
            param_grid={
                "n_estimators": [100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_leaf": [1, 2, 5],
            },
        ),
        "svc": Candidate(
            name="svc",
            estimator=SVC(probability=True, random_state=random_state),
            param_grid={"C": [0.1, 1.0, 10.0], "kernel": ["rbf", "linear"]},
        ),
        "k_nearest_neighbors": Candidate(
            name="k_nearest_neighbors",
            estimator=KNeighborsClassifier(),
            param_grid={"n_neighbors": [3, 5, 11], "weights": ["uniform", "distance"]},
        ),
        "gaussian_naive_bayes": Candidate(
            name="gaussian_naive_bayes",
            estimator=GaussianNB(),
            param_grid={"var_smoothing": [1e-9, 1e-8, 1e-7]},
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
        # Single-threaded for bit-identical repeated predictions; see the
        # note on the classification forests.
        "random_forest": Candidate(
            name="random_forest",
            estimator=RandomForestRegressor(random_state=random_state, n_jobs=1),
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
        "extra_trees": Candidate(
            name="extra_trees",
            estimator=ExtraTreesRegressor(random_state=random_state, n_jobs=1),
            param_grid={
                "n_estimators": [100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_leaf": [1, 2, 5],
            },
        ),
        "svr": Candidate(
            name="svr",
            estimator=SVR(),
            param_grid={"C": [0.1, 1.0, 10.0], "epsilon": [0.05, 0.1]},
        ),
        "k_nearest_neighbors": Candidate(
            name="k_nearest_neighbors",
            estimator=KNeighborsRegressor(),
            param_grid={"n_neighbors": [3, 5, 11], "weights": ["uniform", "distance"]},
        ),
        "elastic_net": Candidate(
            name="elastic_net",
            estimator=ElasticNet(random_state=random_state, max_iter=5000),
            param_grid={"alpha": [0.01, 0.1, 1.0], "l1_ratio": [0.2, 0.5, 0.8]},
            interpretable=True,
        ),
    }


def _import_booster(name: str) -> Any | None:
    """Import an optional boosting library, or return None when absent."""
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


def _xgboost_candidate(module: Any, task: str, random_state: int) -> Candidate:
    estimator = (
        module.XGBClassifier(
            random_state=random_state,
            n_jobs=-1,
            eval_metric="logloss",
            tree_method="hist",
        )
        if task == "classification"
        else module.XGBRegressor(random_state=random_state, n_jobs=-1, tree_method="hist")
    )
    return Candidate(
        name="xgboost",
        estimator=estimator,
        param_grid={
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, 6],
        },
    )


def _lightgbm_candidate(module: Any, task: str, random_state: int) -> Candidate:
    estimator = (
        module.LGBMClassifier(random_state=random_state, n_jobs=-1, verbosity=-1)
        if task == "classification"
        else module.LGBMRegressor(random_state=random_state, n_jobs=-1, verbosity=-1)
    )
    return Candidate(
        name="lightgbm",
        estimator=estimator,
        param_grid={
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
            "num_leaves": [15, 31],
        },
    )


def _catboost_candidate(module: Any, task: str, random_state: int) -> Candidate:
    common = {
        "random_seed": random_state,
        "verbose": 0,
        "allow_writing_files": False,
    }
    estimator = (
        module.CatBoostClassifier(**common)
        if task == "classification"
        else module.CatBoostRegressor(**common)
    )
    return Candidate(
        name="catboost",
        estimator=estimator,
        param_grid={
            "iterations": [100, 200],
            "learning_rate": [0.05, 0.1],
            "depth": [4, 6],
        },
    )


_BOOSTER_BUILDERS = {
    "xgboost": _xgboost_candidate,
    "lightgbm": _lightgbm_candidate,
    "catboost": _catboost_candidate,
}


def _boost_candidates(task: str, random_state: int, log: DecisionLog) -> dict[str, Candidate]:
    """Return the optional boosting candidates whose libraries are installed.

    Each library is imported lazily. A missing library never fails the run:
    it is skipped with a logged note so the report shows why the model was
    not considered.
    """
    candidates: dict[str, Candidate] = {}
    for name, builder in _BOOSTER_BUILDERS.items():
        module = _import_booster(name)
        if module is None:
            log.record(
                "recommend",
                f"Optional booster {name} is not installed: skipped. Install "
                f"it with 'pip install mudra-ml[boost]' to include it.",
                "boost-extra-missing",
                {"library": name},
            )
            continue
        candidates[name] = builder(module, task, random_state)
    return candidates


def _select_supervised(
    pool: dict[str, Candidate],
    task: str,
    n_rows: int,
    n_features: int,
    constraints: dict[str, Any],
    log: DecisionLog,
    sparsity: float | None = None,
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
        kernel_model = "svc" if task == "classification" else "svr"
        for name in ("decision_tree", "random_forest", "gradient_boosting", kernel_model):
            if name in pool:
                chosen.append(pool[name])
        log.record(
            "recommend",
            f"Small dataset ({n_rows} rows): include tree, ensemble, and kernel "
            f"models. Kernel training cost is affordable at this size.",
            "size-small",
            {"n_rows": n_rows, "threshold": SMALL_DATASET_ROWS},
        )
        if "k_nearest_neighbors" in pool:
            low_dimensional = n_features <= KNN_MAX_FEATURES
            dense_enough = sparsity is None or sparsity < KNN_MAX_SPARSITY
            if low_dimensional and dense_enough:
                chosen.append(pool["k_nearest_neighbors"])
                log.record(
                    "recommend",
                    f"K-nearest neighbors included: {n_features} features is at "
                    f"or below {KNN_MAX_FEATURES} and the matrix is dense enough "
                    f"for distances to stay meaningful.",
                    "knn-low-dimensional-dense",
                    {"n_features": n_features, "sparsity": sparsity},
                )
            else:
                reason = (
                    f"{n_features} features exceeds {KNN_MAX_FEATURES}"
                    if not low_dimensional
                    else f"sparsity {sparsity:.2f} is at or above {KNN_MAX_SPARSITY}"
                )
                log.record(
                    "recommend",
                    f"K-nearest neighbors skipped: {reason}. Distances lose "
                    f"meaning in high dimensions and on sparse matrices.",
                    "knn-skipped",
                    {"n_features": n_features, "sparsity": sparsity},
                )
    elif n_rows <= LARGE_DATASET_ROWS:
        for name in ("random_forest", "extra_trees", "gradient_boosting"):
            if name in pool:
                chosen.append(pool[name])
        log.record(
            "recommend",
            f"Medium dataset ({n_rows} rows): prefer the ensemble models "
            f"(random forest, extra trees, gradient boosting).",
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

    if n_features >= n_rows:
        wide_model = "gaussian_naive_bayes" if task == "classification" else "elastic_net"
        if wide_model in pool:
            chosen.append(pool[wide_model])
            log.record(
                "recommend",
                f"Wide data ({n_features} features, {n_rows} rows): include "
                f"{wide_model}, which stays usable when features outnumber rows.",
                "wide-data-model",
                {"n_features": n_features, "n_rows": n_rows, "model": wide_model},
            )

    if speed:
        slow_models = ("gradient_boosting", "svc", "svr")
        dropped = [c.name for c in chosen if c.name in slow_models]
        if dropped:
            chosen = [c for c in chosen if c.name not in slow_models]
            log.record(
                "recommend",
                f"Tight time budget: drop {dropped} from the shortlist.",
                "constraint-speed",
                {"dropped": dropped},
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
    sparsity: float | None = None,
) -> list[Candidate]:
    """Return a candidate shortlist for the task using documented rules.

    Args:
        task: classification, regression, or clustering.
        n_rows: Number of training rows.
        n_features: Number of features after preprocessing.
        constraints: Optional operator constraints.
        random_state: Seed for the estimators.
        log: Decision log.
        use_boost: Whether to add the optional boosters if installed.
        sparsity: Fraction of zero entries in the transformed feature matrix,
            or None when unknown. Drives the nearest-neighbor rule.

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

    chosen = _select_supervised(pool, task, n_rows, n_features, constraints, log, sparsity)

    if use_boost and not constraints.get("interpretable", False):
        boost = _boost_candidates(task, random_state, log)
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
