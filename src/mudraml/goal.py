"""Goal definition and rule-based goal inference.

The Goal is the human-in-the-loop surface. An operator can set any field
explicitly, and inference fills the rest. The report states which fields the
operator set and which the engine inferred.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .constants import (
    CLASSIFICATION_MAX_CLASSES,
    DEFAULT_METRICS,
    VALID_TASKS,
)
from .decisions import DecisionLog
from .profile import BOOLEAN, CATEGORICAL, DataProfile

_CLASSIFICATION_METRICS = {"accuracy", "f1", "f1_macro", "roc_auc", "precision", "recall"}
_REGRESSION_METRICS = {"rmse", "mae", "r2", "mse"}
_CLUSTERING_METRICS = {"silhouette", "davies_bouldin"}

_METRICS_BY_TASK = {
    "classification": _CLASSIFICATION_METRICS,
    "regression": _REGRESSION_METRICS,
    "clustering": _CLUSTERING_METRICS,
}


@dataclass
class Goal:
    """What the operator wants from a run.

    Args:
        target: Name of the target column, or None for unsupervised or inferred.
        task: One of classification, regression, clustering, or None to infer.
        metric: Metric to optimize, or None for the task default.
        constraints: Optional knobs such as {"interpretable": True,
            "max_train_seconds": 120, "missing_threshold": 0.6,
            "outlier_strategy": "iqr"}.
        random_state: Seed threaded through every stochastic step.
    """

    target: str | None = None
    task: str | None = None
    metric: str | None = None
    constraints: dict[str, Any] = field(default_factory=dict)
    random_state: int = 42

    def operator_set_fields(self) -> list[str]:
        """Return the names of fields the operator provided explicitly."""
        fields = []
        if self.target is not None:
            fields.append("target")
        if self.task is not None:
            fields.append("task")
        if self.metric is not None:
            fields.append("metric")
        if self.constraints:
            fields.append("constraints")
        return fields

    def validate(self) -> None:
        if self.task is not None and self.task not in VALID_TASKS:
            raise ValueError(
                f"Unknown task '{self.task}'. Choose from {VALID_TASKS}."
            )
        if self.metric is not None and self.task is not None:
            allowed = _METRICS_BY_TASK.get(self.task, set())
            if self.metric not in allowed:
                raise ValueError(
                    f"Metric '{self.metric}' is not valid for {self.task}. "
                    f"Choose from {sorted(allowed)}."
                )

    def as_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "task": self.task,
            "metric": self.metric,
            "constraints": self.constraints,
            "random_state": self.random_state,
        }


def _infer_task(profile: DataProfile, target: str, log: DecisionLog) -> str:
    """Infer the task from the target column type and cardinality."""
    col = profile.column(target)
    if col.inferred_type == BOOLEAN:
        log.record(
            "goal",
            f"Task inferred as classification from boolean target '{target}'.",
            "task-from-target-dtype",
            {"target_type": col.inferred_type},
        )
        return "classification"

    if col.inferred_type == CATEGORICAL or col.dtype == "object":
        log.record(
            "goal",
            f"Task inferred as classification from categorical target '{target}'.",
            "task-from-target-dtype",
            {"target_type": col.inferred_type, "n_unique": col.n_unique},
        )
        return "classification"

    if col.n_unique <= CLASSIFICATION_MAX_CLASSES:
        log.record(
            "goal",
            f"Task inferred as classification: target '{target}' has "
            f"{col.n_unique} distinct values (<= {CLASSIFICATION_MAX_CLASSES}).",
            "task-from-cardinality",
            {"n_unique": col.n_unique, "threshold": CLASSIFICATION_MAX_CLASSES},
        )
        return "classification"

    log.record(
        "goal",
        f"Task inferred as regression: numeric target '{target}' has "
        f"{col.n_unique} distinct values.",
        "task-from-cardinality",
        {"n_unique": col.n_unique, "threshold": CLASSIFICATION_MAX_CLASSES},
    )
    return "regression"


def infer_goal(
    profile: DataProfile,
    goal: Goal | None = None,
    log: DecisionLog | None = None,
) -> Goal:
    """Fill any unspecified Goal field using rules over the data profile.

    Explicit fields on the incoming Goal are never overwritten. The result is
    a fully specified Goal.

    Args:
        profile: Profile of the dataset.
        goal: Partially specified operator goal, or None.
        log: Decision log to record inference choices.

    Returns:
        A Goal with target, task, and metric all set.
    """
    log = log or DecisionLog()
    goal = goal or Goal()
    goal.validate()

    resolved = Goal(
        target=goal.target,
        task=goal.task,
        metric=goal.metric,
        constraints=dict(goal.constraints),
        random_state=goal.random_state,
    )

    operator_fields = goal.operator_set_fields()
    if operator_fields:
        log.record(
            "goal",
            f"Operator set: {', '.join(operator_fields)}.",
            "operator-override",
            {"fields": operator_fields},
        )

    # Resolve target.
    if resolved.target is None and resolved.task != "clustering":
        if profile.candidate_targets:
            resolved.target = profile.candidate_targets[0]
            log.record(
                "goal",
                f"Target inferred as '{resolved.target}' from candidate ranking.",
                "target-from-candidates",
                {"candidates": profile.candidate_targets[:3]},
            )
        else:
            resolved.task = "clustering"
            log.record(
                "goal",
                "No plausible target found. Falling back to clustering.",
                "no-target-to-clustering",
                {},
            )

    if resolved.target is not None and resolved.target not in profile.columns:
        raise ValueError(
            f"Target column '{resolved.target}' is not in the data. "
            f"Available columns: {list(profile.columns)[:10]}."
        )

    # Resolve task.
    if resolved.task is None:
        if resolved.target is None:
            resolved.task = "clustering"
            log.record(
                "goal",
                "Task inferred as clustering: no target column.",
                "task-from-no-target",
                {},
            )
        else:
            resolved.task = _infer_task(profile, resolved.target, log)

    if resolved.task == "clustering":
        resolved.target = None

    # Resolve metric.
    if resolved.metric is None:
        resolved.metric = DEFAULT_METRICS[resolved.task]
        log.record(
            "goal",
            f"Metric defaulted to '{resolved.metric}' for {resolved.task}.",
            "metric-default-per-task",
            {"task": resolved.task},
        )

    resolved.validate()
    return resolved
