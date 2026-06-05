from __future__ import annotations

import pandas as pd
import pytest

from mudra_ml.goal import Goal, infer_goal
from mudra_ml.profile import DataProfiler


def _profile(frame):
    return DataProfiler().profile(frame)


def test_infer_classification_from_boolean_target():
    frame = pd.DataFrame({"x": range(50), "label": [True, False] * 25})
    goal = infer_goal(_profile(frame), Goal(target="label"))
    assert goal.task == "classification"


def test_infer_regression_from_continuous_target(regression_frame):
    goal = infer_goal(_profile(regression_frame), Goal(target="price"))
    assert goal.task == "regression"
    assert goal.metric == "rmse"


def test_infer_classification_from_low_cardinality(classification_frame):
    goal = infer_goal(_profile(classification_frame), Goal(target="target"))
    assert goal.task == "classification"
    assert goal.metric == "f1"


def test_operator_task_overrides_inference(regression_frame):
    # Force classification even though the target is continuous.
    frame = regression_frame.copy()
    frame["price"] = (frame["price"] > frame["price"].median()).astype(int)
    goal = infer_goal(_profile(frame), Goal(target="price", task="classification"))
    assert goal.task == "classification"


def test_operator_metric_preserved(classification_frame):
    goal = infer_goal(_profile(classification_frame), Goal(target="target", metric="roc_auc"))
    assert goal.metric == "roc_auc"


def test_target_inferred_when_absent():
    frame = pd.DataFrame(
        {"feature1": range(100), "feature2": range(100, 200), "churn": [0, 1] * 50}
    )
    goal = infer_goal(_profile(frame), Goal())
    assert goal.target == "churn"


def test_clustering_when_no_target():
    frame = pd.DataFrame({f"x{i}": range(50) for i in range(4)})
    goal = infer_goal(_profile(frame), Goal(task="clustering"))
    assert goal.task == "clustering"
    assert goal.target is None
    assert goal.metric == "silhouette"


def test_unknown_task_rejected():
    with pytest.raises(ValueError, match="Unknown task"):
        Goal(task="forecasting").validate()


def test_bad_metric_for_task_rejected():
    with pytest.raises(ValueError, match="not valid"):
        Goal(task="regression", metric="f1").validate()


def test_missing_target_column_rejected(classification_frame):
    with pytest.raises(ValueError, match="not in the data"):
        infer_goal(_profile(classification_frame), Goal(target="nope"))


def test_operator_set_fields_tracked():
    goal = Goal(target="y", task="classification")
    assert set(goal.operator_set_fields()) == {"target", "task"}
