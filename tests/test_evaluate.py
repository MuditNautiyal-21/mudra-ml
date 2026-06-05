from __future__ import annotations

from sklearn.model_selection import train_test_split

from mudra_ml.evaluate import evaluate
from mudra_ml.recommend import recommend_models


def _split(frame, target):
    X = frame.drop(columns=[target]).to_numpy()
    y = frame[target].to_numpy()
    return train_test_split(X, y, test_size=0.25, random_state=42)


def test_classification_evaluation_selects_best(classification_frame):
    X_tr, X_te, y_tr, y_te = _split(classification_frame, "target")
    candidates = recommend_models("classification", len(X_tr), X_tr.shape[1], use_boost=False)
    result = evaluate(
        candidates, "classification", "f1",
        [f"f{i}" for i in range(X_tr.shape[1])],
        X_tr, y_tr, X_te, y_te,
    )
    assert result.best_name
    assert result.best.test_metrics["f1"] > 0.8
    assert "accuracy" in result.best.test_metrics
    assert "confusion_matrix" in result.best.test_metrics


def test_regression_evaluation_reports_rmse(regression_frame):
    X_tr, X_te, y_tr, y_te = _split(regression_frame, "price")
    candidates = recommend_models("regression", len(X_tr), X_tr.shape[1], use_boost=False)
    result = evaluate(
        candidates, "regression", "rmse",
        [f"f{i}" for i in range(X_tr.shape[1])],
        X_tr, y_tr, X_te, y_te,
    )
    assert "rmse" in result.best.test_metrics
    assert "r2" in result.best.test_metrics


def test_feature_importance_extracted(classification_frame):
    X_tr, X_te, y_tr, y_te = _split(classification_frame, "target")
    names = [f"f{i}" for i in range(X_tr.shape[1])]
    candidates = recommend_models("classification", len(X_tr), X_tr.shape[1], use_boost=False)
    result = evaluate(candidates, "classification", "f1", names, X_tr, y_tr, X_te, y_te)
    assert len(result.feature_importance) > 0


def test_clustering_evaluation(clustering_frame):
    X = clustering_frame.to_numpy()
    candidates = recommend_models("clustering", len(X), X.shape[1])
    result = evaluate(candidates, "clustering", "silhouette", list(clustering_frame.columns), X)
    assert result.best_name.startswith("kmeans")
    assert result.best.test_metrics["silhouette"] > 0


def test_determinism_same_seed(classification_frame):
    X_tr, X_te, y_tr, y_te = _split(classification_frame, "target")
    names = [f"f{i}" for i in range(X_tr.shape[1])]

    def run():
        cands = recommend_models("classification", len(X_tr), X_tr.shape[1], use_boost=False)
        return evaluate(cands, "classification", "f1", names, X_tr, y_tr, X_te, y_te)

    a = run()
    b = run()
    assert a.best_name == b.best_name
    assert abs(a.best.test_metrics["f1"] - b.best.test_metrics["f1"]) < 1e-12
