from __future__ import annotations

from mudraml.recommend import recommend_models


def test_classification_shortlist_nonempty():
    shortlist = recommend_models("classification", n_rows=500, n_features=10, use_boost=False)
    names = [c.name for c in shortlist]
    assert "logistic_regression" in names
    assert len(names) >= 2


def test_regression_shortlist_nonempty():
    shortlist = recommend_models("regression", n_rows=500, n_features=10, use_boost=False)
    names = [c.name for c in shortlist]
    assert any(n in names for n in ("linear_regression", "ridge"))


def test_interpretable_constraint_limits_models():
    shortlist = recommend_models(
        "classification", n_rows=500, n_features=10, constraints={"interpretable": True}
    )
    assert all(c.interpretable for c in shortlist)
    assert all(c.name in ("logistic_regression", "decision_tree") for c in shortlist)


def test_large_dataset_caps_complexity():
    small = recommend_models("classification", n_rows=500, n_features=10, use_boost=False)
    large = recommend_models("classification", n_rows=100000, n_features=10, use_boost=False)
    assert len(large) <= len(small)


def test_speed_constraint_drops_boosting():
    shortlist = recommend_models(
        "regression",
        n_rows=500,
        n_features=10,
        constraints={"max_train_seconds": 10},
        use_boost=False,
    )
    assert "gradient_boosting" not in [c.name for c in shortlist]


def test_clustering_returns_kmeans():
    shortlist = recommend_models("clustering", n_rows=200, n_features=4)
    assert len(shortlist) == 1
    assert shortlist[0].name == "kmeans"
    assert "n_clusters" in shortlist[0].param_grid


def test_grids_are_curated_and_small():
    for cand in recommend_models("classification", n_rows=500, n_features=10, use_boost=False):
        for values in cand.param_grid.values():
            assert len(values) <= 4
