from __future__ import annotations

from mudra_ml.recommend import recommend_models


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


def test_small_classification_includes_svc():
    names = [c.name for c in recommend_models("classification", n_rows=500, n_features=10, use_boost=False)]
    assert "svc" in names


def test_small_regression_includes_svr():
    names = [c.name for c in recommend_models("regression", n_rows=500, n_features=10, use_boost=False)]
    assert "svr" in names


def test_small_low_dimensional_includes_knn():
    for task in ("classification", "regression"):
        names = [c.name for c in recommend_models(task, n_rows=500, n_features=10, use_boost=False)]
        assert "k_nearest_neighbors" in names


def test_high_dimensional_excludes_knn():
    names = [c.name for c in recommend_models("classification", n_rows=500, n_features=200, use_boost=False)]
    assert "k_nearest_neighbors" not in names


def test_high_sparsity_excludes_knn():
    dense = [c.name for c in recommend_models("classification", n_rows=500, n_features=10, sparsity=0.1, use_boost=False)]
    sparse = [c.name for c in recommend_models("classification", n_rows=500, n_features=10, sparsity=0.8, use_boost=False)]
    assert "k_nearest_neighbors" in dense
    assert "k_nearest_neighbors" not in sparse


def test_wide_classification_includes_gaussian_naive_bayes():
    names = [c.name for c in recommend_models("classification", n_rows=80, n_features=120, use_boost=False)]
    assert "gaussian_naive_bayes" in names


def test_narrow_classification_excludes_gaussian_naive_bayes():
    names = [c.name for c in recommend_models("classification", n_rows=5000, n_features=10, use_boost=False)]
    assert "gaussian_naive_bayes" not in names


def test_wide_regression_includes_elastic_net():
    names = [c.name for c in recommend_models("regression", n_rows=80, n_features=120, use_boost=False)]
    assert "elastic_net" in names


def test_medium_dataset_includes_extra_trees():
    for task in ("classification", "regression"):
        names = [c.name for c in recommend_models(task, n_rows=10000, n_features=10, use_boost=False)]
        assert "extra_trees" in names


def test_medium_dataset_excludes_kernel_and_neighbor_models():
    names = [c.name for c in recommend_models("classification", n_rows=10000, n_features=10, use_boost=False)]
    assert "svc" not in names
    assert "k_nearest_neighbors" not in names


def test_speed_constraint_drops_kernel_models():
    names = [
        c.name
        for c in recommend_models(
            "classification",
            n_rows=500,
            n_features=10,
            constraints={"max_train_seconds": 10},
            use_boost=False,
        )
    ]
    assert "svc" not in names


def test_shortlist_never_contains_every_candidate():
    from mudra_ml.recommend import _classification_candidates

    pool_size = len(_classification_candidates(42))
    names = [c.name for c in recommend_models("classification", n_rows=10000, n_features=10, use_boost=False)]
    assert len(set(names)) < pool_size


def test_missing_boosters_are_skipped_with_a_logged_note(monkeypatch):
    import mudra_ml.recommend as rec
    from mudra_ml.decisions import DecisionLog

    monkeypatch.setattr(rec, "_import_booster", lambda name: None)
    log = DecisionLog()
    shortlist = recommend_models(
        "classification", n_rows=500, n_features=10, use_boost=True, log=log
    )
    names = [c.name for c in shortlist]
    assert "xgboost" not in names
    assert "lightgbm" not in names
    assert "catboost" not in names
    skips = [e for e in log if e.rule == "boost-extra-missing"]
    assert len(skips) == 3
