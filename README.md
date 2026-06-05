# Mudra-ML

Automated, glass-box data science. Point it at a data file, get a fitted model and a report of every decision behind it.

[![PyPI version](https://img.shields.io/pypi/v/mudra-ml.svg?v=1)](https://pypi.org/project/mudra-ml/)
[![Python versions](https://img.shields.io/pypi/pyversions/mudra-ml.svg?v=1)](https://pypi.org/project/mudra-ml/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/MuditNautiyal-21/mudra-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/MuditNautiyal-21/mudra-ml/actions/workflows/ci.yml)

MudraML automates the common data science workflow and shows its work. You point it at a data file, optionally state a goal, and it ingests the data, profiles it, cleans it, picks an algorithm, trains and tunes a shortlist of models, evaluates them, and returns the best fitted model together with a report of every decision it made and the rule behind that decision.

The point of difference is the decision engine. It is rule-based and statistical, not another model. Outlier handling uses IQR or z-score rules. Missing values are filled by median, mode, or a constant, or the column is dropped past a missingness threshold. The algorithm shortlist comes from a documented rule set keyed on the task, the dataset size, the feature count, and your constraints. Every one of those choices is written into the report, so a person can read why the pipeline did what it did and disagree with it if they want.

This is the glass-box position: the models are the product, and the way the pipeline reaches them is auditable rather than hidden inside a search.

## Install

```
pip install mudra-ml
```

Optional extras:

```
pip install mudra-ml[files]    # parquet and excel readers
pip install mudra-ml[boost]    # xgboost and lightgbm candidates
```

The library runs fully on the scikit-learn core. The boosters are added to the shortlist only when the extra is installed.

## Quickstart

Fully automatic. MudraML infers the task, the target, and the metric:

```python
from mudra_ml import Mudra

m = Mudra()
result = m.run("data.csv")
print(result.report_path)   # markdown and HTML report on disk
model = result.best_model   # fitted, ready to predict
```

Operator-defined goal. You set what you care about and MudraML honors it:

```python
result = m.run(
    "churn.csv",
    target="churn",
    task="classification",
    metric="f1",
    constraints={"interpretable": True, "max_train_seconds": 120},
)
```

When `interpretable` is set, the shortlist is limited to models you can read directly, such as logistic regression and a single decision tree. The report states which goal fields you set and which ones were inferred.

## What the report looks like

Every run writes a markdown report and an HTML report. The HTML report carries the same content plus diagnostic charts (confusion matrix heatmap, ROC and precision-recall curves, target distribution, feature correlation, residual and predicted-versus-actual plots for regression). The block below is an excerpt from a real run on the scikit-learn breast cancer dataset.

```
## Trust summary

Held-out test size: 114 rows. Training size: 455 rows.
Baseline: dummy_most_frequent (no learning, predicts the most frequent class or the mean).

| Metric    | Best model | Baseline | Difference |
| --------- | ---------- | -------- | ---------- |
| accuracy  | 0.9737     | 0.6316   | 0.3421     |
| f1        | 0.9793     | 0.7742   | 0.2051     |
| precision | 0.9726     | 0.6316   | 0.3410     |
| recall    | 0.9861     | 1.0000   | -0.0139    |
| roc_auc   | 0.9970     | 0.5000   | 0.4970     |

Train vs test gap on selected metrics (positive means train is better than test).

| Metric    | Train  | Test   | Gap    |
| --------- | ------ | ------ | ------ |
| accuracy  | 0.9846 | 0.9737 | 0.0109 |
| f1        | 0.9878 | 0.9793 | 0.0085 |
| precision | 0.9793 | 0.9726 | 0.0067 |
| recall    | 0.9965 | 0.9861 | 0.0104 |

## Result

Selected model: logistic_regression
Cross-validation score: 0.9801 +/- 0.0129

### Per-class report

| Class | Precision | Recall | F1     | Support |
| ----- | --------- | ------ | ------ | ------- |
| 0     | 0.9756    | 0.9524 | 0.9639 | 42      |
| 1     | 0.9726    | 0.9861 | 0.9793 | 72      |

## Feature importance (permutation, mean across 10 repeats)

Impurity importance is biased toward high-cardinality features. The permutation view is more reliable because it scores each feature by how much shuffling it hurts the model.

- worst smoothness: 0.0193 (+/- 0.0102)
- worst texture: 0.0175 (+/- 0.0111)
- area error: 0.0149 (+/- 0.0111)
- worst concave points: 0.0114 (+/- 0.0056)
- mean smoothness: 0.0105 (+/- 0.0086)
```

The numbers above come from a real run. The HTML report adds confusion matrix, ROC, and precision-recall charts alongside these tables.

## Predict and reuse

```python
result.save("run_artifact")            # pipeline + model + metadata
loaded = Mudra.load("run_artifact")
preds = loaded.predict(new_dataframe)
```

The preprocessing pipeline travels with the model, so new rows are transformed the same way the training rows were.

## Command line

```
mudra-ml run data.csv --target churn --task classification --metric f1
mudra-ml profile data.csv
```

`run` writes the report and prints the selected model and its held-out metrics. `profile` prints the inferred column types, missingness, cardinality, and the candidate target columns.

## What it does, stage by stage

1. Ingest. Readers for csv, tsv, excel, json, and parquet. For delimited text the delimiter, encoding, and header row are detected.
2. Profile. Per-column type inference (numeric, categorical, datetime, boolean, id, text), missingness, cardinality, distribution stats, and candidate-target ranking.
3. Goal. Rule-based inference of the task, target, and metric, with any field you set taking precedence.
4. Preprocess. A leakage-safe scikit-learn Pipeline and ColumnTransformer. Imputation, datetime part extraction, outlier clipping, encoding, and scaling are all fit on the training split only.
5. Recommend. A documented rule set returns a candidate shortlist.
6. Train and evaluate. Cross-validated training, tuning with RandomizedSearchCV at a fixed seed, held-out scoring, best-model selection, and feature importance where the model exposes it.
7. Report. Markdown and HTML that log every decision and the rule that produced it.

## Why leakage safety matters here

Every statistic that preprocessing needs (a median, a category frequency, an outlier bound, a scaler mean) is learned during `fit`. MudraML fits the pipeline on the training split and only transforms the test split. No information from the test data reaches the model through preprocessing. The test suite checks this property directly: it fits on a slice with a known mean and confirms the learned imputation value matches the train slice rather than the whole dataset.

## Determinism

One `random_state` is threaded through every stochastic step (the split, the search, the estimators) and defaults to a fixed value. Two runs on the same data and the same goal produce the same result and the same report.

## Tasks and metrics

| Task | Default metric | Also reported |
| --- | --- | --- |
| classification | f1 | accuracy, precision, recall, roc_auc, per-class precision/recall/f1, confusion matrix, ROC and precision-recall curves |
| regression | rmse | mae, mse, r2, residual mean and std, predicted vs actual |
| clustering | silhouette | davies_bouldin |

## Trust and data quality

Every run reports the headline metrics against a naive baseline (most-frequent class for classification, mean for regression), the cross-validation score as mean plus or minus standard deviation across folds, and the train versus test gap so that overfitting is visible. When the held-out set is below 50 rows the metrics are labeled indicative only. Permutation importance with its standard deviation is reported alongside impurity importance, with a note that impurity importance is biased toward high-cardinality features.

The data-quality section calls out constant columns, duplicate rows, high-cardinality categoricals, class imbalance, missing targets, and features that look suspiciously predictive of the target (a simple leakage check). A limitations and next-steps section turns those warnings into concrete actions.

## Stress tested

The pipeline is exercised against a battery of adversarial datasets: a tiny set, a single-feature set, a single-class target, an all-missing column, a constant column, all-duplicate rows, a wide dataset, an id-like high-cardinality feature, a strongly imbalanced target, mixed dtypes with messy datetimes, a leakage-injected dataset where a feature equals the target, a target with missing values, and a 10k-row dataset. All four task variants run: binary and multiclass classification, regression, and clustering.

## Scope

This release covers the supervised classification and regression cases and KMeans clustering, end to end, with the decision log and the report. Deep text modeling, time series, model-based imputation, and a search beyond curated grids are out of scope by design, since the engine is meant to stay explainable. See the changelog for the version history.

## License

MIT. See [LICENSE](LICENSE).
