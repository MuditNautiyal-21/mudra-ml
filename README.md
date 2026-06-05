# MudraML

MudraML automates the common data science workflow and shows its work. You point it at a data file, optionally state a goal, and it ingests the data, profiles it, cleans it, picks an algorithm, trains and tunes a shortlist of models, evaluates them, and returns the best fitted model together with a report of every decision it made and the rule behind that decision.

The point of difference is the decision engine. It is rule-based and statistical, not another model. Outlier handling uses IQR or z-score rules. Missing values are filled by median, mode, or a constant, or the column is dropped past a missingness threshold. The algorithm shortlist comes from a documented rule set keyed on the task, the dataset size, the feature count, and your constraints. Every one of those choices is written into the report, so a person can read why the pipeline did what it did and disagree with it if they want.

This is the glass-box position: the models are the product, and the way the pipeline reaches them is auditable rather than hidden inside a search.

## Install

```
pip install mudraml
```

Optional extras:

```
pip install mudraml[files]    # parquet and excel readers
pip install mudraml[boost]    # xgboost and lightgbm candidates
```

The library runs fully on the scikit-learn core. The boosters are added to the shortlist only when the extra is installed.

## Quickstart

Fully automatic. MudraML infers the task, the target, and the metric:

```python
from mudraml import Mudra

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

## Predict and reuse

```python
result.save("run_artifact")            # pipeline + model + metadata
loaded = Mudra.load("run_artifact")
preds = loaded.predict(new_dataframe)
```

The preprocessing pipeline travels with the model, so new rows are transformed the same way the training rows were.

## Command line

```
mudraml run data.csv --target churn --task classification --metric f1
mudraml profile data.csv
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
| classification | f1 | accuracy, precision, recall, roc_auc, confusion matrix |
| regression | rmse | mae, mse, r2 |
| clustering | silhouette | davies_bouldin |

## Scope

This release covers the supervised classification and regression cases and KMeans clustering, end to end, with the decision log and the report. Deep text modeling, time series, model-based imputation, and a search beyond curated grids are out of scope by design, since the engine is meant to stay explainable. See the changelog for the version history.

## License

MIT. See [LICENSE](LICENSE).
