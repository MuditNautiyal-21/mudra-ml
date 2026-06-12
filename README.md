# Mudra-ML

Point it at a table of data. It cleans the data, trains several models, picks the best one, and writes a report that explains every choice it made.

[![PyPI version](https://img.shields.io/pypi/v/mudra-ml.svg?v=1)](https://pypi.org/project/mudra-ml/)
[![Python versions](https://img.shields.io/pypi/pyversions/mudra-ml.svg?v=1)](https://pypi.org/project/mudra-ml/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/MuditNautiyal-21/mudra-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/MuditNautiyal-21/mudra-ml/actions/workflows/ci.yml)

## What it does

Mudra-ML automates the routine part of supervised machine learning on tables: reading the file, working out what each column holds, repairing messy values, splitting the data safely, choosing which algorithms to try, tuning them, and measuring the winner on data it never saw during training. Every one of those choices follows a written rule, and every rule that fired is logged and printed in the report. Nothing is decided by a hidden model, and the library never calls the network.

## Install

```
pip install mudra-ml
```

Optional extras:

```
pip install mudra-ml[files]    # parquet and excel readers
pip install mudra-ml[boost]    # xgboost, lightgbm, and catboost candidates
```

The base install runs fully on scikit-learn. The boosting libraries join the candidate list only when they are installed. A missing one is skipped with a note in the log, never an error.

## Quickstart

```python
from mudra_ml import Mudra

m = Mudra()
result = m.run("data.csv")

print(result.task)          # what it decided to do
print(result.metrics)       # how well the chosen model did
print(result.report_path)   # the full explanation, on disk
```

That is the whole loop. `run` accepts a file path (csv, tsv, excel, json, or parquet) or a pandas DataFrame. If you do not name a target column, it looks for a plausible one and tells you what it picked. The report lands next to your script as markdown and HTML.

## Classification

Classification predicts a label, such as whether a customer leaves or stays. Name the column you want predicted:

```python
result = Mudra().run("churn.csv", target="churned")

print(result.metrics["f1"])
print(result.positive_label)   # the class treated as the event of interest
```

For a two-class target the minority class is treated as the positive class, because the rarer outcome (the churn, the fraud, the diagnosis) is usually the one you care about. The report records this choice.

## Regression

Regression predicts a number, such as a price:

```python
result = Mudra().run("houses.csv", target="price")

print(result.metrics["rmse"])   # typical size of the prediction error
print(result.metrics["r2"])     # share of variation the model explains
```

## Clustering

Clustering groups similar rows when there is nothing to predict:

```python
result = Mudra().run("customers.csv", task="clustering")

labels = result.predict(my_dataframe)   # cluster id per row
```

## Steering the run

You can set as much or as little as you want. Anything you do not set is inferred and the report says so.

```python
result = Mudra(random_state=7).run(
    "churn.csv",
    target="churned",
    task="classification",
    metric="f1",
    constraints={"interpretable": True, "max_train_seconds": 120},
)
```

With `interpretable` set, only models a person can read directly are trained, such as logistic regression and a small decision tree.

## What data it accepts

Tables. One row per example, one column per attribute, with a header row. Numeric, categorical, boolean, date, text, and id columns are all recognized and handled. Messy values that real exports produce are repaired on the way in:

- Numbers written as text, with thousands separators, currency symbols, or percent signs, are parsed back to numbers.
- Missing-value spellings that pandas does not catch (`--`, `?`, the word `missing`) are treated as missing. Real categories such as `Unknown` or `Other` are left alone.
- Booleans written as `yes`/`no`, `true`/`false`, `t`/`f`, or 0/1 all work.
- Dates are expanded into year, month, day, and weekday features.
- Id columns carry no signal and are dropped.

Data the library cannot handle stops the run with a `MudraError` that names the column, states the problem, and suggests a fix. You never see a raw pandas or scikit-learn traceback.

## What happens at each step

1. Ingest. The file is read. Delimiter, encoding, and header row are detected.
2. Profile. Each column gets a type (numeric, categorical, boolean, date, text, id) by rule.
3. Goal. Target, task, and metric are taken from you or inferred, in that order of priority.
4. Quality. Constant columns, duplicate rows, class imbalance, and features that look leaked from the target are flagged.
5. Preprocess. Imputation, outlier clipping, encoding, and scaling are fitted on the training split only, so nothing from the test split can leak into the model.
6. Recommend. A documented rule set picks a small shortlist of algorithms that fit the task, the dataset size, the dimensionality, and the sparsity. It never runs every model on every dataset.
7. Tune and select. Each shortlisted model is tuned with a fixed-seed randomized search. The best is chosen by cross-validation alone. The held-out test set is scored once, for the winner, for reporting only.
8. Report. Markdown and HTML, listing every decision and the rule that produced it.

## Models it can train

Classification: logistic regression, decision tree, random forest, extra trees, gradient boosting, support vector classifier, k-nearest neighbors, and gaussian naive bayes. Regression: linear regression, ridge, elastic net, decision tree, random forest, extra trees, gradient boosting, support vector regressor, and k-nearest neighbors. Clustering: k-means with a swept cluster count. XGBoost, LightGBM, and CatBoost join the shortlist when the `boost` extra is installed.

Which of these actually run depends on your data. For example, k-nearest neighbors is only tried when the feature count is modest and the data is dense, and kernel models are only tried when the row count keeps their cost reasonable. The report states which models were shortlisted and why.

## Save, load, and predict on new data

```python
result = Mudra().run("churn.csv", target="churned")

path = result.save("churn_model")        # writes churn_model.joblib + churn_model.json
loaded = Mudra.load("churn_model")       # restores the exact model

preds = loaded.predict(new_dataframe)          # labels
probs = loaded.predict_proba(new_dataframe)    # class probabilities
```

A saved then loaded model gives predictions identical to the original. The `.json` file next to the model records the library version, python version, creation date, task, target, metric, selected model, seed, positive class, and the input schema.

New data is checked against that schema before prediction. A missing column, an unexpected column, a column whose type changed, or a category the model never saw all raise a clear `MudraError` instead of returning silently wrong numbers.

The result object has a stable surface you can build on: `best_model`, `pipeline`, `metrics`, `report_path`, `task`, `target`, `feature_names`, `positive_label`, and `model_path`.

## The decision report

The report is the product as much as the model is. Open `result.report_path` (markdown) or the `.html` file next to it. It contains the goal and which parts of it were inferred, the full decision log by stage, the data-quality findings, every candidate model with its cross-validation score, the winner's held-out metrics next to a naive baseline, the train-versus-test gap, per-class breakdowns and curves for classification, residual diagnostics for regression, and feature importance with its uncertainty.

## Determinism

One seed is threaded through the split, the search, the estimators, and any sampling. Two runs on the same input produce the same model, the same numbers, and byte-identical reports. Set the seed with `Mudra(random_state=...)`.

## Command line

```
mudra-ml run data.csv --target churn --metric f1 --save churn_model
mudra-ml profile data.csv
```

`run` trains and writes the report. `profile` prints the inferred column types, missingness, cardinality, and candidate targets without training anything.

## Scope

Mudra-ML is built for tabular data that fits in memory: the everyday spreadsheet, database extract, or csv export with up to a few hundred thousand rows. It covers binary and multiclass classification, regression, and k-means clustering, end to end, with an audit trail.

## Limits

Know what it does not do, and when not to use it:

- No deep learning. Images, audio, and video are out of scope.
- Text columns are reduced to simple length features. For real language understanding, use a dedicated text pipeline.
- No time-series forecasting. The random split assumes rows are exchangeable, which time series are not.
- Data larger than memory is not supported.
- Do not use the output for decisions that affect people (credit, hiring, medical, legal) without a person reviewing the report, the data quality findings, and the limits of the data. The report flags many problems, and it cannot flag them all.

## License

MIT. See [LICENSE](LICENSE).
