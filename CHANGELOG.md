# Changelog

All notable changes to this project are recorded here. The format follows
Keep a Changelog, and the project uses semantic versioning.

## [0.2.0] - 2026-06-05

This release overhauls the report and the evaluation so the output earns the
trust of a working data scientist. The public API is unchanged.

### Added
- Trust summary in every report: a naive baseline (most-frequent class for
  classification, mean for regression) scored alongside the selected model,
  cross-validation reported as mean plus or minus standard deviation across
  folds, train metrics next to test metrics so the overfitting gap is visible,
  and a small-sample warning when the held-out set is too small to trust.
- Diagnostic depth for classification: per-class precision, recall, F1, and
  support, the confusion matrix, and for binary tasks the ROC and
  precision-recall curves with their AUC and average precision.
- Diagnostic depth for regression: residual mean, std, mean absolute error,
  and max absolute residual.
- Permutation importance with its standard deviation, alongside impurity or
  coefficient importance, with a note that impurity importance is biased
  toward high-cardinality features.
- HTML report charts embedded as base64 PNGs: confusion matrix heatmap, ROC
  curve, precision-recall curve, feature importance bars, target distribution,
  feature correlation heatmap, residual plot, and predicted-versus-actual
  plot. Chart rendering uses the matplotlib Agg backend so it works headless.
- Data-quality section: class balance, missingness, cardinality, and outlier
  counts. Warnings for constant columns, duplicate rows, high-cardinality
  categoricals, all-missing columns, single-class targets, class imbalance,
  missing target values, and leakage suspects (features highly correlated with
  the target or equal to the target).
- Limitations and next-steps section that turns the warnings into concrete
  recommendations.
- New module `mudra_ml.quality` with `check_quality`, `QualityReport`, and
  `QualityWarning`.
- New module `mudra_ml.plots` with the chart-rendering helpers and graceful
  degradation when an individual chart cannot be produced.
- Stress-test battery covering tiny, single-feature, single-class,
  all-missing, constant, duplicate, wide, id-like high-cardinality,
  imbalanced, mixed-dtype, leakage-injected, missing-target, and 10k-row
  datasets across binary classification, multiclass classification,
  regression, and clustering.

### Changed
- `matplotlib` is now a runtime dependency.
- Report renderers gain new sections; existing sections keep their names and
  formats. The markdown report stays text and tables. The richer visuals live
  in the HTML report.

### Dependencies
- Added `matplotlib>=3.7` to the core dependency list.

## [0.1.0] - 2026-06-04

First public release.

### Added
- File ingestion for csv, tsv, excel, json, and parquet, with delimiter,
  encoding, and header auto-detection.
- Data profiler with per-column type inference (numeric, categorical,
  datetime, boolean, id, text), missingness, cardinality, distribution stats,
  and candidate-target ranking.
- Goal object and rule-based goal inference for task, target, and metric, with
  operator-set fields taking precedence over inference.
- Leakage-safe cleaning and preprocessing as a scikit-learn Pipeline and
  ColumnTransformer: statistical imputation, datetime part extraction, IQR or
  z-score outlier clipping, one-hot and frequency encoding, and scaling.
- Rule-based algorithm recommendation keyed on task, dataset size, feature
  count, cardinality, and operator constraints.
- Cross-validated training and tuning with RandomizedSearchCV at a fixed seed,
  task-appropriate evaluation, best-model selection, and feature importance.
- Markdown and HTML run reports that log every decision and the rule behind it.
- Model and pipeline persistence through joblib, with a predict path on loaded
  artifacts.
- Command line interface with `run` and `profile`.
- Optional xgboost and lightgbm candidates through the `boost` extra.
