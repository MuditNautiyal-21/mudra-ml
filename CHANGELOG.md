# Changelog

All notable changes to this project are recorded here. The format follows
Keep a Changelog, and the project uses semantic versioning.

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
