# Changelog

All notable changes to this project are recorded here. The format follows
Keep a Changelog, and the project uses semantic versioning.

## [0.3.1] - 2026-06-09

### Documentation
- The README now describes the 0.3.0 robustness improvements: messy-numeric
  coercion with preserved categories, boolean-column handling, binary
  classification metrics for any pair of labels with the recorded positive
  class, class-imbalance safety through a stratified split and capped
  cross-validation folds, and clear `MudraError` failures that name the
  offending column. There are no code changes in 0.3.1.

## [0.3.0] - 2026-06-08

This release makes the pipeline survive dirty, real-world tabular data, or
fail with a clear, specific error. It was driven by a battery of messy public
datasets (Titanic, Adult, credit-g, bank-marketing) and a stroke-prediction
replica. The dominant failure the battery surfaced was not ingestion but the
metrics path: a binary target labelled anything other than the integer 1
crashed the run.

### Fixed
- Classification metrics now work for any binary labels, not only the integer
  `1`. Targets labelled `<=50K`/`>50K`, `bad`/`good`, `1`/`2`, `yes`/`no`, and
  similar previously crashed because precision, recall, and f1 assumed
  `pos_label=1`. The positive class is now chosen by a deterministic rule (the
  minority class, since that is usually the event of interest such as stroke,
  churn, or fraud), recorded in the report, and threaded through precision,
  recall, f1, and roc_auc. Multiclass behaviour is unchanged.
- The data-quality outlier check no longer crashes on boolean columns. It was
  running a quantile on a raw boolean series, which numpy rejects (`numpy
  boolean subtract`). Boolean columns are skipped (they have no meaningful
  outliers) and numeric values are cast to float before the quantile.
- Boolean-dtype feature columns no longer crash the imputer. `BooleanToNumeric`
  now runs before `SimpleImputer`, casting bool, string, and numeric forms to a
  0/1 float array and leaving missing entries as NaN, so a raw boolean column is
  converted to numbers before the imputer, which rejects bool dtype, ever sees
  it. Mode imputation still fills the missing entries.

### Added
- Dirty-numeric coercion on the run path. Object columns whose values are
  numbers wearing thousands separators, currency symbols, or percent signs are
  coerced to numeric when at least 90 percent of their non-empty values parse.
  Missing tokens that pandas does not handle by default (`--`, the word
  `missing`, and a bare `?`) are treated as missing. Legitimate categories such
  as `Unknown`, `None`, and `Other` are never turned into missing, and the
  pandas-default tokens (`N/A`, `NA`, `null`, and the rest) are not re-added
  because pandas already treats them as missing at read time. Every coercion is
  logged.
- `MudraError`, with a `DataError` subclass, raised for data the library cannot
  handle. The public run path is wrapped so that any failure surfaces as a
  `MudraError` with a message that names the offending column and suggests a
  fix, never a raw pandas or scikit-learn traceback. A single-class target and a
  class with too few examples to split and cross-validate both stop with a clear
  message.
- The chosen positive class is shown in the report Result section.

### Changed
- Cross-validation folds are capped at the smallest class count, so a severely
  imbalanced but modellable target (for example a five percent positive rate)
  trains without crashing.
- `IngestError` is now a subclass of `MudraError`, so file-read failures are
  caught by the same handling as every other library error.

## [0.2.0] - 2026-06-05

This release overhauls the report and the evaluation so the output earns the
trust of a working data scientist. The public API is unchanged.

### Fixed
- Model selection now uses cross-validation only. The previous selection rule
  picked the candidate with the best held-out test score, which is the
  classic test-set leakage mistake: the test set should be scored once, for
  the single selected model, and used for reporting, not to choose among
  candidates. Selection now follows the highest cross-validation mean across
  folds (lower-is-better metrics like rmse and mae are still picked as
  smallest). The held-out test set is scored once, only for the selected
  model, and its metrics remain the headline numbers in the report. The
  candidates-compared table is ranked by cross-validation and shows CV mean,
  CV std, and a Selected column; the per-candidate test metric column has
  been removed.
- Skewed binary 0/1 features were being collapsed to a constant by IQR
  outlier clipping in the numeric pipeline. A 10/90 binary column has
  Q1 = Q3 = 0, the IQR rule clipped everything to zero, and the column lost
  all signal. Boolean columns now go through a discrete pipeline (mode
  imputation, cast to 0/1 float, no clipping, no scaling) so a binary feature
  that predicts the target retains nonzero permutation importance. Integer
  columns with three to ten distinct values are now classified as categorical
  and one-hot encoded rather than scaled as continuous values, on the same
  principle. The leakage check is unchanged and still flags a feature equal
  to the target. New constant: `DISCRETE_NUMERIC_MAX_UNIQUE = 10`.
- Decision-log entries from the profile, goal, recommend, and quality stages
  were being lost when the caller passed an empty `DecisionLog` because the
  `log or DecisionLog()` pattern treated an empty log as falsy. The check is
  now `log if log is not None else DecisionLog()` so the caller's log is
  always used.

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
