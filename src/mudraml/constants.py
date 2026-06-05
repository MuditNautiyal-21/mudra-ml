"""Shared defaults and thresholds.

These values drive the rule-based engine. They are named here so the rules that
use them stay readable and so a single edit changes behavior everywhere.
"""

from __future__ import annotations

DEFAULT_RANDOM_STATE = 42

# Profiling thresholds.
ID_UNIQUE_RATIO = 0.95
CATEGORICAL_MAX_UNIQUE = 20
CATEGORICAL_MAX_RATIO = 0.5
TEXT_MIN_AVG_LENGTH = 25
TEXT_MIN_WORD_COUNT = 3
HIGH_CARDINALITY_THRESHOLD = 30

# Cleaning thresholds.
DEFAULT_MISSING_DROP_THRESHOLD = 0.6
IQR_MULTIPLIER = 1.5
ZSCORE_THRESHOLD = 3.0

# Goal inference thresholds.
CLASSIFICATION_MAX_CLASSES = 20
REGRESSION_MIN_UNIQUE = 20

# Training.
DEFAULT_CV_FOLDS = 5
DEFAULT_SEARCH_ITER = 10
SMALL_DATASET_ROWS = 2000
LARGE_DATASET_ROWS = 50000

# Default metrics per task.
DEFAULT_METRICS = {
    "classification": "f1",
    "regression": "rmse",
    "clustering": "silhouette",
}

VALID_TASKS = ("classification", "regression", "clustering")
