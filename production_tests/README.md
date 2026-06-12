# Production tests

This suite exercises the library against real public datasets, the real
optional boosters, and progressively larger synthetic data. It downloads
datasets from OpenML and scikit-learn on first run, so it needs the network,
and it takes much longer than the offline unit suite. It is run on demand
and is not part of normal CI.

The offline unit suite under `tests/` never depends on anything here.

## Requirements

```
pip install seaborn xgboost lightgbm catboost psutil
```

Downloaded datasets are cached by scikit-learn under the user home
directory, outside this repository. Nothing is written into the repo except
the results file under `.agent/`, which is not tracked.

## Run

```
python -m pytest production_tests -q -p no:warnings
```

Each test appends a JSON line to `.agent/production_results.jsonl` with the
dataset, task, selected model, headline metric, run time, and memory
reading, so partial results survive an interrupted run.
