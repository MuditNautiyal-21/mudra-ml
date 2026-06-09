"""The Mudra orchestrator and the RunResult artifact."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .constants import DEFAULT_RANDOM_STATE
from .decisions import DecisionLog, configure_logging
from .errors import DataError, MudraError
from .evaluate import evaluate
from .goal import Goal, infer_goal
from .ingest import coerce_numeric_like, load
from .preprocess import build_pipeline
from .profile import DataProfile, DataProfiler
from .quality import check_quality
from .recommend import recommend_models
from .report import build_context, write_report

_ARTIFACT_VERSION = 1


@dataclass
class RunResult:
    """The output of a run: the fitted model, the report, and the metadata.

    The preprocessing pipeline and the model are kept separate so that
    predictions transform new data the same way the training data was
    transformed.
    """

    best_model: Any
    pipeline: Any
    goal: Goal
    task: str
    metric: str
    report_path: Path
    evaluation: dict[str, Any]
    profile: dict[str, Any]
    feature_names: list[str]

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Transform new rows with the fitted pipeline and predict.

        Args:
            data: New rows with the same feature columns as training.

        Returns:
            Model predictions (labels, values, or cluster ids).
        """
        transformed = self.pipeline.transform(data)
        return self.best_model.predict(transformed)

    def save(self, path: str | Path) -> Path:
        """Persist the pipeline, model, and metadata to one joblib file.

        Args:
            path: Destination path. A .joblib suffix is added if absent.

        Returns:
            The path written.
        """
        path = Path(path)
        if path.suffix != ".joblib":
            path = path.with_suffix(".joblib")
        payload = {
            "version": _ARTIFACT_VERSION,
            "best_model": self.best_model,
            "pipeline": self.pipeline,
            "goal": self.goal.as_dict(),
            "task": self.task,
            "metric": self.metric,
            "evaluation": self.evaluation,
            "profile": self.profile,
            "feature_names": self.feature_names,
        }
        joblib.dump(payload, path)
        return path


class Mudra:
    """Run the full data science workflow and explain every decision.

    Example:
        >>> m = Mudra()
        >>> result = m.run("data.csv")
        >>> preds = result.predict(new_frame)
    """

    def __init__(
        self,
        random_state: int = DEFAULT_RANDOM_STATE,
        verbose: bool = False,
        test_size: float = 0.2,
    ) -> None:
        self.random_state = random_state
        self.test_size = test_size
        self.log = DecisionLog()
        self._loaded_payload: dict[str, Any] | None = None
        if verbose:
            configure_logging()

    def run(
        self,
        data: str | Path | pd.DataFrame,
        target: str | None = None,
        task: str | None = None,
        metric: str | None = None,
        constraints: dict[str, Any] | None = None,
        report_path: str | Path = "mudra_ml_report",
        html: bool = True,
        use_boost: bool = True,
    ) -> RunResult:
        """Ingest, profile, plan, train, evaluate, and report.

        Args:
            data: Path to a data file or an in-memory DataFrame.
            target: Target column, or None to infer.
            task: classification, regression, clustering, or None to infer.
            metric: Metric to optimize, or None for the task default.
            constraints: Optional constraints, for example
                {"interpretable": True, "max_train_seconds": 120}.
            report_path: Where to write the report (without suffix).
            html: Whether to also write an HTML report.
            use_boost: Whether to include xgboost and lightgbm if installed.

        Returns:
            A RunResult with the fitted model and the report path.

        Raises:
            MudraError: If the data cannot be handled. The message names the
                offending column, states the problem, and suggests a fix. No
                raw pandas or scikit-learn traceback reaches the caller.
        """
        try:
            return self._execute(
                data, target, task, metric, constraints, report_path, html, use_boost
            )
        except MudraError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise MudraError(
                f"MudraML could not complete the run: {exc}. Check the target "
                f"column and the column types, then try again."
            ) from exc

    def _execute(
        self,
        data: str | Path | pd.DataFrame,
        target: str | None,
        task: str | None,
        metric: str | None,
        constraints: dict[str, Any] | None,
        report_path: str | Path,
        html: bool,
        use_boost: bool,
    ) -> RunResult:
        """Run the pipeline. Wrapped by run() so failures surface as MudraError."""
        frame, dataset_name = self._as_frame(data)
        self.log = DecisionLog()

        # Repair numeric-like columns read as text (dirty tokens, separators,
        # currency, percent) before profiling sees them.
        frame = coerce_numeric_like(frame, self.log)

        profiler = DataProfiler(self.log)
        profile = profiler.profile(frame)

        operator_goal = Goal(
            target=target,
            task=task,
            metric=metric,
            constraints=constraints or {},
            random_state=self.random_state,
        )
        operator_fields = operator_goal.operator_set_fields()
        goal = infer_goal(profile, operator_goal, self.log)
        # infer_goal always resolves task and metric.
        assert goal.task is not None and goal.metric is not None

        quality = check_quality(frame, profile, goal.target, goal.task, self.log)

        if goal.task == "classification" and goal.target is not None:
            self._check_classification_feasible(frame, goal.target)

        if goal.task == "clustering":
            evaluation = self._run_clustering(frame, profile, goal)
        else:
            evaluation = self._run_supervised(frame, profile, goal)

        ctx = build_context(
            dataset_name=dataset_name,
            n_rows=profile.n_rows,
            n_columns=profile.n_columns,
            goal=goal.as_dict(),
            operator_set_fields=operator_fields,
            log=self.log,
            evaluation=evaluation["evaluation_dict"],
            profile=profile,
            quality=quality,
            frame=frame,
        )
        written = write_report(ctx, report_path, html=html)

        return RunResult(
            best_model=evaluation["result"].best_estimator,
            pipeline=evaluation["pipeline"],
            goal=goal,
            task=goal.task,
            metric=goal.metric,
            report_path=written,
            evaluation=evaluation["evaluation_dict"],
            profile=profile.as_dict(),
            feature_names=evaluation["feature_names"],
        )

    def _run_supervised(
        self, frame: pd.DataFrame, profile: DataProfile, goal: Goal
    ) -> dict[str, Any]:
        target = goal.target
        assert target is not None and goal.task is not None and goal.metric is not None
        clean = frame.dropna(subset=[target])
        X = clean.drop(columns=[target])
        y = clean[target]

        stratify = y if goal.task == "classification" and y.nunique() > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify,
        )
        self.log.record(
            "preprocess",
            f"Split into {len(X_train)} train and {len(X_test)} test rows "
            f"({'stratified' if stratify is not None else 'random'}).",
            "train-test-split",
            {"test_size": self.test_size},
        )

        pipeline, _ = build_pipeline(profile, target, goal.constraints, self.log)
        X_train_t = pipeline.fit_transform(X_train, y_train)
        X_test_t = pipeline.transform(X_test)
        feature_names = self._feature_names(pipeline, X_train_t.shape[1])

        candidates = recommend_models(
            task=goal.task,
            n_rows=len(X_train),
            n_features=X_train_t.shape[1],
            constraints=goal.constraints,
            random_state=self.random_state,
            log=self.log,
            use_boost=goal.constraints.get("interpretable") is not True,
        )

        result = evaluate(
            candidates=candidates,
            task=goal.task,
            metric=goal.metric,
            feature_names=feature_names,
            X_train=X_train_t,
            y_train=y_train.to_numpy(),
            X_test=X_test_t,
            y_test=y_test.to_numpy(),
            random_state=self.random_state,
            log=self.log,
        )
        eval_dict = result.as_dict()
        eval_dict["feature_importance"] = self._named_importance(
            result.feature_importance, feature_names
        )
        return {
            "result": result,
            "pipeline": pipeline,
            "evaluation_dict": eval_dict,
            "feature_names": feature_names,
        }

    def _run_clustering(
        self, frame: pd.DataFrame, profile: DataProfile, goal: Goal
    ) -> dict[str, Any]:
        assert goal.metric is not None
        pipeline, _ = build_pipeline(profile, None, goal.constraints, self.log)
        X_t = pipeline.fit_transform(frame)
        feature_names = self._feature_names(pipeline, X_t.shape[1])

        candidates = recommend_models(
            task="clustering",
            n_rows=len(frame),
            n_features=X_t.shape[1],
            constraints=goal.constraints,
            random_state=self.random_state,
            log=self.log,
        )
        result = evaluate(
            candidates=candidates,
            task="clustering",
            metric=goal.metric,
            feature_names=feature_names,
            X_train=X_t,
            random_state=self.random_state,
            log=self.log,
        )
        return {
            "result": result,
            "pipeline": pipeline,
            "evaluation_dict": result.as_dict(),
            "feature_names": feature_names,
        }

    @staticmethod
    def _feature_names(pipeline: Any, n_features: int) -> list[str]:
        try:
            names = pipeline.named_steps["columns"].get_feature_names_out()
            return [str(n) for n in names]
        except (AttributeError, KeyError, ValueError):
            return [f"feature_{i}" for i in range(n_features)]

    @staticmethod
    def _named_importance(
        importance: dict[str, float], feature_names: list[str]
    ) -> dict[str, float]:
        return importance

    @staticmethod
    def _check_classification_feasible(frame: pd.DataFrame, target: str) -> None:
        """Stop with a clear message when a target cannot be classified.

        A single-class target cannot train a classifier, and a class with a
        single example cannot be put on both sides of a split or into
        cross-validation. Both raise a DataError naming the target rather than
        letting scikit-learn raise a bare traceback.
        """
        counts = frame[target].dropna().value_counts()
        if len(counts) < 2:
            only = counts.index[0] if len(counts) else "none"
            raise DataError(
                f"Target '{target}' has only one class ('{only}'). A classifier "
                f"needs at least two classes. Add rows for the other class or "
                f"reframe the problem."
            )
        rarest_label = counts.index[-1]
        rarest = int(counts.iloc[-1])
        if rarest < 2:
            raise DataError(
                f"Target '{target}' has a class ('{rarest_label}') with only "
                f"{rarest} example. That is too few to split and cross-validate. "
                f"Collect more examples of that class or merge it into another."
            )

    @staticmethod
    def _as_frame(data: str | Path | pd.DataFrame) -> tuple[pd.DataFrame, str]:
        if isinstance(data, pd.DataFrame):
            return data.copy(), "in-memory DataFrame"
        return load(data), Path(data).name

    @classmethod
    def load(cls, path: str | Path) -> RunResult:
        """Load a saved artifact and return a RunResult ready to predict.

        Args:
            path: Path to a .joblib artifact written by RunResult.save.

        Returns:
            A RunResult with the fitted pipeline and model.
        """
        path = Path(path)
        if path.suffix != ".joblib":
            path = path.with_suffix(".joblib")
        payload = joblib.load(path)
        goal = Goal(**payload["goal"])
        return RunResult(
            best_model=payload["best_model"],
            pipeline=payload["pipeline"],
            goal=goal,
            task=payload["task"],
            metric=payload["metric"],
            report_path=Path("loaded-artifact"),
            evaluation=payload["evaluation"],
            profile=payload["profile"],
            feature_names=payload["feature_names"],
        )
