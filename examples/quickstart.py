"""Run MudraML on three task types using scikit-learn bundled data.

This writes a CSV for each task to a temporary folder, runs the pipeline, and
prints the selected model and where the report was written.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from sklearn.datasets import fetch_california_housing, load_breast_cancer, load_iris

from mudra_ml import Mudra


def main() -> None:
    out = Path(tempfile.mkdtemp(prefix="mudra_ml_example_"))

    classification = load_breast_cancer(as_frame=True).frame
    clf_path = out / "breast_cancer.csv"
    classification.to_csv(clf_path, index=False)
    clf = Mudra().run(clf_path, target="target", report_path=out / "classification")
    print(f"classification: {clf.evaluation['best_name']} -> {clf.report_path}")

    regression = fetch_california_housing(as_frame=True).frame.sample(1000, random_state=0)
    reg_path = out / "housing.csv"
    regression.to_csv(reg_path, index=False)
    reg = Mudra().run(reg_path, target="MedHouseVal", report_path=out / "regression")
    print(f"regression: {reg.evaluation['best_name']} -> {reg.report_path}")

    clustering = load_iris(as_frame=True).frame.drop(columns=["target"])
    clu_path = out / "iris.csv"
    clustering.to_csv(clu_path, index=False)
    clu = Mudra().run(clu_path, task="clustering", report_path=out / "clustering")
    print(f"clustering: {clu.evaluation['best_name']} -> {clu.report_path}")


if __name__ == "__main__":
    main()
