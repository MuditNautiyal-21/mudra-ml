from __future__ import annotations

from typer.testing import CliRunner

from mudraml.cli import app

runner = CliRunner()


def test_cli_profile(tmp_path, classification_frame):
    path = tmp_path / "data.csv"
    classification_frame.to_csv(path, index=False)
    result = runner.invoke(app, ["profile", str(path)])
    assert result.exit_code == 0
    assert "Candidate targets" in result.stdout


def test_cli_profile_json(tmp_path, regression_frame):
    path = tmp_path / "data.csv"
    regression_frame.to_csv(path, index=False)
    result = runner.invoke(app, ["profile", str(path), "--json"])
    assert result.exit_code == 0
    assert '"n_rows"' in result.stdout


def test_cli_run_classification(tmp_path, classification_frame):
    path = tmp_path / "data.csv"
    classification_frame.to_csv(path, index=False)
    out = tmp_path / "report"
    result = runner.invoke(
        app,
        ["run", str(path), "--target", "target", "--task", "classification",
         "--metric", "f1", "--output", str(out)],
    )
    assert result.exit_code == 0
    assert "Selected model" in result.stdout
    assert out.with_suffix(".md").exists()


def test_cli_run_with_save(tmp_path, regression_frame):
    path = tmp_path / "data.csv"
    regression_frame.to_csv(path, index=False)
    result = runner.invoke(
        app,
        ["run", str(path), "--target", "price", "--output", str(tmp_path / "r"),
         "--save", str(tmp_path / "model"), "--no-html"],
    )
    assert result.exit_code == 0
    assert (tmp_path / "model.joblib").exists()


def test_cli_run_interpretable(tmp_path, classification_frame):
    path = tmp_path / "data.csv"
    classification_frame.to_csv(path, index=False)
    result = runner.invoke(
        app,
        ["run", str(path), "--target", "target", "--interpretable",
         "--output", str(tmp_path / "r"), "--no-html"],
    )
    assert result.exit_code == 0
