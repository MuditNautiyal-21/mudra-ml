"""Command line interface for MudraML."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from .core import Mudra
from .ingest import load
from .profile import DataProfiler

app = typer.Typer(
    add_completion=False,
    help="Glass-box autonomous data science from the command line.",
    no_args_is_help=True,
)


@app.command()
def run(
    data: str = typer.Argument(..., help="Path to the data file."),
    target: str | None = typer.Option(None, help="Target column to predict."),
    task: str | None = typer.Option(
        None, help="classification, regression, or clustering."
    ),
    metric: str | None = typer.Option(None, help="Metric to optimize."),
    interpretable: bool = typer.Option(
        False, help="Restrict the shortlist to interpretable models."
    ),
    max_train_seconds: int | None = typer.Option(
        None, help="Soft time budget that caps model complexity."
    ),
    output: str = typer.Option("mudraml_report", help="Report path without suffix."),
    save: str | None = typer.Option(None, help="Save the artifact to this path."),
    no_html: bool = typer.Option(False, help="Skip the HTML report."),
) -> None:
    """Run the full pipeline and write a report."""
    constraints: dict[str, object] = {}
    if interpretable:
        constraints["interpretable"] = True
    if max_train_seconds is not None:
        constraints["max_train_seconds"] = max_train_seconds

    mudra = Mudra(verbose=True)
    result = mudra.run(
        data,
        target=target,
        task=task,
        metric=metric,
        constraints=constraints or None,
        report_path=output,
        html=not no_html,
    )

    typer.echo("")
    typer.echo(f"Task: {result.task}")
    typer.echo(f"Selected model: {result.evaluation['best_name']}")
    best = next(
        c for c in result.evaluation["candidates"] if c["name"] == result.evaluation["best_name"]
    )
    for name, value in best["test_metrics"].items():
        if name == "confusion_matrix":
            continue
        typer.echo(f"  {name}: {value:.4f}" if isinstance(value, float) else f"  {name}: {value}")
    typer.echo(f"Report: {result.report_path}")

    if save:
        saved = result.save(save)
        typer.echo(f"Artifact: {saved}")


@app.command()
def profile(
    data: str = typer.Argument(..., help="Path to the data file."),
    as_json: bool = typer.Option(False, "--json", help="Print the profile as JSON."),
) -> None:
    """Profile a dataset and print column types and statistics."""
    frame = load(data)
    profiler = DataProfiler()
    result = profiler.profile(frame)

    if as_json:
        typer.echo(json.dumps(result.as_dict(), indent=2, default=str))
        return

    typer.echo(f"Dataset: {Path(data).name}")
    typer.echo(
        f"Rows: {result.n_rows}  Columns: {result.n_columns}  "
        f"Duplicates: {result.duplicate_rows}"
    )
    typer.echo("")
    header = f"{'column':<24}{'type':<14}{'missing':<10}{'unique':<10}"
    typer.echo(header)
    typer.echo("-" * len(header))
    for col in result.columns.values():
        typer.echo(
            f"{col.name[:23]:<24}{col.inferred_type:<14}"
            f"{col.missing_fraction:<10.2%}{col.n_unique:<10}"
        )
    typer.echo("")
    if result.candidate_targets:
        typer.echo(f"Candidate targets: {', '.join(result.candidate_targets[:3])}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
