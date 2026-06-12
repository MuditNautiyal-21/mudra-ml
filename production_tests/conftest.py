"""Shared fixtures for the production test pass.

Every test appends one JSON line per result to .agent/production_results.jsonl
so partial results survive an interrupted run. The file is reset at session
start.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

RESULTS_PATH = Path(__file__).resolve().parents[1] / ".agent" / "production_results.jsonl"


def pytest_sessionstart(session: pytest.Session) -> None:
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text("", encoding="utf-8")


def peak_memory_mb() -> float | None:
    """Peak working-set size of this process in megabytes, when measurable."""
    try:
        import psutil

        info = psutil.Process().memory_info()
        peak = getattr(info, "peak_wset", None) or info.rss
        return round(peak / (1024 * 1024), 1)
    except Exception:  # noqa: BLE001
        return None


@pytest.fixture(scope="session")
def record() -> Callable[[dict[str, Any]], None]:
    def _record(row: dict[str, Any]) -> None:
        with RESULTS_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=str) + "\n")

    return _record
