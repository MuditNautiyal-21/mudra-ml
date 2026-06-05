"""File ingestion with format and dialect auto-detection."""

from __future__ import annotations

import csv
import io
from pathlib import Path

import pandas as pd

_CSV_LIKE = {".csv", ".tsv", ".txt"}
_EXCEL = {".xlsx", ".xls", ".xlsm"}
_JSON = {".json"}
_PARQUET = {".parquet", ".pq"}

_ENCODINGS = ("utf-8", "utf-8-sig", "latin-1")


class IngestError(Exception):
    """Raised when a file cannot be read into a DataFrame."""


def _read_text(path: Path) -> tuple[str, str]:
    """Return the file text and the encoding that decoded it."""
    last_error: Exception | None = None
    for encoding in _ENCODINGS:
        try:
            return path.read_text(encoding=encoding), encoding
        except (UnicodeDecodeError, UnicodeError) as exc:
            last_error = exc
    raise IngestError(
        f"Could not decode {path.name} with any of {_ENCODINGS}."
    ) from last_error


def _sniff_delimiter(sample: str) -> str:
    """Detect a column delimiter from a text sample."""
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        return dialect.delimiter
    except csv.Error:
        counts = {sep: sample.count(sep) for sep in (",", ";", "\t", "|")}
        return max(counts, key=lambda sep: counts[sep])


def _has_header(sample: str, delimiter: str) -> bool:
    try:
        return csv.Sniffer().has_header(sample)
    except csv.Error:
        first = sample.splitlines()[0] if sample.splitlines() else ""
        cells = first.split(delimiter)
        return all(not _looks_numeric(c) for c in cells if c.strip())


def _looks_numeric(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def _load_csv(path: Path) -> pd.DataFrame:
    text, encoding = _read_text(path)
    if not text.strip():
        raise IngestError(f"{path.name} is empty.")
    sample = "\n".join(text.splitlines()[:50])
    delimiter = "\t" if path.suffix.lower() == ".tsv" else _sniff_delimiter(sample)
    header: int | None = 0 if _has_header(sample, delimiter) else None
    frame = pd.read_csv(
        io.StringIO(text),
        sep=delimiter,
        header=header,
        engine="python",
    )
    if header is None:
        frame.columns = [f"column_{i}" for i in range(frame.shape[1])]
    return frame


def _load_excel(path: Path) -> pd.DataFrame:
    try:
        return pd.read_excel(path)
    except ImportError as exc:
        raise IngestError(
            "Reading Excel files needs openpyxl. Install it with "
            "'pip install mudraml[excel]'."
        ) from exc


def _load_json(path: Path) -> pd.DataFrame:
    text, _ = _read_text(path)
    try:
        return pd.read_json(io.StringIO(text))
    except ValueError:
        return pd.read_json(io.StringIO(text), lines=True)


def _load_parquet(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except ImportError as exc:
        raise IngestError(
            "Reading Parquet files needs pyarrow. Install it with "
            "'pip install mudraml[parquet]'."
        ) from exc


def load(path: str | Path) -> pd.DataFrame:
    """Read a data file into a DataFrame, choosing the reader from the suffix.

    Supports csv, tsv, excel, json, and parquet. For delimited text the
    delimiter, encoding, and header row are detected automatically.

    Args:
        path: Path to the data file.

    Returns:
        The loaded DataFrame.

    Raises:
        IngestError: If the file is missing, empty, or in an unsupported format.
    """
    path = Path(path)
    if not path.exists():
        raise IngestError(f"File not found: {path}")
    if path.is_dir():
        raise IngestError(f"Expected a file but got a directory: {path}")

    suffix = path.suffix.lower()
    if suffix in _CSV_LIKE:
        frame = _load_csv(path)
    elif suffix in _EXCEL:
        frame = _load_excel(path)
    elif suffix in _JSON:
        frame = _load_json(path)
    elif suffix in _PARQUET:
        frame = _load_parquet(path)
    else:
        raise IngestError(
            f"Unsupported file type '{suffix}'. Supported: csv, tsv, xlsx, json, parquet."
        )

    if frame.empty:
        raise IngestError(f"{path.name} contains no rows.")
    return frame
