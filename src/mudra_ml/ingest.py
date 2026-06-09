"""File ingestion with format and dialect auto-detection."""

from __future__ import annotations

import csv
import io
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .decisions import DecisionLog
from .errors import MudraError

_CSV_LIKE = {".csv", ".tsv", ".txt"}
_EXCEL = {".xlsx", ".xls", ".xlsm"}
_JSON = {".json"}
_PARQUET = {".parquet", ".pq"}

_ENCODINGS = ("utf-8", "utf-8-sig", "latin-1")

# Missing tokens that pandas does NOT treat as missing by default. The pandas
# defaults (N/A, NA, n/a, null, none, empty, and more) are already handled at
# read time, so they are deliberately not repeated here.
_EXTRA_MISSING_TOKENS = frozenset({"--", "?", "missing"})

# Legitimate category values that must never be turned into missing.
_PROTECTED_CATEGORIES = frozenset({"unknown", "none", "other"})

# Thousands separators, currency symbols, percent signs, and whitespace are
# stripped before a numeric parse is attempted.
_STRIP_PATTERN = re.compile(r"[,$%£€\s]")

# A column is coerced to numeric only when at least this fraction of its
# non-empty, non-missing-token values parse as numbers after stripping.
_COERCE_PARSE_THRESHOLD = 0.90


class IngestError(MudraError):
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
            "'pip install mudra-ml[excel]'."
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
            "'pip install mudra-ml[parquet]'."
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


def _strip_symbols(text: str) -> str:
    return _STRIP_PATTERN.sub("", text)


def _coerce_cell(value: Any) -> Any:
    """Map one cell to a numeric-ready string or NaN."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return np.nan
    text = str(value).strip()
    if text.lower() in _EXTRA_MISSING_TOKENS:
        return np.nan
    return _strip_symbols(text)


def _try_numeric_coercion(series: pd.Series) -> pd.Series | None:
    """Return a numeric version of a numeric-like object column, or None.

    A column is coerced only when, after dropping the extra missing tokens and
    stripping separators and symbols, at least 90 percent of the remaining
    non-empty values parse as numbers. Columns that carry a legitimate
    ``Unknown``, ``None``, or ``Other`` category are left untouched so those
    values are never silently turned into missing.
    """
    non_null = series.dropna()
    if non_null.empty:
        return None
    stripped_text = non_null.astype(str).str.strip()
    is_token = stripped_text.str.lower().isin(_EXTRA_MISSING_TOKENS)
    candidates = stripped_text[~is_token]
    if candidates.empty:
        return None
    parsed = pd.to_numeric(candidates.map(_strip_symbols), errors="coerce")
    if float(parsed.notna().mean()) < _COERCE_PARSE_THRESHOLD:
        return None
    non_parseable = candidates[parsed.isna()]
    if non_parseable.str.lower().isin(_PROTECTED_CATEGORIES).any():
        return None
    return pd.to_numeric(series.map(_coerce_cell), errors="coerce")


def coerce_numeric_like(
    frame: pd.DataFrame, log: DecisionLog | None = None
) -> pd.DataFrame:
    """Coerce numeric-like object columns to numeric, logging each coercion.

    Object columns whose values are numbers wearing thousands separators,
    currency symbols, percent signs, or non-default missing tokens (``--``,
    ``missing``, a bare ``?``) are read by pandas as text, which breaks the
    numeric path. This pass repairs them. It is order-independent: every column
    is judged on its own values, so two runs on the same frame are identical.

    Args:
        frame: The loaded frame.
        log: Optional decision log to record each coercion.

    Returns:
        A new frame with numeric-like columns coerced.
    """
    frame = frame.copy()
    for name in frame.columns:
        series = frame[name]
        if not (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
        ):
            continue
        coerced = _try_numeric_coercion(series)
        if coerced is None:
            continue
        added_missing = int(coerced.isna().sum() - series.isna().sum())
        frame[name] = coerced
        if log is not None:
            log.record(
                "ingest",
                f"Column '{name}': coerced to numeric after stripping separators "
                f"and symbols and treating non-default missing tokens as missing. "
                f"{added_missing} value(s) set to missing.",
                "dirty-numeric-coercion",
                {"column": str(name), "values_set_missing": added_missing},
            )
    return frame
