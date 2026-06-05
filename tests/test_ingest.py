from __future__ import annotations

import json

import pytest

from mudra_ml.ingest import IngestError, load


def test_load_csv(tmp_path, classification_frame):
    path = tmp_path / "data.csv"
    classification_frame.to_csv(path, index=False)
    loaded = load(path)
    assert loaded.shape == classification_frame.shape


def test_load_csv_semicolon_delimiter(tmp_path):
    path = tmp_path / "semi.csv"
    path.write_text("a;b;c\n1;2;3\n4;5;6\n", encoding="utf-8")
    loaded = load(path)
    assert list(loaded.columns) == ["a", "b", "c"]
    assert loaded.shape == (2, 3)


def test_load_csv_no_header(tmp_path):
    path = tmp_path / "noheader.csv"
    path.write_text("1,2,3\n4,5,6\n7,8,9\n", encoding="utf-8")
    loaded = load(path)
    assert list(loaded.columns) == ["column_0", "column_1", "column_2"]


def test_load_tsv(tmp_path):
    path = tmp_path / "data.tsv"
    path.write_text("a\tb\n1\t2\n3\t4\n", encoding="utf-8")
    loaded = load(path)
    assert list(loaded.columns) == ["a", "b"]


def test_load_json_records(tmp_path):
    path = tmp_path / "data.json"
    records = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
    path.write_text(json.dumps(records), encoding="utf-8")
    loaded = load(path)
    assert loaded.shape == (2, 2)


def test_load_json_lines(tmp_path):
    path = tmp_path / "data.json"
    path.write_text('{"a": 1}\n{"a": 2}\n', encoding="utf-8")
    loaded = load(path)
    assert loaded.shape == (2, 1)


def test_load_excel(tmp_path, regression_frame):
    pytest.importorskip("openpyxl")
    path = tmp_path / "data.xlsx"
    regression_frame.to_excel(path, index=False)
    loaded = load(path)
    assert loaded.shape == regression_frame.shape


def test_load_parquet(tmp_path, regression_frame):
    pytest.importorskip("pyarrow")
    path = tmp_path / "data.parquet"
    regression_frame.to_parquet(path)
    loaded = load(path)
    assert loaded.shape == regression_frame.shape


def test_missing_file_raises():
    with pytest.raises(IngestError, match="not found"):
        load("does_not_exist.csv")


def test_unsupported_type_raises(tmp_path):
    path = tmp_path / "data.xyz"
    path.write_text("nonsense", encoding="utf-8")
    with pytest.raises(IngestError, match="Unsupported"):
        load(path)


def test_empty_file_raises(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")
    with pytest.raises(IngestError):
        load(path)


def test_directory_raises(tmp_path):
    with pytest.raises(IngestError, match="directory"):
        load(tmp_path)


def test_latin1_encoding_fallback(tmp_path):
    path = tmp_path / "latin.csv"
    rows = "name,city\n" + "".join(f"Jos\xe9{i},M\xe1laga\n" for i in range(10))
    path.write_bytes(rows.encode("latin-1"))
    loaded = load(path)
    assert loaded.shape == (10, 2)
    assert "city" in loaded.columns


def test_pipe_delimiter(tmp_path):
    path = tmp_path / "pipe.csv"
    path.write_text("a|b|c\n1|2|3\n", encoding="utf-8")
    loaded = load(path)
    assert list(loaded.columns) == ["a", "b", "c"]


def test_utf8_bom(tmp_path):
    path = tmp_path / "bom.csv"
    path.write_bytes("﻿a,b\n1,2\n".encode())
    loaded = load(path)
    assert "a" in loaded.columns


def test_load_accepts_string_path(tmp_path, regression_frame):
    path = tmp_path / "data.csv"
    regression_frame.to_csv(path, index=False)
    loaded = load(str(path))
    assert loaded.shape == regression_frame.shape
