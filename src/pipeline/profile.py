from __future__ import annotations

import csv
from io import StringIO
from typing import Any, Iterable

import pandas as pd
from fastapi import HTTPException, UploadFile


def _sniff_delimiter(sample_text: str) -> str | None:
    """
    Try to infer delimiter from sample text.
    Returns delimiter char or none if sniffing fails.
    """
    try:
        dialect = csv.Sniffer().sniff(sample_text, delimiters=[",", ";", "\t", "|"])
        return dialect.delimiter
    except Exception:
        return None


def _try_read_with_delimiters(text: str, delimiters: Iterable[str]) -> pd.DataFrame:
    """
    Try reading with candidate delimiters; return the first that parses cleanly.
    """
    last_err: Exception | None = None

    for delim in delimiters:
        try:
            df = pd.read_csv(
                StringIO(text),
                sep=delim,
                dtype=str,             # keep raw strings for cleaning later
                keep_default_na=False, # keep empty strings as empty (standardize later)
            )

            # if it became a single column, delimiter is probably wrong
            if df.shape[1] <= 1 and delim != ",":
                continue

            return df
        except Exception as e:
            last_err = e

    raise ValueError(f"failed to read csv with tried delimiters: {last_err}")


def read_uploaded_csv(upload: UploadFile) -> pd.DataFrame:
    """
    Read a user-uploaded csv into a dataframe, handling common messy cases:
    - Odd encodings
    - Delimiter variations (comma/semicolon/tab/pipe)

    We intentionally read everything as strings here to preserve raw values.
    """
    try:
        raw = upload.file.read()
        if not raw:
            raise ValueError("empty file")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"failed to read upload: {e}")

    encodings_to_try = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
    last_err: Exception | None = None

    for enc in encodings_to_try:
        try:
            text = raw.decode(enc, errors="replace")

            sample = text[:50_000]
            sniffed = _sniff_delimiter(sample)

            delimiters = [sniffed] if sniffed else []
            delimiters += [",", ";", "\t", "|"]

            # remove duplicates while keeping order
            seen: set[str] = set()
            delimiters = [d for d in delimiters if d and not (d in seen or seen.add(d))]

            return _try_read_with_delimiters(text, delimiters)

        except Exception as e:
            last_err = e

    raise HTTPException(status_code=400, detail=f"could not parse csv: {last_err}")


def profile_dataframe(df: pd.DataFrame, filename: str | None = None) -> dict[str, Any]:
    """
    Basic profiling stats used by /profile and /clean.

    We treat empty strings as missing here (since keep_default_na=False).
    """
    if df.shape[1] == 0:
        raise ValueError("dataframe has no columns")

    n_rows, n_cols = df.shape

    missing_by_col: dict[str, dict[str, float | int]] = {}
    for col in df.columns:
        series = df[col].astype(str)
        missing_count = int((series.str.strip() == "").sum())
        missing_by_col[str(col)] = {
            "missing_count": missing_count,
            "missing_pct": float(missing_count / max(n_rows, 1)),
        }

    duplicate_rows = int(df.duplicated().sum())
    preview = df.head(10).fillna("").to_dict(orient="records")

    payload: dict[str, Any] = {
        "shape": {"rows": int(n_rows), "columns": int(n_cols)},
        "columns": [str(c) for c in df.columns],
        "missing_by_column": missing_by_col,
        "duplicate_row_count": duplicate_rows,
        "preview_rows": preview,
    }

    if filename is not None:
        payload["filename"] = filename

    return payload
