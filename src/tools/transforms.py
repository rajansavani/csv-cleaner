from __future__ import annotations

import re
from typing import Any

import pandas as pd


_NULL_LIKE = {"", "na", "n/a", "null", "none", "nan", "inf", "-inf", "—", "-"}


def normalize_column_names(cols: list[str]) -> tuple[list[str], dict[str, str]]:
    """
    Normalize column names and return (new_cols, rename_map).

    goals:
    - Strip leading/trailing whitespace
    - Collapse repeated spaces
    - Remove obvious mojibake replacement chars (�)
    """
    new_cols: list[str] = []
    rename_map: dict[str, str] = {}

    for c in cols:
        orig = str(c)
        s = orig.strip()
        s = s.replace("�", "")
        s = re.sub(r"\s+", " ", s)

        new_cols.append(s)
        rename_map[orig] = s

    return new_cols, rename_map


def strip_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trim whitespace in all string cells.
    Assumes df was read with dtype=str.
    """
    out = df.copy()
    for col in out.columns:
        out[col] = out[col].astype(str).str.strip()
    return out


def standardize_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert common null-like tokens to empty string.
    We keep "" as missing because the rest of the pipeline treats it as missing.
    """
    out = df.copy()
    for col in out.columns:
        s = out[col].astype(str).str.strip()
        lowered = s.str.lower()
        out[col] = s.where(~lowered.isin(_NULL_LIKE), "")
    return out


def drop_fully_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns where every row is empty after stripping.
    """
    out = df.copy()
    keep_cols: list[str] = []
    for col in out.columns:
        if (out[col].astype(str).str.strip() != "").any():
            keep_cols.append(col)
    return out[keep_cols]


def drop_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop exact duplicate rows.
    """
    return df.drop_duplicates().reset_index(drop=True)


def basic_clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Safe deterministic cleaning pass.

    This version is intentionally conservative:
    - Normalize column names
    - Strip whitespace
    - Standardize null-like tokens
    - Drop fully empty columns
    - Drop exact duplicate rows

    It does NOT try to parse dates or coerce numbers here.
    """
    before_shape = df.shape

    out = df.copy()

    new_cols, rename_map = normalize_column_names([str(c) for c in out.columns])
    out.columns = new_cols

    out = strip_whitespace(out)
    out = standardize_nulls(out)

    # drop empty columns and duplicates
    out = drop_fully_empty_columns(out)
    out = drop_exact_duplicates(out)

    after_shape = out.shape
    after_cols = [str(c) for c in out.columns]

    # true drops are columns that existed after rename but were removed later
    dropped_cols = sorted(list(set(new_cols) - set(after_cols)))

    # only keep real renames (orig != new)
    renamed_cols = {k: v for k, v in rename_map.items() if k != v}

    stats: dict[str, Any] = {
        "before_shape": {"rows": int(before_shape[0]), "columns": int(before_shape[1])},
        "after_shape": {"rows": int(after_shape[0]), "columns": int(after_shape[1])},
        "renamed_columns": renamed_cols,
        "dropped_columns": dropped_cols,
    }

    return out, stats
