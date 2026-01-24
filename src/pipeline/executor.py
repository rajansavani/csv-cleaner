from __future__ import annotations

from typing import Any

import pandas as pd

from src.llm.schemas import CleaningPlan


class ExecutionError(RuntimeError):
    pass


def execute_plan(df: pd.DataFrame, plan: CleaningPlan) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Apply a cleaning plan to a dataframe and return (cleaned_df, execution_report).

    Notes:
    - We keep data mostly as strings to avoid surprises
    - Parsing actions normalize values into consistent string formats
    - Validations run at the end and report pass/fail counts
    """
    out = df.copy()
    report: dict[str, Any] = {
        "plan_version": plan.version,
        "summary": plan.summary,
        "actions_applied": [],
        "warnings": [],
        "validations": {},
    }

    for action in plan.actions:
        kind = action.action

        if kind == "trim_whitespace":
            before_cols = list(out.columns)
            out = _trim_whitespace(out, action.columns)
            report["actions_applied"].append({"action": kind, "columns": action.columns, "columns_seen": before_cols})

        elif kind == "standardize_nulls":
            out = _standardize_nulls(out, action.null_tokens)
            report["actions_applied"].append({"action": kind, "null_tokens": action.null_tokens})

        elif kind == "rename_columns":
            out, did = _rename_columns(out, action.mapping)
            report["actions_applied"].append({"action": kind, "mapping": action.mapping, "applied": did})

        elif kind == "drop_columns":
            out, dropped = _drop_columns(out, action.columns)
            report["actions_applied"].append({"action": kind, "columns": action.columns, "dropped": dropped})

        elif kind == "parse_numeric":
            out, per_col = _parse_numeric(
                out,
                columns=action.columns,
                numeric_type=action.numeric_type,
                allow_currency=action.allow_currency,
                allow_thousands=action.allow_thousands_separators,
                fix_typos=action.fix_common_typos,
            )
            report["actions_applied"].append({"action": kind, "columns": action.columns, "stats": per_col})

        elif kind == "parse_dates":
            out, per_col = _parse_dates(
                out,
                columns=action.columns,
                day_first=action.day_first,
                output_format=action.output_format,
            )
            report["actions_applied"].append({"action": kind, "columns": action.columns, "stats": per_col})

        elif kind == "deduplicate_rows":
            before_rows = int(out.shape[0])
            out = out.drop_duplicates(subset=action.subset).reset_index(drop=True)
            after_rows = int(out.shape[0])
            report["actions_applied"].append(
                {"action": kind, "subset": action.subset, "dropped_rows": before_rows - after_rows}
            )

        else:
            raise ExecutionError(f"unknown action: {kind}")

    report["validations"] = _run_validations(out, plan)
    return out, report


def _trim_whitespace(df: pd.DataFrame, columns: list[str] | None) -> pd.DataFrame:
    out = df.copy()
    cols = columns if columns is not None else list(out.columns)
    for col in cols:
        if col in out.columns:
            out[col] = out[col].astype(str).str.strip()
    return out

def _standardize_nulls(df: pd.DataFrame, null_tokens: list[str]) -> pd.DataFrame:
    out = df.copy()
    token_set = {t.strip().lower() for t in null_tokens}

    for col in out.columns:
        s = out[col].astype(str).str.strip()
        lowered = s.str.lower()
        out[col] = s.where(~lowered.isin(token_set), "")
    
    return out

def _rename_columns(df: pd.DataFrame, mapping: dict[str, str]) -> tuple[pd.DataFrame, dict[str, str]]:
    out = df.copy()
    applied: dict[str, str] = {}

    for old, new in mapping.items():
        if old in out.columns and new:
            applied[old] = new
    
    if applied:
        out = out.rename(columns=applied)
    
    return out, applied

def _drop_columns(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    to_drop = [c for c in columns if c in out.columns]
    if to_drop:
        out = out.drop(columns=to_drop)
    return out, to_drop


def _parse_numeric(
    df: pd.DataFrame,
    *,
    columns: list[str],
    numeric_type: str,
    allow_currency: bool,
    allow_thousands: bool,
    fix_typos: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = df.copy()
    stats: dict[str, Any] = {}

    for col in columns:
        if col not in out.columns:
            stats[col] = {"skipped": True, "reason": "missing column"}
            continue

        before = out[col].astype(str).fillna("")
        cleaned = before.str.strip()

        if fix_typos:
            cleaned = cleaned.str.replace("o", "0", regex=False).str.replace("O", "0", regex=False)

        if allow_currency:
            # remove common currency symbols/words, keep separators for next steps
            cleaned = cleaned.str.replace(r"[\$\€\£\¥]", "", regex=True)

        if allow_thousands:
            # allow formats like 1,234 or 1.234.567 (we just strip separators)
            cleaned = cleaned.str.replace(r"[,\s]", "", regex=True)
            # european-ish: 2.278.845 -> 2278845 (strip dots when they look like separators)
            cleaned = cleaned.str.replace(r"(?<=\d)\.(?=\d{3}\b)", "", regex=True)

        # now keep digits/dot/negative only
        cleaned = cleaned.str.replace(r"[^0-9\.\-]", "", regex=True)

        # normalize weird multiple dots: 8..8 -> 8.8
        cleaned = cleaned.str.replace(r"\.{2,}", ".", regex=True)

        # turn empty into ""
        cleaned = cleaned.where(cleaned.str.len() > 0, "")

        if numeric_type == "int":
            # remove decimal part if present
            cleaned = cleaned.str.replace(r"\..*$", "", regex=True)
            cleaned = cleaned.where(cleaned.str.len() > 0, "")
        else:
            # handle trailing dot: 9. -> 9.0
            cleaned = cleaned.str.replace(r"^(\-?\d+)\.$", r"\1.0", regex=True)

            def _to_float_str(x: str) -> str:
                if x == "":
                    return ""
                try:
                    return str(float(x))
                except Exception:
                    return ""

            cleaned = cleaned.apply(_to_float_str)

        changed = int((before != cleaned).sum())
        stats[col] = {"changed_cells": changed}

        out[col] = cleaned

    return out, stats


def _parse_dates(df: pd.DataFrame, columns: list[str], day_first: bool = False):
    out = df.copy()
    per_col: dict[str, dict] = {}

    for col in columns:
        if col not in out.columns:
            per_col[col] = {"status": "skipped", "reason": "missing column"}
            continue

        before = out[col]

        # normalize empties to NA
        cleaned = before.replace("", pd.NA)

        # pandas removed infer_datetime_format in newer versions
        try:
            dt = pd.to_datetime(
                cleaned,
                errors="coerce",
                dayfirst=day_first,
                infer_datetime_format=True,  # older pandas only
            )
        except TypeError:
            dt = pd.to_datetime(
                cleaned,
                errors="coerce",
                dayfirst=day_first,
            )

        out[col] = dt

        per_col[col] = {
            "status": "ok",
            "parsed_non_null": int(dt.notna().sum()),
            "total": int(len(dt)),
        }

    return out, per_col

def _run_validations(df: pd.DataFrame, plan: CleaningPlan) -> dict[str, Any]:
    """
    Run simple validations and return a structured summary.
    """
    results: dict[str, Any] = {"required": [], "ranges": [], "enums": []}

    # required columns
    for rule in plan.validations.required:
        ok = rule.column in df.columns
        results["required"].append({"column": rule.column, "ok": ok})

    # ranges (attempt float conversion on non-empty values)
    for rule in plan.validations.ranges:
        if rule.column not in df.columns:
            results["ranges"].append({"column": rule.column, "ok": False, "reason": "missing column"})
            continue

        s = df[rule.column].astype(str).str.strip()
        s = s.where(s != "", pd.NA)

        numeric = pd.to_numeric(s, errors="coerce")
        total = int(numeric.notna().sum())

        if total == 0:
            results["ranges"].append({"column": rule.column, "ok": True, "checked": 0, "violations": 0})
            continue

        violations = 0
        if rule.min is not None:
            violations += int((numeric < rule.min).sum())
        if rule.max is not None:
            violations += int((numeric > rule.max).sum())

        results["ranges"].append(
            {
                "column": rule.column,
                "ok": violations == 0,
                "checked": total,
                "violations": violations,
                "min": rule.min,
                "max": rule.max,
            }
        )

    # enums (check non-empty values are subset of allowed)
    for rule in plan.validations.enums:
        if rule.column not in df.columns:
            results["enums"].append({"column": rule.column, "ok": False, "reason": "missing column"})
            continue

        allowed = set(rule.allowed)
        vals = df[rule.column].astype(str).str.strip()
        vals = vals[vals != ""]

        bad = sorted({v for v in vals.unique().tolist() if v not in allowed})
        results["enums"].append(
            {
                "column": rule.column,
                "ok": len(bad) == 0,
                "bad_values": bad[:50],
                "bad_value_count": len(bad),
                "allowed_count": len(allowed),
            }
        )

    return results