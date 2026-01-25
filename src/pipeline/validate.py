from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.llm.schemas import CleaningPlan

@dataclass
class PlanValidationResult:
    ok: bool
    errors: list[dict[str, Any]]
    warnings: list[dict[str, Any]]
    final_columns: list[str] | None = None

class PlanValidationError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        errors: list[dict[str, Any]] | None = None,
        warnings: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(message)
        self.errors: list[dict[str, Any]] = errors or []
        self.warnings: list[dict[str, Any]] = warnings or []

def ensure_valid_plan(plan: CleaningPlan, *, df_columns: list[str] | None = None) -> PlanValidationResult:
    """
    Validate a plan and raise if it contains semantic errors.
    """
    result = validate_plan(plan, df_columns=df_columns)
    if not result.ok:
        raise PlanValidationError(
            "plan validation failed",
            errors=result.errors,
            warnings=result.warnings,
        )
    return result

def validate_plan(plan: CleaningPlan, *, df_columns: list[str] | None = None) -> PlanValidationResult:
    """
    Validate plan semantics (not schema), returning errors + warnings.
    - Errors are things that should stop execution
    - Warnings are issues that might cause partial / no-ops but are still executable
    """
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    if not isinstance(plan.version, str) or not plan.version.strip():
        errors.append({"path": "version", "message": "Version must be a non-empty string"})
    
    if not isinstance(plan.summary, str):
        errors.append({"path": "summary", "message": "Summary must be a string"})

    # validate validation spec
    _validate_validations(plan, errors, warnings)

    # if df columns are provided, simulate column set as actions apply
    final_cols: list[str] | None = None
    if df_columns is not None:
        final_cols = _validate_actions_against_columns(plan, df_columns, errors, warnings)

    ok = len(errors) == 0
    return PlanValidationResult(ok=ok, errors=errors, warnings=warnings, final_columns=final_cols)


def _validate_validations(plan: CleaningPlan, errors: list[dict[str, Any]], warnings: list[dict[str, Any]]) -> None:
    required_cols = [r.column for r in plan.validations.required]
    dup_required = _find_dupes(required_cols)
    if dup_required:
        warnings.append(
            {
                "path": "validations.required",
                "message": f"Duplicate required columns: {sorted(dup_required)}",
            }
        )

    # range rules: min/max must be consistent
    for i, rr in enumerate(plan.validations.ranges):
        if rr.min is not None and rr.max is not None and rr.min > rr.max:
            errors.append(
                {
                    "path": f"validations.ranges[{i}]",
                    "message": f"Min ({rr.min}) cannot be greater than max ({rr.max}) for column '{rr.column}'",
                }
            )

    # enum rules: allowed cannot be empty and should not contain empty strings
    for i, er in enumerate(plan.validations.enums):
        if not er.allowed:
            errors.append(
                {
                    "path": f"validations.enums[{i}].allowed",
                    "message": f"Allowed list cannot be empty for column '{er.column}'",
                }
            )
        if any((not isinstance(v, str) or v.strip() == "") for v in er.allowed):
            warnings.append(
                {
                    "path": f"validations.enums[{i}].allowed",
                    "message": f"Allowed list contains empty/non-string values for column '{er.column}'",
                }
            )

def _find_dupes(items: list[str]) -> set[str]:
    seen: set[str] = set()
    dupes: set[str] = set()
    for x in items:
        if x in seen:
            dupes.add(x)
        seen.add(x)
    return dupes

def _validate_actions_against_columns(
    plan: CleaningPlan,
    df_columns: list[str],
    errors: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
) -> list[str]:
    # simulate current columns as actions apply
    cols = set(df_columns)

    for idx, action in enumerate(plan.actions):
        kind = action.action

        if kind == "rename_columns":
            mapping = action.mapping
            if not isinstance(mapping, dict) or len(mapping) == 0:
                warnings.append({"path": f"actions[{idx}].mapping", "message": "rename_columns mapping is empty"})
                continue

            # duplicate new names can cause collisions
            new_names = [v for v in mapping.values() if isinstance(v, str) and v.strip() != ""]
            dup_new = _find_dupes(new_names)
            if dup_new:
                warnings.append(
                    {
                        "path": f"actions[{idx}].mapping",
                        "message": f"Rename mapping has duplicate targets: {sorted(dup_new)}",
                    }
                )

            for old, new in mapping.items():
                if not isinstance(new, str) or new.strip() == "":
                    errors.append(
                        {
                            "path": f"actions[{idx}].mapping",
                            "message": f"Rename target for '{old}' must be a non-empty string",
                        }
                    )
                    continue

                if old not in cols:
                    warnings.append(
                        {
                            "path": f"actions[{idx}].mapping",
                            "message": f"Rename source column '{old}' not found in current columns",
                        }
                    )
                    continue

                cols.remove(old)
                cols.add(new)

        elif kind == "drop_columns":
            if not action.columns:
                warnings.append({"path": f"actions[{idx}].columns", "message": "drop_columns has empty columns list"})
            for c in action.columns:
                if c not in cols:
                    warnings.append(
                        {
                            "path": f"actions[{idx}].columns",
                            "message": f"drop_columns references missing column '{c}'",
                        }
                    )
                else:
                    cols.remove(c)

        elif kind == "trim_whitespace":
            # columns=None means apply to all
            if action.columns is not None:
                if len(action.columns) == 0:
                    warnings.append(
                        {"path": f"actions[{idx}].columns", "message": "trim_whitespace has empty columns list"}
                    )
                for c in action.columns:
                    if c not in cols:
                        warnings.append(
                            {
                                "path": f"actions[{idx}].columns",
                                "message": f"trim_whitespace references missing column '{c}'",
                            }
                        )

        elif kind == "standardize_nulls":
            # no column references here
            pass

        elif kind == "parse_numeric":
            if not action.columns:
                warnings.append({"path": f"actions[{idx}].columns", "message": "parse_numeric has empty columns list"})
            for c in action.columns:
                if c not in cols:
                    warnings.append(
                        {"path": f"actions[{idx}].columns", "message": f"parse_numeric references missing column '{c}'"}
                    )

        elif kind == "parse_dates":
            if not action.columns:
                warnings.append({"path": f"actions[{idx}].columns", "message": "parse_dates has empty columns list"})
            for c in action.columns:
                if c not in cols:
                    warnings.append(
                        {"path": f"actions[{idx}].columns", "message": f"parse_dates references missing column '{c}'"}
                    )

        elif kind == "deduplicate_rows":
            if action.subset is not None:
                if len(action.subset) == 0:
                    warnings.append(
                        {"path": f"actions[{idx}].subset", "message": "deduplicate_rows has empty subset list"}
                    )
                for c in action.subset:
                    if c not in cols:
                        warnings.append(
                            {
                                "path": f"actions[{idx}].subset",
                                "message": f"deduplicate_rows subset references missing column '{c}'",
                            }
                        )
        else:
            errors.append({"path": f"actions[{idx}].action", "message": f"unknown action '{kind}'"})

    # validations vs final columns (warnings, not errors)
    for rr in plan.validations.required:
        if rr.column not in cols:
            warnings.append(
                {
                    "path": "validations.required",
                    "message": f"required column '{rr.column}' not present after actions (may fail at runtime validation)",
                }
            )

    for rr in plan.validations.ranges:
        if rr.column not in cols:
            warnings.append(
                {
                    "path": "validations.ranges",
                    "message": f"range rule column '{rr.column}' not present after actions (will be skipped/fail at runtime)",
                }
            )

    for er in plan.validations.enums:
        if er.column not in cols:
            warnings.append(
                {
                    "path": "validations.enums",
                    "message": f"Enum rule column '{er.column}' not present after actions (will be skipped/fail at runtime)",
                }
            )

    return sorted(cols)