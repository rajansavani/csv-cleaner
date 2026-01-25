import pytest

from src.llm.schemas import CleaningPlan
from src.pipeline.validate import PlanValidationError, ensure_valid_plan, validate_plan


def test_validate_plan_ok() -> None:
    raw = {
        "version": "1",
        "summary": "basic plan",
        "actions": [
            {"action": "rename_columns", "mapping": {"A": "B"}},
            {"action": "parse_numeric", "columns": ["B"], "numeric_type": "float"},
        ],
        "validations": {
            "required": [{"column": "B"}],
            "ranges": [{"column": "B", "min": 0, "max": None}],
            "enums": [],
        },
    }

    plan = CleaningPlan.model_validate(raw)
    result = ensure_valid_plan(plan, df_columns=["A"])

    assert result.ok is True
    assert result.errors == []
    # plan references B after rename, so it should be present
    assert result.final_columns is not None
    assert "B" in result.final_columns


def test_validate_plan_errors_on_bad_range() -> None:
    raw = {
        "version": "1",
        "summary": "bad range",
        "actions": [],
        "validations": {
            "required": [],
            "ranges": [{"column": "Score", "min": 10, "max": 0}],
            "enums": [],
        },
    }

    plan = CleaningPlan.model_validate(raw)

    with pytest.raises(PlanValidationError) as e:
        ensure_valid_plan(plan)

    # make sure we actually report a range error
    assert any("min" in err["message"].lower() for err in e.value.errors)


def test_validate_plan_errors_on_empty_enum_allowed() -> None:
    raw = {
        "version": "1",
        "summary": "bad enum",
        "actions": [],
        "validations": {
            "required": [],
            "ranges": [],
            "enums": [{"column": "Content Rating", "allowed": []}],
        },
    }

    plan = CleaningPlan.model_validate(raw)

    with pytest.raises(PlanValidationError) as e:
        ensure_valid_plan(plan)

    assert any("allowed list cannot be empty" in err["message"].lower() for err in e.value.errors)


def test_validate_plan_warns_on_missing_columns() -> None:
    raw = {
        "version": "1",
        "summary": "missing column warning",
        "actions": [
            {"action": "parse_numeric", "columns": ["MissingCol"], "numeric_type": "float"},
        ],
        "validations": {
            "required": [{"column": "MissingCol"}],
            "ranges": [],
            "enums": [],
        },
    }

    plan = CleaningPlan.model_validate(raw)
    result = validate_plan(plan, df_columns=["A"])

    assert result.ok is True  # warnings only
    assert result.errors == []
    assert len(result.warnings) > 0
    assert any("MissingCol" in w["message"] for w in result.warnings)
