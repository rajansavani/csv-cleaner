from __future__ import annotations

from src.llm.prompts import build_planner_prompt
from src.llm.schemas import CleaningPlan


def test_build_planner_prompt_contains_schema() -> None:
    fake_profile = {
        "filename": "test.csv",
        "columns": ["A", "B"],
        "missing_by_column": {"A": {"missing_count": 0, "missing_pct": 0.0}},
        "preview_rows": [{"A": "1", "B": "x"}],
    }

    prompts = build_planner_prompt(fake_profile)
    assert "system" in prompts
    assert "user" in prompts
    assert "Create a cleaning plan" in prompts["user"]
    # schema should be embedded in the user prompt
    assert "CleaningPlan" in prompts["user"] or "model_json_schema" in prompts["user"] or "properties" in prompts["user"]


def test_cleaning_plan_parses_minimal_plan() -> None:
    plan_dict = {
        "version": "1",
        "summary": "basic cleanup",
        "actions": [{"action": "trim_whitespace", "columns": None}],
        "validations": {"required": [], "ranges": [], "enums": []},
    }

    plan = CleaningPlan.model_validate(plan_dict)
    assert plan.version == "1"
    assert len(plan.actions) == 1
