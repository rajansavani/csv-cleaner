from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from src.llm.client import LLMError, OpenAIClient
from src.llm.prompts import build_planner_prompt
from src.llm.schemas import CleaningPlan


class PlanError(RuntimeError):
    pass


def generate_cleaning_plan(profile: dict[str, Any], *, model: str = "gpt-4o-mini") -> CleaningPlan:
    """
    Given a dataset profile (from profile_dataframe), ask the LLM for a plan and validate it against our CleaningPlan schema.
    """
    prompts = build_planner_prompt(profile)
    client = OpenAIClient(model=model)

    try:
        raw = client.generate_json(system=prompts["system"], user=prompts["user"])
    except LLMError as e:
        raise PlanError(str(e))

    try:
        return CleaningPlan.model_validate(raw)
    except ValidationError as e:
        raise PlanError(f"LLM returned invalid plan schema: {e}")
