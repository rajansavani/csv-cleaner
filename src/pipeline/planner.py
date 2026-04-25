from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from src.llm.client import LLMError, LLMResponse, OpenAIClient
from src.llm.prompts import build_planner_prompt
from src.llm.schemas import CleaningPlan


class PlanError(RuntimeError):
    pass


def generate_cleaning_plan(
    profile: dict[str, Any],
    *,
    model: str = "gpt-4o-mini",
) -> tuple[CleaningPlan, LLMResponse]:
    """
    Given a dataset profile (from profile_dataframe), ask the LLM for a plan and validate
    it against our CleaningPlan schema.

    Returns (plan, llm_response). The LLMResponse is returned so callers (the orchestrator)
    can accumulate token usage. Callers that don't care can ignore it.
    """
    prompts = build_planner_prompt(profile)
    client = OpenAIClient(model=model)

    try:
        response = client.generate_json(system=prompts["system"], user=prompts["user"])
    except LLMError as e:
        raise PlanError(str(e))

    try:
        plan = CleaningPlan.model_validate(response.data)
    except ValidationError as e:
        raise PlanError(f"LLM returned invalid plan schema: {e}")

    return plan, response
