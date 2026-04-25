from __future__ import annotations

from typing import Any, Literal

from pydantic import ValidationError

from src.llm.client import LLMError, LLMJSONParseError, LLMResponse, OpenAIClient
from src.llm.reflection_prompts import build_reflector_prompt
from src.llm.schemas import CleaningPlan, ReflectionDecision, ReflectionResponse

ReflectionFailureStage = Literal["llm_call", "json_parse", "schema_validation"]


class ReflectionError(RuntimeError):
    """
    Raised when the reflector cannot produce a valid ReflectionDecision.

    Carries the stage information so the orchestrator can construct a ReflectionFailed
    sentinel with the matching stage value.
    """

    def __init__(self, message: str, *, stage: ReflectionFailureStage) -> None:
        super().__init__(message)
        self.stage: ReflectionFailureStage = stage


def reflect_on_cleaning(
    *,
    cleaned_profile: dict[str, Any],
    last_plan: CleaningPlan,
    last_execution_report: dict[str, Any],
    model: str = "gpt-4o-mini",
) -> tuple[ReflectionDecision, LLMResponse]:
    """
    Ask the LLM to inspect a freshly-cleaned dataset and return a decision.

    Returns (decision, llm_response). The LLMResponse is returned so the orchestrator can track token usage.

    Raises ReflectionError on llm_call / json_parse / schema_validation failure.
    Plan-validation and execution failures for ProposeRevision.revised_plan are the orchestrator's responsibility.
    """
    prompts = build_reflector_prompt(
        cleaned_profile=cleaned_profile,
        last_plan=last_plan,
        last_execution_report=last_execution_report,
    )

    client = OpenAIClient(model=model)

    try:
        response = client.generate_json(system=prompts["system"], user=prompts["user"])
    except LLMJSONParseError as e:
        raise ReflectionError(str(e), stage="json_parse")
    except LLMError as e:
        raise ReflectionError(str(e), stage="llm_call")

    try:
        parsed = ReflectionResponse.model_validate(response.data)
    except ValidationError as e:
        raise ReflectionError(
            f"Reflector returned JSON that does not match ReflectionResponse: {e}",
            stage="schema_validation",
        )

    return parsed.result, response
