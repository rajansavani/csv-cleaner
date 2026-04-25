from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.llm.client import LLMResponse
from src.llm.reflector import ReflectionError, reflect_on_cleaning
from src.llm.schemas import (
    CleaningPlan,
    FinalReflection,
    FlagUnrecoverable,
    MarkClean,
    MaxIterationsExceeded,
    ProposeRevision,
    ReflectionDecision,
    ReflectionFailed,
)
from src.pipeline.executor import ExecutionError, execute_plan
from src.pipeline.planner import generate_cleaning_plan
from src.pipeline.profile import profile_dataframe
from src.pipeline.validate import (
    PlanValidationError,
    PlanValidationResult,
    ensure_valid_plan,
)

# current gpt-4o-mini pricing (USD per 1M tokens) as of April 2026
PRICE_PER_1M_INPUT_TOKENS = 0.15
PRICE_PER_1M_OUTPUT_TOKENS = 0.60

DEFAULT_MAX_ITERATIONS = 3


@dataclass
class IterationRecord:
    pass_: int
    triggering_reflection: ReflectionDecision | None  # none for the first pass
    plan: CleaningPlan
    execution_report: dict[str, Any]


@dataclass
class LoopMetrics:
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    wall_clock_seconds: float


@dataclass
class LoopResult:
    final_df: pd.DataFrame
    iterations: list[IterationRecord]
    final_reflection: FinalReflection
    metrics: LoopMetrics
    before_profile: dict[str, Any]
    initial_plan_validation: PlanValidationResult


# get token usage and timing metrics for the entire loop run, accumulating across all LLM calls
@dataclass
class _MetricsAccumulator:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    _start: float = field(default_factory=time.perf_counter)

    def add(self, response: LLMResponse) -> None:
        self.prompt_tokens += response.prompt_tokens
        self.completion_tokens += response.completion_tokens

    def finalize(self) -> LoopMetrics:
        total = self.prompt_tokens + self.completion_tokens
        cost = (
            (self.prompt_tokens / 1_000_000) * PRICE_PER_1M_INPUT_TOKENS
            + (self.completion_tokens / 1_000_000) * PRICE_PER_1M_OUTPUT_TOKENS
        )
        return LoopMetrics(
            total_prompt_tokens=self.prompt_tokens,
            total_completion_tokens=self.completion_tokens,
            total_tokens=total,
            estimated_cost_usd=round(cost, 6),
            wall_clock_seconds=round(time.perf_counter() - self._start, 4),
        )


# helper to attach plan validation warnings to the execution report, so they get surfaced in the API response
def _attach_plan_validation_warnings(
    exec_report: dict[str, Any], plan_validation: PlanValidationResult
) -> None:
    if plan_validation.warnings:
        exec_report.setdefault("warnings", [])
        exec_report["warnings"].extend(
            [{"type": "plan_validation", **w} for w in plan_validation.warnings]
        )


def run_clean_loop(
    df: pd.DataFrame,
    *,
    filename: str | None = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> LoopResult:
    """
    Plan -> execute -> reflect is the agent loop pattern implemented here. 
    The loop runs until either the reflector returns MarkClean or FlagUnrecoverable,
    or we hit the max_iterations cap.

    Pass 1 is planner-driven and any failure (PlanError / PlanValidationError /
    ExecutionError) propagates to the caller as a hard error. Passes 2..N are
    reflection-driven and fail-graceful: any failure stops the loop and is
    captured in LoopResult.final_reflection as a ReflectionFailed sentinel.

    The reflection that triggered a pass goes on that pass's
    IterationRecord.triggering_reflection. The reflection (or sentinel) that
    ended the loop goes on LoopResult.final_reflection. They are never the
    same object.
    """
    metrics = _MetricsAccumulator()

    before_profile = profile_dataframe(df, filename=filename)

    # pass 1: planner-driven, fail-hard on any error
    plan, planner_response = generate_cleaning_plan(before_profile)
    metrics.add(planner_response)

    initial_plan_validation = ensure_valid_plan(plan, df_columns=list(df.columns))

    cleaned_df, exec_report = execute_plan(df, plan)
    _attach_plan_validation_warnings(exec_report, initial_plan_validation)

    iterations: list[IterationRecord] = [
        IterationRecord(
            pass_=1,
            triggering_reflection=None,
            plan=plan,
            execution_report=exec_report,
        )
    ]

    last_plan: CleaningPlan = plan
    last_exec_report: dict[str, Any] = exec_report
    final_reflection: FinalReflection | None = None

    # passes 2..max_iterations: reflection-driven, fail-graceful
    for pass_num in range(2, max_iterations + 1):
        cleaned_profile = profile_dataframe(cleaned_df, filename=filename)

        try:
            decision, reflector_response = reflect_on_cleaning(
                cleaned_profile=cleaned_profile,
                last_plan=last_plan,
                last_execution_report=last_exec_report,
            )
            metrics.add(reflector_response)
        except ReflectionError as e:
            final_reflection = ReflectionFailed(stage=e.stage, error=str(e))
            break

        if isinstance(decision, MarkClean):
            final_reflection = decision
            break
        if isinstance(decision, FlagUnrecoverable):
            final_reflection = decision
            break
        assert isinstance(decision, ProposeRevision)

        try:
            revised_validation = ensure_valid_plan(
                decision.revised_plan, df_columns=list(cleaned_df.columns)
            )
        except PlanValidationError as e:
            final_reflection = ReflectionFailed(
                stage="plan_validation",
                error=f"revised plan failed semantic validation: errors={e.errors}",
            )
            break

        try:
            cleaned_df, exec_report = execute_plan(cleaned_df, decision.revised_plan)
        except ExecutionError as e:
            final_reflection = ReflectionFailed(stage="execution", error=str(e))
            break

        _attach_plan_validation_warnings(exec_report, revised_validation)

        iterations.append(
            IterationRecord(
                pass_=pass_num,
                triggering_reflection=decision,
                plan=decision.revised_plan,
                execution_report=exec_report,
            )
        )
        last_plan = decision.revised_plan
        last_exec_report = exec_report

    # loop termination: if we exited the loop without a reflection verdict, it means we hit the max_iterations cap
    if final_reflection is None:
        # the decision that triggered the final pass is the last verdict we have
        # if max_iterations == 1, no reflection ever ran and last_verdict is None
        last_verdict: ReflectionDecision | None = (
            iterations[-1].triggering_reflection if len(iterations) > 1 else None
        )
        final_reflection = MaxIterationsExceeded(
            iteration_cap=max_iterations,
            last_verdict=last_verdict,
        )

    return LoopResult(
        final_df=cleaned_df,
        iterations=iterations,
        final_reflection=final_reflection,
        metrics=metrics.finalize(),
        before_profile=before_profile,
        initial_plan_validation=initial_plan_validation,
    )
