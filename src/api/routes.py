from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from dataclasses import asdict

from src.pipeline.artifacts import (
    cleaned_csv_path,
    read_report_json,
    write_cleaned_csv,
    write_plan_json,
    write_report_json,
)
from src.pipeline.executor import ExecutionError
from src.pipeline.loop import IterationRecord, run_clean_loop
from src.pipeline.planner import PlanError, generate_cleaning_plan
from src.pipeline.profile import profile_dataframe, read_uploaded_csv
from src.pipeline.validate import PlanValidationError, ensure_valid_plan
from src.tools.transforms import basic_clean


router = APIRouter()

TAG_SYSTEM = "system"
TAG_PROFILE = "profile"
TAG_PLANNING = "planning"
TAG_CLEANING = "cleaning"
TAG_JOBS = "jobs"


def _new_job_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    rand = uuid4().hex[:8]
    return f"{stamp}_{rand}"


@router.get(
    "/health",
    tags=[TAG_SYSTEM],
    summary="Health check",
    description="Simple liveness probe for the API.",
)
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post(
    "/profile",
    tags=[TAG_PROFILE],
    summary="Profile a CSV",
    description=(
        "Uploads a CSV and returns a lightweight dataset profile:\n"
        "- shape (rows/cols)\n"
        "- column list\n"
        "- missingness by column\n"
        "- duplicate row count\n"
        "- preview rows\n\n"
        "This endpoint does not run any LLM calls."
    ),
)
def profile_csv(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="missing filename")

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="only .csv files are supported")

    df = read_uploaded_csv(file)

    if df.shape[1] == 0:
        raise HTTPException(status_code=400, detail="no columns found in csv")

    return profile_dataframe(df, filename=file.filename)


@router.post(
    "/clean/basic",
    tags=[TAG_CLEANING],
    summary="Clean a CSV with deterministic rules",
    description=(
        "Runs a deterministic cleaning pass (no LLM):\n"
        "- standardize null-like tokens\n"
        "- trim whitespace\n"
        "- basic type normalization where safe\n"
        "- drop fully-empty columns (if implemented in your basic_clean)\n\n"
        "Saves artifacts:\n"
        "- cleaned csv\n"
        "- report json\n"
    ),
)
def clean_basic(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="missing filename")

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="only .csv files are supported")

    df = read_uploaded_csv(file)

    if df.shape[1] == 0:
        raise HTTPException(status_code=400, detail="no columns found in csv")

    before = profile_dataframe(df, filename=file.filename)

    cleaned_df, clean_stats = basic_clean(df)

    after = profile_dataframe(cleaned_df, filename=file.filename)

    job_id = _new_job_id()

    report = {
        "job_id": job_id,
        "filename": file.filename,
        "cleaning_mode": "basic",
        "clean_stats": clean_stats,
        "before": before,
        "after": after,
    }

    cleaned_path = write_cleaned_csv(cleaned_df, job_id)
    report_path = write_report_json(report, job_id)

    return {
        "job_id": job_id,
        "filename": file.filename,
        "cleaning_mode": "basic",
        "clean_stats": clean_stats,
        "artifacts": {
            "cleaned_csv": str(cleaned_path),
            "report_json": str(report_path),
        },
        "before": before,
        "after": after,
        "cleaned_preview_rows": cleaned_df.head(10).fillna("").to_dict(orient="records"),
    }


@router.post(
    "/plan",
    tags=[TAG_PLANNING],
    summary="Generate an LLM cleaning plan",
    description=(
        "Uploads a CSV, profiles it, then asks the LLM for a structured cleaning plan.\n\n"
        "Notes:\n"
        "- this endpoint does not modify the data\n"
        "- the returned plan is validated semantically (plan-level checks)\n"
        "- the plan is saved to disk as an artifact\n"
    ),
)
def plan_cleaning(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="missing filename")

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="only .csv files are supported")

    df = read_uploaded_csv(file)

    if df.shape[1] == 0:
        raise HTTPException(status_code=400, detail="no columns found in csv")

    profile = profile_dataframe(df, filename=file.filename)

    job_id = _new_job_id()

    try:
        plan, _ = generate_cleaning_plan(profile)
        plan_validation = ensure_valid_plan(plan, df_columns=list(df.columns))
    except PlanError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except PlanValidationError as e:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "plan validation failed",
                "errors": e.errors,
                "warnings": e.warnings,
            },
        )

    plan_dict = plan.model_dump()
    plan_path = write_plan_json(plan_dict, job_id)

    return {
        "job_id": job_id,
        "filename": file.filename,
        "model": "gpt-4o-mini",
        "plan": plan_dict,
        "plan_validation": {
            "ok": True,
            "warnings": plan_validation.warnings,
            "final_columns": plan_validation.final_columns,
        },
        "artifacts": {
            "plan_json": str(plan_path),
        },
    }


@router.post(
    "/clean/llm",
    tags=[TAG_CLEANING],
    summary="Clean a CSV using an LLM-generated plan",
    description=(
        "Agentic cleaning pipeline:\n"
        "1) profile the dataset\n"
        "2) generate a structured plan with the LLM\n"
        "3) validate plan semantics\n"
        "4) execute deterministically with pandas transforms\n"
        "5) validate constraints and generate a report\n"
        "6) save artifacts (cleaned csv, report json, plan json)\n"
    ),
)
def clean_llm(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="missing filename")

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="only .csv files are supported")

    df = read_uploaded_csv(file)

    if df.shape[1] == 0:
        raise HTTPException(status_code=400, detail="no columns found in csv")

    job_id = _new_job_id()

    try:
        result = run_clean_loop(df, filename=file.filename)
    except PlanError as e:
        raise HTTPException(status_code=500, detail=f"planning failed: {e}")
    except PlanValidationError as e:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "plan validation failed",
                "errors": e.errors,
                "warnings": e.warnings,
            },
        )
    except ExecutionError as e:
        raise HTTPException(status_code=500, detail=f"execution failed: {e}")

    initial_plan = result.iterations[0].plan
    initial_plan_dict = initial_plan.model_dump()
    final_exec_report = result.iterations[-1].execution_report

    after = profile_dataframe(result.final_df, filename=file.filename)

    iterations_serialized = [_serialize_iteration(it) for it in result.iterations]
    final_reflection_dict = result.final_reflection.model_dump()
    metrics_dict = asdict(result.metrics)

    report = {
        "job_id": job_id,
        "filename": file.filename,
        "cleaning_mode": "llm",
        "plan": initial_plan_dict,
        "execution_report": final_exec_report,
        "iterations": iterations_serialized,
        "total_iterations": len(result.iterations),
        "final_reflection": final_reflection_dict,
        "metrics": metrics_dict,
        "before": result.before_profile,
        "after": after,
    }

    cleaned_path = write_cleaned_csv(result.final_df, job_id)
    report_path = write_report_json(report, job_id)
    plan_path = write_plan_json(initial_plan_dict, job_id)

    return {
        "job_id": job_id,
        "filename": file.filename,
        "cleaning_mode": "llm",
        "plan": initial_plan_dict,
        "plan_validation": {
            "ok": True,
            "warnings": result.initial_plan_validation.warnings,
            "final_columns": result.initial_plan_validation.final_columns,
        },
        "execution_report": final_exec_report,
        "iterations": iterations_serialized,
        "total_iterations": len(result.iterations),
        "final_reflection": final_reflection_dict,
        "metrics": metrics_dict,
        "artifacts": {
            "cleaned_csv": str(cleaned_path),
            "report_json": str(report_path),
            "plan_json": str(plan_path),
        },
        "before": result.before_profile,
        "after": after,
        "cleaned_preview_rows": result.final_df.head(10).fillna("").to_dict(orient="records"),
    }


def _serialize_iteration(it: IterationRecord) -> dict[str, Any]:
    return {
        "pass": it.pass_,
        "triggering_reflection": (
            it.triggering_reflection.model_dump() if it.triggering_reflection is not None else None
        ),
        "plan": it.plan.model_dump(),
        "execution_report": it.execution_report,
    }


@router.get(
    "/jobs/{job_id}",
    tags=[TAG_JOBS],
    summary="Fetch a job report",
    description="Returns the saved report JSON for a previous run (basic or llm).",
)
def get_job(job_id: str) -> dict[str, Any]:
    try:
        return read_report_json(job_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="job_id not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to read report: {e}")


@router.get(
    "/jobs/{job_id}/cleaned.csv",
    tags=[TAG_JOBS],
    summary="Download the cleaned CSV",
    description="Downloads the cleaned CSV artifact for a previous run.",
)
def download_cleaned_csv(job_id: str) -> FileResponse:
    try:
        path = cleaned_csv_path(job_id)
        return FileResponse(
            path=str(path),
            media_type="text/csv",
            filename=f"{job_id}.csv",
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="job_id not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to read csv: {e}")
