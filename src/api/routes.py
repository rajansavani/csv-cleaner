from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from src.pipeline.artifacts import write_cleaned_csv, write_report_json, read_report_json, cleaned_csv_path, write_plan_json
from src.pipeline.profile import profile_dataframe, read_uploaded_csv
from src.pipeline.planner import PlanError, generate_cleaning_plan
from src.pipeline.executor import ExecutionError, execute_plan
from src.pipeline.validate import PlanValidationError, ensure_valid_plan
from src.tools.transforms import basic_clean


router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/profile")
def profile_csv(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="missing filename")

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="only .csv files are supported")

    df = read_uploaded_csv(file)

    if df.shape[1] == 0:
        raise HTTPException(status_code=400, detail="no columns found in csv")

    return profile_dataframe(df, filename=file.filename)


def _new_job_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    rand = uuid4().hex[:8]
    return f"{stamp}_{rand}"


@router.post("/clean/basic")
def clean_basic(file: UploadFile = File(...)) -> dict[str, Any]:
    """
    Perform basic, deterministic cleaning on the uploaded CSV.
    Does not call any LLMs.
    """
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

    # build a report artifact that’s useful for later debugging
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

@router.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    try:
        return read_report_json(job_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="job_id not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to read report: {e}")


@router.get("/jobs/{job_id}/cleaned.csv")
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

@router.post("/plan")
def plan_cleaning(file: UploadFile = File(...)) -> dict[str, Any]:
    """
    Generate an LLM cleaning plan (JSON) from an uploaded CSV.
    This does not modify the data yet, just generates a plan.
    """
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
        plan = generate_cleaning_plan(profile)
        plan_validation = ensure_valid_plan(plan, df_columns=list(df.columns))
    except PlanError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except PlanValidationError as e:
        # semantic plan issues -> 422
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

@router.post("/clean/llm")
def clean_llm(file: UploadFile = File(...)) -> dict[str, Any]:
    """
    Agentic cleaning:
    - Profile csv
    - Use OpenAI to generate a structured cleaning plan
    - Execute the plan deterministically
    - Save artifacts (cleaned csv, report json, plan json)
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="missing filename")

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="only .csv files are supported")

    df = read_uploaded_csv(file)

    if df.shape[1] == 0:
        raise HTTPException(status_code=400, detail="no columns found in csv")

    job_id = _new_job_id()

    # profile first (used for both plan + report)
    before = profile_dataframe(df, filename=file.filename)

    # plan (llm)
    try:
        plan = generate_cleaning_plan(before)
        plan_validation = ensure_valid_plan(plan, df_columns=list(df.columns))
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

    # execute
    try:
        cleaned_df, exec_report = execute_plan(df, plan)
    except ExecutionError as e:
        raise HTTPException(status_code=500, detail=f"execution failed: {e}")

    # attach plan validation warnings to the execution report (if any)
    if plan_validation.warnings:
        exec_report.setdefault("warnings", [])
        exec_report["warnings"].extend(
            [{"type": "plan_validation", **w} for w in plan_validation.warnings]
        )

    after = profile_dataframe(cleaned_df, filename=file.filename)

    plan_dict = plan.model_dump()

    report = {
        "job_id": job_id,
        "filename": file.filename,
        "cleaning_mode": "llm",
        "plan": plan_dict,
        "execution_report": exec_report,
        "before": before,
        "after": after,
    }

    # save artifacts
    cleaned_path = write_cleaned_csv(cleaned_df, job_id)
    report_path = write_report_json(report, job_id)
    plan_path = write_plan_json(plan_dict, job_id)

    return {
        "job_id": job_id,
        "filename": file.filename,
        "cleaning_mode": "llm",
        "plan": plan_dict,
        "plan_validation": {
            "ok": True,
            "warnings": plan_validation.warnings,
            "final_columns": plan_validation.final_columns,
        },
        "execution_report": exec_report,
        "artifacts": {
            "cleaned_csv": str(cleaned_path),
            "report_json": str(report_path),
            "plan_json": str(plan_path),
        },
        "before": before,
        "after": after,
        "cleaned_preview_rows": cleaned_df.head(10).fillna("").to_dict(orient="records"),
    }