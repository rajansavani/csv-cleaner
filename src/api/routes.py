from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from src.pipeline.artifacts import write_cleaned_csv, write_report_json, read_report_json, cleaned_csv_path
from src.pipeline.profile import profile_dataframe, read_uploaded_csv
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
