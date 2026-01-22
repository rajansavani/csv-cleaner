from __future__ import annotations

from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile

from src.pipeline.profile import profile_dataframe, read_uploaded_csv


app = FastAPI(
    title="CSVCleaner",
    version="0.1.0",
    description="Upload a CSV, get a quick profile, then clean + validate it using LLMs",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/profile")
def profile_csv(file: UploadFile = File(...)) -> dict[str, Any]:
    """
    Upload a csv and return a basic profile:
    - Columns, missingness, duplicate count
    - Small preview for debugging
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="missing filename")

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="only .csv files are supported")

    df = read_uploaded_csv(file)

    if df.shape[1] == 0:
        raise HTTPException(status_code=400, detail="no columns found in csv")

    return profile_dataframe(df, filename=file.filename)