from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile

from src.pipeline.profile import profile_dataframe, read_uploaded_csv
from src.tools.transforms import basic_clean


router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/profile")
def profile_csv(file: UploadFile = File(...)) -> dict[str, Any]:
    """
    Profile an uploaded csv file (quick, lightweight).
    Returns:
      - Shape, columns
      - Missingness by column
      - Duplicate row count
      - Preview rows
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="missing filename")

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="only .csv files are supported")

    df = read_uploaded_csv(file)

    if df.shape[1] == 0:
        raise HTTPException(status_code=400, detail="no columns found in csv")

    return profile_dataframe(df, filename=file.filename)


@router.post("/clean/basic")
def clean_basic(file: UploadFile = File(...)) -> dict[str, Any]:
    """
    Deterministic cleaning pass (no OpenAI calls).
    Returns before/after profiles + cleaning stats + cleaned preview.
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

    # warnings for debugging
    warnings: list[str] = []
    sem = clean_stats.get("semantic_cleaning", {}).get("columns_cleaned", {})
    for col, info in sem.items():
        changed = int(info.get("changed_cells", 0))
        n_rows = int(after["shape"]["rows"])
        if n_rows > 0 and changed / n_rows > 0.95:
            warnings.append(f"column '{col}' changed on most rows ({changed}/{n_rows})")

    cleaned_preview = cleaned_df.head(10).fillna("").to_dict(orient="records")

    return {
        "filename": file.filename,
        "cleaning_mode": "basic",
        "clean_stats": clean_stats,
        "warnings": warnings,
        "before": before,
        "after": after,
        "cleaned_preview_rows": cleaned_preview,
    }