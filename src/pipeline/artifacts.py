from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

def ensure_output_dirs(base_dir: str = "outputs") -> dict[str, Path]:
    """
    Create runtime output directories if they don't exist.
    """
    base = Path(base_dir)
    cleaned_dir = base / "cleaned"
    reports_dir = base / "reports"
    plans_dir = base / "plans"

    cleaned_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    plans_dir.mkdir(parents=True, exist_ok=True)

    return {
        "base": base,
        "cleaned": cleaned_dir,
        "reports": reports_dir,
        "plans": plans_dir,
    }

def write_cleaned_csv(df: pd.DataFrame, job_id: str, *, base_dir: str = "outputs") -> Path:
    paths = ensure_output_dirs(base_dir)
    out_path = paths["cleaned"] / f"{job_id}.csv"
    df.to_csv(out_path, index=False)
    return out_path

def write_report_json(report: dict[str, Any], job_id: str, *, base_dir: str = "outputs") -> Path:
    paths = ensure_output_dirs(base_dir)
    out_path = paths["reports"] / f"{job_id}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return out_path

def write_plan_json(plan: dict[str, Any], job_id: str, *, base_dir: str = "outputs") -> Path:
    paths = ensure_output_dirs(base_dir)
    out_path = paths["plans"] / f"{job_id}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)
    return out_path

def read_report_json(job_id: str, *, base_dir: str = "outputs") -> dict[str, Any]:
    paths = ensure_output_dirs(base_dir)
    report_path = paths["reports"] / f"{job_id}.json"
    if not report_path.exists():
        raise FileNotFoundError(f"Report not found for job_id={job_id}")
    with report_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def cleaned_csv_path(job_id: str, *, base_dir: str = "outputs") -> Path:
    paths = ensure_output_dirs(base_dir)
    p = paths["cleaned"] / f"{job_id}.csv"
    if not p.exists():
        raise FileNotFoundError(f"Cleaned CSV not found for job_id={job_id}")
    return p