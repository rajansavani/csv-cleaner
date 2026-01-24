from __future__ import annotations

from typing import Any

from src.llm.schemas import CleaningPlan

SYSTEM_PROMPT = """\
You are a careful data cleaning planner.

Your job:
- read a dataset profile (columns, missingness, and example rows)
- propose a cleaning plan that is SAFE and EXECUTABLE
- output ONLY valid JSON that matches the provided schema

Rules:
- do not invent columns that do not exist
- keep the plan minimal: prefer a few high-impact actions over many tiny ones
- if something is ambiguous, choose a conservative action (or skip it)
- avoid irreversible steps unless strongly justified (dropping columns is allowed only for obviously empty/junk columns)
- actions should be ordered in the way they should run (rename/trim/null handling first, then parsing, then dedupe)
- validations should be reasonable and not overly strict
"""

USER_PROMPT_TEMPLATE = """\
Here is the dataset profile for a CSV file.

FILENAME:
{filename}

COLUMNS:
{columns}

MISSINGNESS (per column):
{missingness}

PREVIEW ROWS (first {n_preview} rows):
{preview_rows}

Task:
Create a cleaning plan in JSON that matches this schema exactly:

{schema_json}

Return ONLY the JSON object. No markdown. No commentary.
"""

def build_planner_prompt(profile: dict[str, Any]) -> dict[str, str]:
    """
    Turns a /profile response into (system_prompt, user_prompt) strings.
    Keep formatting readable but compact.
    """
    filename = profile.get("filename", "uploaded.csv")
    columns = profile.get("columns", [])
    missingness = profile.get("missing_by_column", {})
    preview_rows = profile.get("preview_rows", [])
    n_preview = len(preview_rows)

    user_prompt = USER_PROMPT_TEMPLATE.format(
        filename=filename,
        columns=columns,
        missingness=missingness,
        n_preview=n_preview,
        preview_rows=preview_rows,
        schema_json=CleaningPlan.json_schema(),
    )

    return {
        "system": SYSTEM_PROMPT,
        "user": user_prompt,
    }