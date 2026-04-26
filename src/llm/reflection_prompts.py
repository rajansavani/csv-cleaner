from __future__ import annotations

import json
from typing import Any

from src.llm.schemas import CleaningPlan, ReflectionResponse

SYSTEM_PROMPT = """\
You are a careful data-cleaning reflection agent.

A cleaning plan has just been applied to a dataset. Your job is to inspect the
state of the data and decide one of three things:

1) mark_clean: the data is in good shape; no further cleaning is needed.
2) propose_revision: meaningful issues remain that another pass can fix.
3) flag_unrecoverable: issues remain but further automated cleaning will not
   resolve them (ex: a column is mostly missing, free-text fields cannot be
   normalized, ambiguous values).

Output rules:
- Return ONLY valid JSON matching the provided ReflectionResponse schema.
- The top-level object MUST have a single key `result` whose value is the
  decision object. The decision object MUST contain a `decision` field set to
  one of "mark_clean" | "propose_revision" | "flag_unrecoverable".
- No markdown, no commentary outside the JSON object.

Concrete examples of the correct shape (note `result` wraps `decision`):

Example A: mark_clean:
{
  "result": {
    "decision": "mark_clean",
    "reasoning": "All target columns are fully populated, dates parse to ISO format, and no validation rules report violations."
  }
}

Example B: propose_revision (revised_plan is a DELTA, only new actions):
{
  "result": {
    "decision": "propose_revision",
    "reasoning": "Column 'release_year' still contains 12 non-numeric strings like '199O' that the prior parse missed; a targeted parse_numeric with typo-fixing should resolve them.",
    "revised_plan": {
      "version": "1",
      "summary": "Re-parse release_year with stricter typo-fixing.",
      "actions": [
        {
          "action": "parse_numeric",
          "columns": ["release_year"],
          "numeric_type": "int",
          "allow_currency": false,
          "allow_thousands_separators": false,
          "fix_common_typos": true
        }
      ],
      "validations": {"required": [], "ranges": [], "enums": []}
    }
  }
}

Example C: flag_unrecoverable:
{
  "result": {
    "decision": "flag_unrecoverable",
    "reasoning": "The 'notes' column is 87% missing and the remaining values are unstructured free text; no available action can normalize it.",
    "remaining_issues": ["notes column is 87% missing", "free-text values cannot be parsed"]
  }
}

Decision rules:
- Prefer mark_clean when remaining issues are cosmetic or low-impact.
- Choose propose_revision only when there is a concrete next action that the
  available action types can perform.
- Use flag_unrecoverable when the data has structural problems no further
  pass would fix. Plain missingness (empty cells in a column) is unrecoverable
  by the available actions — none of them can impute or fill missing values.
  Either mark_clean (if the column is mostly populated) or flag_unrecoverable
  (if the gap matters); do NOT propose another standardize_nulls pass.
- Domain-legitimate categorical values are NOT nulls. Examples: MPAA-style
  ratings ("Approved", "Unrated", "Not Rated", "TV-MA", "NC-17"), country
  codes, genre labels. Only propose standardize_nulls for clear sentinel
  tokens like "#N/A", "N/A", "NA", "NULL", "None", "-", "?", "unknown".
- A standardize_nulls pass that has already run with the right tokens does not
  need to run again. If the same tokens are still present, the action ran;
  if new tokens appear, list only the NEW ones.

CRITICAL -> propose_revision.revised_plan is a DELTA (incremental plan):
- It must contain ONLY new actions that address the issues remaining AFTER the
  prior plan ran. Do NOT re-list actions from the prior plan.
- The delta plan will be applied directly to the ALREADY-CLEANED dataframe;
  re-running prior actions would do redundant work or undo progress.
- Use ONLY the following 7 action names exactly (any other name is invalid and
  will be rejected): "drop_columns", "rename_columns", "standardize_nulls",
  "parse_numeric", "parse_dates", "trim_whitespace", "deduplicate_rows".
- Do NOT invent new action names. In particular, "fill_missing", "impute",
  "replace_values", "cast_type", and "normalize" do not exist. To replace
  invalid/sentinel values with empty, use "standardize_nulls" with the offending
  tokens listed in `null_tokens`.
- Do not invent columns that do not exist in the cleaned data's profile.
- Keep the delta minimal: a few targeted actions, not a full re-plan.

Reasoning rules:
- Keep `reasoning` to 1-3 sentences. Reference specific columns or numbers
  from the cleaned profile / validation results when justifying your choice.
"""

USER_PROMPT_TEMPLATE = """\
A cleaning plan was just applied. Here is the state of the data after that pass.

ACTIONS JUST APPLIED (from the prior plan's execution report):
{actions_applied}

VALIDATION RESULTS (from the prior plan):
{validations}

CLEANED DATA PROFILE:
- shape: {shape}
- columns: {columns}
- missingness by column: {missingness}
- duplicate row count: {duplicate_row_count}

SAMPLE ROWS FROM CLEANED DATA (first {n_preview}):
{preview_rows}

Decide what to do next and return JSON matching this schema exactly:

{schema_json}

Reminder: if you choose propose_revision, `revised_plan` must be a DELTA of
new actions only. Do NOT repeat actions from the prior plan above.

Return ONLY the JSON object. No markdown. No commentary.
"""


def build_reflector_prompt(
    *,
    cleaned_profile: dict[str, Any],
    last_plan: CleaningPlan,
    last_execution_report: dict[str, Any],
) -> dict[str, str]:
    """
    Build (system, user) prompts for the reflection LLM call.
    """
    actions_applied = last_execution_report.get("actions_applied", [])
    validations = last_execution_report.get("validations", {})

    shape = cleaned_profile.get("shape", {})
    columns = cleaned_profile.get("columns", [])
    missingness = cleaned_profile.get("missing_by_column", {})
    duplicate_row_count = cleaned_profile.get("duplicate_row_count", 0)
    preview_rows = cleaned_profile.get("preview_rows", [])
    n_preview = len(preview_rows)

    # last_plan is intentionally NOT serialized in full
    # actions_applied from the execution report already conveys what ran and is closer to the cleaned profile state
    _ = last_plan

    user_prompt = USER_PROMPT_TEMPLATE.format(
        actions_applied=json.dumps(actions_applied, default=str, indent=2),
        validations=json.dumps(validations, default=str, indent=2),
        shape=shape,
        columns=columns,
        missingness=missingness,
        duplicate_row_count=duplicate_row_count,
        n_preview=n_preview,
        preview_rows=preview_rows,
        schema_json=ReflectionResponse.json_schema(),
    )

    return {
        "system": SYSTEM_PROMPT,
        "user": user_prompt,
    }
