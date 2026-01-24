from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# define action schemas

class DropColumns(BaseModel):
    action: Literal["drop_columns"] = "drop_columns"
    columns: list[str] = Field(default_factory=list, description="columns to drop")

class RenameColumns(BaseModel):
    action: Literal["rename_columns"] = "rename_columns"
    mapping: dict[str, str] = Field(default_factory=dict, description="old_name -> new_name")

class StandardizeNulls(BaseModel):
    action: Literal["standardize_nulls"] = "standardize_nulls"
    null_tokens: list[str] = Field(
        default_factory=lambda: ["", "na", "n/a", "null", "none", "nan"],
        description="strings to treat as missing"
    )

class ParseNumeric(BaseModel):
    action: Literal["parse_numeric"] = "parse_numeric"
    columns: list[str] = Field(default_factory=list)
    numeric_type: Literal["int", "float"] = "float"
    allow_currency: bool = Field(default=False, description="strip currency symbols/commas")
    allow_thousands_separators: bool = Field(default=True, description="handle 1,234 or 1.234.567 formats")
    fix_common_typos: bool = Field(default=True, description="fix small issues like letter O vs. 0 in numbers")

class ParseDates(BaseModel):
    action: Literal["parse_dates"] = "parse_dates"
    columns: list[str] = Field(default_factory=list)
    day_first: bool = Field(default=False, description ="useful for dd-mm-yyyy style dates")
    output_format: Literal["iso_date", "year"] = "iso_date"

class TrimWhitespace(BaseModel):
    action: Literal["trim_whitespace"] = "trim_whitespace"
    columns: list[str] | None = Field(default=None, description="if null, apply to all columns")

class DeduplicateRows(BaseModel):
    action: Literal["deduplicate_rows"] = "deduplicate_rows"
    subset: list[str] | None = Field(default=None, description="columns to dedupe on; null means full-row duplicates")

Action = DropColumns | RenameColumns | StandardizeNulls | ParseNumeric | ParseDates | TrimWhitespace | DeduplicateRows

# define validation schemas
class RangeRule(BaseModel):
    column: str
    min: float | None = None
    max: float | None = None

class EnumRule(BaseModel):
    column: str
    allowed: list[str]

class RequiredRule(BaseModel):
    column: str

class ValidationSpec(BaseModel):
    required: list[RequiredRule] = Field(default_factory=list)
    ranges: list[RangeRule] = Field(default_factory=list)
    enums: list[EnumRule] = Field(default_factory=list)

# define overall cleaning plan schema
class CleaningPlan(BaseModel):
    """
    LLM-produced plan describing how to clean a dataset.
    """
    version: str = "1"
    summary: str = Field(default="", description="one-paragraph description of what the plan will do")
    actions: list[Action] = Field(default_factory=list)
    validations: ValidationSpec = Field(default_factory=ValidationSpec)

    @classmethod
    def json_schema(cls) -> dict:
        return cls.model_json_schema()
    