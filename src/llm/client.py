from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from openai import OpenAI


class LLMError(RuntimeError):
    pass


class LLMJSONParseError(LLMError):
    """Raised when the LLM returned a non-empty response that is not valid JSON."""
    pass


@dataclass
class LLMResponse:
    data: dict[str, Any]
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


# wrapper around OpenAI client to enforce consistent response parsing and error handling
class OpenAIClient:
    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini") -> None:
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise LLMError("Missing OPENAI_API_KEY env var")

        self._client = OpenAI(api_key=key)
        self._model = model

    def generate_json(self, *, system: str, user: str) -> LLMResponse:
        if not isinstance(system, str):
            raise LLMError(f"System prompt must be a string, got {type(system)}")
        if not isinstance(user, str):
            raise LLMError(f"User prompt must be a string, got {type(user)}")

        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.2,
                response_format={"type": "json_object"},
            )
        except Exception as e:
            raise LLMError(f"OpenAI request failed: {e}")

        content = (resp.choices[0].message.content or "").strip()
        if not content:
            raise LLMError("OpenAI returned empty response")

        try:
            data = json.loads(content)
        except Exception as e:
            raise LLMJSONParseError(
                f"Failed to parse json: {e}. Raw response starts with: {content[:300]!r}"
            )

        usage = resp.usage
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0

        return LLMResponse(
            data=data,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
