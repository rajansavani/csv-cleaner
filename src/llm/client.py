from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI

class LLMError(RuntimeError):
    pass

class OpenAIClient:
    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini") -> None:
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise LLMError("Missing OPENAI_API_KEY env var")

        self._client = OpenAI(api_key=key)
        self._model = model

    def generate_json(self, *, system: str, user: str) -> dict[str, Any]:
        """
        Ask the model for a JSON object only and parse it.
        """
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
            return json.loads(content)
        except Exception as e:
            snippet = content[:500]
            raise LLMError(f"Failed to parse json: {e}. Raw response starts with: {snippet!r}")