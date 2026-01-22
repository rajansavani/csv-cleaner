from __future__ import annotations

from fastapi import FastAPI

from src.api.routes import router


app = FastAPI(
    title="CSVCleaner",
    version="0.1.0",
    description="Upload a CSV, get a quick profile, then clean + validate it using LLMs",
)

app.include_router(router)
