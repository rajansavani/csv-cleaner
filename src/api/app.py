from __future__ import annotations

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from fastapi.responses import HTMLResponse

from src.api.routes import router


def _custom_openapi(app: FastAPI) -> dict:
    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    schema.setdefault("info", {})
    schema["info"]["x-logo"] = {
        "url": "https://raw.githubusercontent.com/github/explore/main/topics/csv/csv.png",
        "altText": "CSVCleaner",
    }

    app.openapi_schema = schema
    return app.openapi_schema


app = FastAPI(
    title="CSVCleaner",
    version="0.1.0",
    description=(
        "Upload a CSV, get a quick profile, then clean + validate it using an LLM-generated plan.\n\n"
        "This service is designed to be safe-by-default: the LLM only proposes a plan, and the executor "
        "applies deterministic pandas transforms."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.openapi = lambda: _custom_openapi(app)  # type: ignore[assignment]

# small landing page
@app.get("/", include_in_schema=False)
def home() -> HTMLResponse:
    html = """
    <html>
      <head>
        <title>CSVCleaner</title>
        <style>
          body { font-family: ui-sans-serif, system-ui; max-width: 900px; margin: 40px auto; padding: 0 16px; }
          .card { border: 1px solid #e5e7eb; border-radius: 12px; padding: 18px; }
          a { text-decoration: none; }
          code { background: #f3f4f6; padding: 2px 6px; border-radius: 6px; }
        </style>
      </head>
      <body>
        <h1>CSVCleaner</h1>
        <div class="card">
          <p>FastAPI service for profiling and cleaning messy CSVs using a structured LLM plan.</p>
          <p>
            <a href="/docs">Open Swagger UI</a> ·
            <a href="/redoc">Open ReDoc</a> ·
            <a href="/health">Health</a>
          </p>
          <p>Try: <code>POST /clean/basic</code> or <code>POST /clean/llm</code></p>
        </div>
      </body>
    </html>
    """
    return HTMLResponse(html)


app.include_router(
    router,
    tags=["csv-cleaner"],
)
