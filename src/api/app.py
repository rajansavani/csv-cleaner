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

@app.get("/", include_in_schema=False)
def home() -> HTMLResponse:
    return HTMLResponse(_render_home())


def _render_home() -> str:
    return """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>csv cleaner</title>
<style>
:root {
  --bg: #0d100f;
  --bg-elev: #141816;
  --text: #e8e8e8;
  --text-muted: #888888;
  --text-faint: #5a5e5c;
  --border: #2a2d2c;
  --border-strong: #3a3d3c;
  --accent: #4ade80;
  --accent-dim: #22c55e;
  --error: #ef4444;
  --warn: #d97706;
  --mono: ui-monospace, SFMono-Regular, "SF Mono", Consolas, "Liberation Mono", monospace;
  --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}
* { box-sizing: border-box; }
html, body { background: var(--bg); color: var(--text); margin: 0; padding: 0; }
body { font-family: var(--sans); font-size: 14px; line-height: 1.55; }
.container { max-width: 720px; margin: 0 auto; padding: 56px 24px 96px; }
a { color: var(--text-muted); text-decoration: none; border-bottom: 1px solid transparent; }
a:hover { color: var(--accent); border-bottom-color: var(--accent); }
.mono { font-family: var(--mono); }
.muted { color: var(--text-muted); }
.faint { color: var(--text-faint); }

header { margin-bottom: 48px; }
header .title { font-family: var(--mono); font-size: 13px; color: var(--text); letter-spacing: 0.02em; }
header .title::before { content: "$ "; color: var(--accent); }
header .subtitle { color: var(--text-muted); margin-top: 8px; font-size: 13px; }
header .links { margin-top: 16px; font-family: var(--mono); font-size: 12px; }
header .links a { margin-right: 16px; }

section { margin-bottom: 40px; }
.section-label { font-family: var(--mono); font-size: 11px; color: var(--text-faint); text-transform: lowercase; letter-spacing: 0.08em; margin-bottom: 12px; }

.drop {
  border: 1px dashed var(--border-strong);
  padding: 56px 24px;
  text-align: center;
  cursor: pointer;
  transition: border-color 120ms, background 120ms;
  border-radius: 2px;
}
.drop:hover { border-color: var(--accent); }
.drop.dragging { border-style: solid; border-color: var(--accent); background: rgba(74, 222, 128, 0.04); }
.drop .label { font-family: var(--mono); font-size: 13px; color: var(--text); }
.drop .label .accent { color: var(--accent); }
.drop-caption { font-size: 12px; color: var(--text-faint); margin-top: 10px; }

.file-row {
  display: flex; align-items: center; justify-content: space-between;
  border: 1px solid var(--border); padding: 12px 16px; font-family: var(--mono); font-size: 13px;
}
.file-row .name { color: var(--text); }
.file-row .size { color: var(--text-muted); margin-left: 12px; font-size: 12px; }
.file-row .remove { color: var(--text-muted); font-size: 12px; cursor: pointer; }
.file-row .remove:hover { color: var(--error); }

.actions { margin-top: 16px; display: flex; align-items: center; gap: 16px; }
button.primary {
  font-family: var(--mono); font-size: 13px;
  background: transparent; color: var(--accent);
  border: 1px solid var(--accent); padding: 10px 20px; cursor: pointer;
  border-radius: 2px; letter-spacing: 0.02em;
}
button.primary:hover { background: rgba(74, 222, 128, 0.08); }
button.primary:disabled { opacity: 0.5; cursor: not-allowed; }

.status { font-family: var(--mono); font-size: 13px; color: var(--text-muted); display: flex; align-items: center; gap: 10px; }
.status .dot { width: 6px; height: 6px; background: var(--accent); border-radius: 50%; animation: pulse 1.2s infinite ease-in-out; }
.status .slow-hint { color: var(--text-faint); font-size: 12px; margin-top: 6px; display: block; }
@keyframes pulse { 0%, 100% { opacity: 0.3; } 50% { opacity: 1; } }

.summary {
  border: 1px solid var(--border); padding: 20px 20px;
  display: grid; grid-template-columns: 1fr auto; gap: 16px 24px; align-items: center;
}
.summary .verdict { display: flex; align-items: center; gap: 12px; font-family: var(--mono); font-size: 14px; }
.summary .verdict.ok { color: var(--accent); }
.summary .verdict.warn { color: var(--warn); }
.summary .verdict.error { color: var(--error); }
.summary .verdict svg { width: 16px; height: 16px; flex: 0 0 16px; }
.summary .meta { font-family: var(--mono); font-size: 12px; color: var(--text-muted); margin-top: 4px; }
.summary .download { justify-self: end; }

.timeline { position: relative; padding-left: 24px; margin-top: 24px; }
.timeline::before {
  content: ""; position: absolute; left: 6px; top: 6px; bottom: 6px;
  width: 1px; background: var(--accent-dim); opacity: 0.5;
}
.iter { position: relative; margin-bottom: 32px; }
.iter::before {
  content: ""; position: absolute; left: -22px; top: 6px;
  width: 9px; height: 9px; background: var(--bg); border: 1px solid var(--accent); border-radius: 50%;
}
.iter .trigger { font-style: italic; color: var(--text-faint); font-size: 12px; margin-bottom: 4px; }
.iter h3 { font-family: var(--mono); font-size: 16px; margin: 0 0 8px; font-weight: 500; color: var(--text); }
.iter .summary-line { color: var(--text-muted); font-size: 13px; margin-bottom: 12px; }
.chips { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 12px; }
.chip { font-family: var(--mono); font-size: 11px; color: var(--text-muted); border: 1px solid var(--border); padding: 2px 8px; border-radius: 2px; }
.exec-stats { font-family: var(--mono); font-size: 12px; color: var(--text-muted); }
.exec-stats .row { display: grid; grid-template-columns: 1fr auto; gap: 16px; padding: 4px 0; border-bottom: 1px dotted var(--border); }
.exec-stats .row:last-child { border-bottom: none; }
.exec-stats .key { color: var(--text); }
.exec-stats .val { color: var(--text-muted); white-space: pre; }

.final-note {
  margin-top: 16px; padding: 12px 16px; border-left: 2px solid var(--accent);
  background: rgba(74, 222, 128, 0.03); font-size: 13px; color: var(--text-muted); font-style: italic;
}
.final-note.warn { border-left-color: var(--warn); background: rgba(217, 119, 6, 0.04); }
.final-note.error { border-left-color: var(--error); background: rgba(239, 68, 68, 0.04); }

.preview { overflow-x: auto; border: 1px solid var(--border); }
table { width: 100%; border-collapse: collapse; font-family: var(--mono); font-size: 12px; }
th, td { text-align: left; padding: 8px 12px; border-bottom: 1px solid var(--border); white-space: nowrap; }
th { color: var(--text); font-weight: 600; background: var(--bg-elev); }
td { color: var(--text-muted); }
tr:last-child td { border-bottom: none; }

.error-box {
  border: 1px solid var(--error); border-left-width: 2px; padding: 14px 16px;
  background: rgba(239, 68, 68, 0.04); color: var(--error);
  font-family: var(--mono); font-size: 13px; margin-bottom: 24px;
}
.hidden { display: none !important; }
</style>
</head>
<body>
<div class="container">

<header>
  <div class="title">csv cleaner</div>
  <div class="subtitle">upload a messy csv → llm proposes a plan → executor runs it → reflector revises until clean</div>
  <div class="links">
    <a href="/docs">api docs</a>
    <a href="/health">health</a>
  </div>
</header>

<section id="upload-section">
  <div class="section-label">input</div>
  <div id="drop" class="drop">
    <div class="label">drop a csv here, or <span class="accent">click to choose</span></div>
  </div>
  <div class="drop-caption">comma, semicolon, tab, or pipe delimited. utf-8 / cp1252 / latin-1.</div>
  <input type="file" id="file-input" accept=".csv" style="display:none" />
  <div id="file-row" class="file-row hidden">
    <div><span class="name" id="file-name"></span><span class="size" id="file-size"></span></div>
    <div class="remove" id="file-remove">remove</div>
  </div>
  <div class="actions">
    <button id="clean-btn" class="primary hidden">clean</button>
    <div id="status" class="status hidden">
      <span class="dot"></span>
      <span><span id="status-text">running pass 1...</span><span id="slow-hint" class="slow-hint hidden">this can take 30-60 seconds for a multi-pass clean.</span></span>
    </div>
  </div>
</section>

<div id="error-area"></div>

<section id="results" class="hidden">
  <div class="section-label">result</div>
  <div id="summary" class="summary"></div>

  <div class="section-label" style="margin-top: 40px;">passes</div>
  <div id="timeline" class="timeline"></div>

  <div class="section-label" style="margin-top: 40px;">cleaned data preview</div>
  <div id="preview" class="preview"></div>
</section>

</div>

<script>
const SVG_CHECK = '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M3 8.5L6.5 12L13 4.5"/></svg>';
const SVG_WARN = '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M8 2L14.5 13.5H1.5L8 2Z"/><path d="M8 6.5V9.5"/><circle cx="8" cy="11.5" r="0.6" fill="currentColor"/></svg>';
const SVG_ERROR = '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="8" cy="8" r="6.5"/><path d="M5.5 5.5L10.5 10.5M10.5 5.5L5.5 10.5"/></svg>';

const $ = (id) => document.getElementById(id);

let selectedFile = null;
let statusTimer = null;
let slowTimer = null;

const drop = $("drop");
const fileInput = $("file-input");
const fileRow = $("file-row");
const cleanBtn = $("clean-btn");
const statusEl = $("status");

drop.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", (e) => { if (e.target.files[0]) selectFile(e.target.files[0]); });

["dragenter", "dragover"].forEach(ev => drop.addEventListener(ev, (e) => { e.preventDefault(); e.stopPropagation(); drop.classList.add("dragging"); }));
["dragleave", "drop"].forEach(ev => drop.addEventListener(ev, (e) => { e.preventDefault(); e.stopPropagation(); if (ev === "dragleave" && e.target !== drop) return; drop.classList.remove("dragging"); }));
drop.addEventListener("drop", (e) => {
  const f = e.dataTransfer.files[0];
  if (f) selectFile(f);
});

$("file-remove").addEventListener("click", () => resetFile());
cleanBtn.addEventListener("click", () => runClean());

function selectFile(f) {
  if (!f.name.toLowerCase().endsWith(".csv")) {
    showError("only .csv files are supported");
    return;
  }
  clearError();
  selectedFile = f;
  $("file-name").textContent = f.name;
  $("file-size").textContent = " · " + formatBytes(f.size);
  fileRow.classList.remove("hidden");
  drop.classList.add("hidden");
  cleanBtn.classList.remove("hidden");
}

function resetFile() {
  selectedFile = null;
  fileInput.value = "";
  fileRow.classList.add("hidden");
  drop.classList.remove("hidden");
  cleanBtn.classList.add("hidden");
  $("results").classList.add("hidden");
  clearError();
}

function formatBytes(n) {
  if (n < 1024) return n + " B";
  if (n < 1024 * 1024) return (n / 1024).toFixed(1) + " KB";
  return (n / 1024 / 1024).toFixed(1) + " MB";
}

const STATUS_MESSAGES = ["uploading...", "profiling...", "planning...", "executing pass 1...", "reflecting...", "executing revised plan...", "reflecting...", "finalizing..."];

function startStatusRotation() {
  let i = 0;
  $("status-text").textContent = STATUS_MESSAGES[0];
  statusTimer = setInterval(() => {
    i = Math.min(i + 1, STATUS_MESSAGES.length - 1);
    $("status-text").textContent = STATUS_MESSAGES[i];
  }, 2000);
  slowTimer = setTimeout(() => $("slow-hint").classList.remove("hidden"), 5000);
}

function stopStatusRotation() {
  if (statusTimer) clearInterval(statusTimer);
  if (slowTimer) clearTimeout(slowTimer);
  statusTimer = null;
  slowTimer = null;
  $("slow-hint").classList.add("hidden");
}

async function runClean() {
  if (!selectedFile) return;
  clearError();
  $("results").classList.add("hidden");
  cleanBtn.classList.add("hidden");
  statusEl.classList.remove("hidden");
  startStatusRotation();

  try {
    const fd = new FormData();
    fd.append("file", selectedFile);
    const resp = await fetch("/clean/llm", { method: "POST", body: fd });
    const body = await resp.json().catch(() => ({ detail: "non-json response from server" }));
    if (!resp.ok) {
      const detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail || body);
      throw new Error(`http ${resp.status}: ${detail}`);
    }
    renderResults(body);
  } catch (err) {
    showError(err.message || String(err));
  } finally {
    stopStatusRotation();
    statusEl.classList.add("hidden");
    cleanBtn.classList.remove("hidden");
  }
}

function clearError() { $("error-area").innerHTML = ""; }
function showError(msg) {
  $("error-area").innerHTML = `<div class="error-box">${escapeHtml(msg)}</div>`;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#39;"}[c]));
}

function renderResults(body) {
  $("results").classList.remove("hidden");
  renderSummary(body);
  renderTimeline(body);
  renderPreview(body.cleaned_preview_rows || []);
}

function renderSummary(body) {
  const fr = body.final_reflection || {};
  const m = body.metrics || {};
  let cls = "ok", icon = SVG_CHECK, label = "data is clean";

  if (fr.decision === "mark_clean") { cls = "ok"; icon = SVG_CHECK; label = "data is clean"; }
  else if (fr.decision === "flag_unrecoverable") { cls = "warn"; icon = SVG_WARN; label = "data has unrecoverable issues"; }
  else if (fr.decision === "max_iterations_exceeded") { cls = "warn"; icon = SVG_WARN; label = `stopped after ${body.total_iterations} passes`; }
  else if (fr.decision === "reflection_failed") { cls = "error"; icon = SVG_ERROR; label = `system error: ${fr.stage || "unknown"}`; }

  const tokens = m.total_tokens || 0;
  const cost = (m.estimated_cost_usd || 0).toFixed(4);
  const secs = (m.wall_clock_seconds || 0).toFixed(2);

  $("summary").innerHTML = `
    <div>
      <div class="verdict ${cls}">${icon}<span>${escapeHtml(label)}</span></div>
      <div class="meta">${body.total_iterations} ${body.total_iterations === 1 ? "pass" : "passes"} · ${tokens} tokens · $${cost} · ${secs}s</div>
    </div>
    <div class="download">
      <a class="primary-link" href="/jobs/${encodeURIComponent(body.job_id)}/cleaned.csv" download><button class="primary">download cleaned csv</button></a>
    </div>
  `;
}

function renderTimeline(body) {
  const iters = body.iterations || [];
  const fr = body.final_reflection || {};
  const tl = $("timeline");
  tl.innerHTML = "";

  iters.forEach((it) => {
    const block = document.createElement("div");
    block.className = "iter";
    const trigger = it.triggering_reflection;
    const triggerHtml = trigger && trigger.reasoning
      ? `<div class="trigger">triggered by: ${escapeHtml(trigger.reasoning)}</div>` : "";

    const plan = it.plan || {};
    const actions = (plan.actions || []).map(a => a.action);
    const chipsHtml = actions.length
      ? `<div class="chips">${actions.map(a => `<span class="chip">${escapeHtml(a)}</span>`).join("")}</div>` : "";
    const summaryHtml = plan.summary ? `<div class="summary-line">${escapeHtml(plan.summary)}</div>` : "";

    const stats = renderExecStats(it.execution_report || {});

    block.innerHTML = `
      ${triggerHtml}
      <h3>pass ${it.pass}</h3>
      ${summaryHtml}
      ${chipsHtml}
      ${stats}
    `;
    tl.appendChild(block);
  });

  // final reflection note
  if (fr.reasoning || fr.error) {
    let cls = "";
    if (fr.decision === "flag_unrecoverable" || fr.decision === "max_iterations_exceeded") cls = "warn";
    else if (fr.decision === "reflection_failed") cls = "error";
    const text = fr.reasoning || fr.error || "";
    const issues = (fr.remaining_issues && fr.remaining_issues.length)
      ? `<div style="margin-top:8px;">remaining issues: ${fr.remaining_issues.map(escapeHtml).join("; ")}</div>` : "";
    const note = document.createElement("div");
    note.className = `final-note ${cls}`;
    note.innerHTML = `${escapeHtml(text)}${issues}`;
    tl.appendChild(note);
  }
}

function renderExecStats(report) {
  const applied = report.actions_applied || [];
  if (!applied.length) return "";
  const rows = [];
  for (const a of applied) {
    const key = a.action;
    const val = formatActionStat(a);
    if (val) rows.push(`<div class="row"><span class="key">${escapeHtml(key)}</span><span class="val">${escapeHtml(val)}</span></div>`);
  }
  if (!rows.length) return "";
  return `<div class="exec-stats">${rows.join("")}</div>`;
}

function formatActionStat(a) {
  if (a.action === "deduplicate_rows") return `${a.dropped_rows || 0} rows dropped`;
  if (a.action === "drop_columns") return `${(a.dropped || []).length} cols dropped`;
  if (a.action === "rename_columns") return `${Object.keys(a.applied || {}).length} renamed`;
  if (a.action === "parse_numeric" || a.action === "parse_dates") {
    const stats = a.stats || {};
    const parts = Object.entries(stats).map(([col, s]) => {
      if (s && typeof s === "object") {
        if ("changed_cells" in s) return `${col}: ${s.changed_cells} changed`;
        if ("parsed_non_null" in s) return `${col}: ${s.parsed_non_null}/${s.total} parsed`;
      }
      return null;
    }).filter(Boolean);
    return parts.join(", ");
  }
  if (a.action === "trim_whitespace") return Array.isArray(a.columns) ? `${a.columns.length} cols` : "all cols";
  if (a.action === "standardize_nulls") return `${(a.null_tokens || []).length} tokens`;
  return "";
}

function renderPreview(rows) {
  if (!rows.length) { $("preview").innerHTML = '<div style="padding:16px;color:var(--text-faint);font-family:var(--mono);font-size:12px;">no rows</div>'; return; }
  const cols = Object.keys(rows[0]);
  const head = `<thead><tr>${cols.map(c => `<th>${escapeHtml(c)}</th>`).join("")}</tr></thead>`;
  const body = `<tbody>${rows.map(r => `<tr>${cols.map(c => `<td>${escapeHtml(r[c] ?? "")}</td>`).join("")}</tr>`).join("")}</tbody>`;
  $("preview").innerHTML = `<table>${head}${body}</table>`;
}
</script>
</body>
</html>
"""


app.include_router(
    router,
    tags=["csv-cleaner"],
)
