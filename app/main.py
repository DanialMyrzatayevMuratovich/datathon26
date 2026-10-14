"""Demo API for the jury: runs, reports, charts and live predictions of the best model.

`uv run uvicorn app.main:app --port 8000` -> http://localhost:8000 (page) and /docs (OpenAPI).
"""

import json
import os
import re
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from app.config import MODELS_DIR, OUTPUTS, ROOT, RUNS_DIR
from app.llm import llm_mode

app = FastAPI(title="Datathon'26 — решение команды", version="0.1.0")
RUN_ID = re.compile(r"^[\w.-]+$")
STATIC = ROOT / "static"
_BUNDLES: dict = {}


def _run_dir(run_id: str) -> Path:
    # run_id comes from the URL: contain it to a direct child of runs/
    if not RUN_ID.match(run_id):
        raise HTTPException(400, "bad run_id")
    path = (RUNS_DIR / run_id).resolve()
    if path.parent != RUNS_DIR.resolve() or not path.is_dir():
        raise HTTPException(404, "run not found")
    return path


@app.get("/api/health")
def health() -> dict:
    best = MODELS_DIR / "best.json"
    return {"ok": True, "llm": llm_mode(), "commit": os.getenv("COMMIT", "dev"),
            "best": json.loads(best.read_text(encoding="utf-8")) if best.exists() else None}


@app.get("/api/runs")
def runs() -> list[dict]:
    board = OUTPUTS / "leaderboard.csv"
    if not board.exists():
        return []
    df = pd.read_csv(board).fillna("")
    return df.iloc[::-1].to_dict(orient="records")


@app.get("/api/runs/{run_id}")
def run(run_id: str) -> dict:
    d = _run_dir(run_id)
    metrics = d / "metrics.json"
    log = d / "agent_log.jsonl"
    return {
        "run_id": run_id,
        "metrics": json.loads(metrics.read_text(encoding="utf-8")) if metrics.exists() else None,
        "report": (d / "report.md").read_text(encoding="utf-8") if (d / "report.md").exists() else "",
        "steps": [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines() if x.strip()]
        if log.exists() else [],
        "figures": sorted(p.name for p in d.glob("*.png")),
    }


@app.get("/api/runs/{run_id}/files/{name}")
def run_file(run_id: str, name: str):
    d = _run_dir(run_id)
    if not re.match(r"^[\w.-]+\.(png|csv|md|json)$", name):
        raise HTTPException(400, "bad file name")
    path = d / name
    if not path.is_file():
        raise HTTPException(404, "file not found")
    return FileResponse(path)


class PredictRequest(BaseModel):
    records: list[dict] = Field(min_length=1, max_length=10_000)
    model: str | None = None  # run_id; default models/best.json


@app.post("/api/predict")
def predict(req: PredictRequest) -> dict:
    from app import bundle

    key = req.model or "best"
    try:
        if key not in _BUNDLES:
            _BUNDLES[key] = bundle.load(req.model, models_dir=MODELS_DIR)
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc
    b = _BUNDLES[key]
    try:
        preds = b.predict(pd.DataFrame(req.records))
    except Exception as exc:  # bad columns / types from the client
        raise HTTPException(422, f"{type(exc).__name__}: {str(exc)[:200]}") from exc
    values = preds.tolist()
    return {"run_id": b.run_id, "task": b.task, "target": b.target, "predictions": values}


@app.get("/", response_class=HTMLResponse)
def index():
    page = STATIC / "index.html"
    if not page.exists():
        return HTMLResponse("<h1>Datathon'26</h1><p>static/index.html ещё нет</p>")
    return FileResponse(page)
