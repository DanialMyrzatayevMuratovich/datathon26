"""Contract of the agent artifacts: log lines, model scores, run summary, advice. Change only via the lead."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

Status = Literal["ok", "warn", "fail", "skip"]


class AgentStep(BaseModel):
    ts: datetime
    run_id: str
    step: int
    tool: str
    status: Status
    summary: str
    decision: str | None = None
    reason: str | None = None
    args: dict = Field(default_factory=dict)
    duration_ms: int = 0


class ModelScore(BaseModel):
    model: str
    status: Status
    mean: float | None = None
    std: float | None = None
    folds: list[float] = Field(default_factory=list)
    seconds: float = 0.0
    best_iteration: int | None = None
    error: str | None = None


class Advice(BaseModel):
    """Next experiments. The LLM may only fill this shape; it never touches data or files."""

    ideas: list[str] = Field(default_factory=list)  # callers keep the first 8
    risks: list[str] = Field(default_factory=list)  # callers keep the first 5


class RunSummary(BaseModel):
    run_id: str
    author: str
    note: str = ""
    started: datetime
    seconds: float
    task: str
    metric: str
    greater_is_better: bool
    cv: str
    folds: int
    n_train: int
    n_test: int
    n_features: int
    best_model: str
    best_score: float
    best_std: float
    used_for_submission: str  # model name or "ensemble(a+b)"
    submission_score: float
    leaderboard: list[ModelScore]
    top_features: list[tuple[str, float]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    advice: Advice = Field(default_factory=Advice)
    advice_mode: Literal["llm", "rules"] = "rules"
    submission: str | None = None
    model_path: str | None = None
    command: str = ""
