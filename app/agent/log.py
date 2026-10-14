"""JSONL agent log: one AgentStep per line in runs/<run_id>/agent_log.jsonl — every decision with its reason."""

import json
import time
from datetime import UTC, datetime
from pathlib import Path

from app.schemas import AgentStep


class RunLog:
    def __init__(self, run_id: str, run_dir: Path, echo: bool = True):
        self.run_id = run_id
        self.dir = run_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "agent_log.jsonl"
        self.steps: list[AgentStep] = []
        self.echo = echo

    def step(self, tool: str, status: str, summary: str, *, decision: str | None = None,
             reason: str | None = None, args: dict | None = None,
             started: float | None = None) -> AgentStep:
        entry = AgentStep(
            ts=datetime.now(UTC),
            run_id=self.run_id,
            step=len(self.steps) + 1,
            tool=tool,
            status=status,  # type: ignore[arg-type]
            summary=summary,
            decision=decision,
            reason=reason,
            args=args or {},
            duration_ms=int((time.perf_counter() - started) * 1000) if started else 0,
        )
        self.steps.append(entry)
        with self.path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(entry.model_dump(mode="json"), ensure_ascii=False) + "\n")
        if self.echo:
            mark = {"ok": "✔", "warn": "⚠", "fail": "✖", "skip": "–"}[status]
            tail = f" → {decision}" if decision else ""
            print(f"{mark} {entry.step:>2}. {tool}: {summary}{tail}", flush=True)
        return entry

    @property
    def warnings(self) -> list[str]:
        return [f"{s.tool}: {s.summary}" for s in self.steps if s.status in {"warn", "fail"}]

    @staticmethod
    def read(run_dir: Path) -> list[AgentStep]:
        path = run_dir / "agent_log.jsonl"
        if not path.exists():
            return []
        with path.open(encoding="utf-8") as f:
            return [AgentStep.model_validate_json(line) for line in f if line.strip()]
