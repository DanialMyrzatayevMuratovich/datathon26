"""Runtime settings. Secrets stay in the environment or the ignored .env.local."""

import math
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    api_key: str
    jobs_dir: Path = ROOT / "runs" / "jobs"
    ui_dir: Path = ROOT / "ui"
    engine: str = "mock"
    workers: int = 2
    capacity: int = 8
    timeout_sec: float = 270
    retention_sec: float = 86400
    max_upload_bytes: int = 200 * 1024 * 1024

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv(ROOT / ".env.local", override=False)
        return cls(
            api_key=os.getenv("API_KEY", "").strip(),
            jobs_dir=Path(os.getenv("JOBS_DIR", str(ROOT / "runs" / "jobs"))).resolve(),
            engine=os.getenv("VISION_ENGINE", "mock"),
            workers=int(os.getenv("SERVICE_WORKERS", "2")),
            capacity=int(os.getenv("SERVICE_CAPACITY", "8")),
            timeout_sec=float(os.getenv("SERVICE_TIMEOUT_SEC", "270")),
            retention_sec=float(os.getenv("SERVICE_RETENTION_SEC", "86400")),
        )

    def validate(self) -> None:
        if not self.api_key:
            raise ValueError("API_KEY is required. Run: uv run python -m service.cli init")
        if self.engine not in {"mock", "real"}:
            raise ValueError("VISION_ENGINE must be mock or real")
        if not 1 <= self.workers <= self.capacity <= 64:
            raise ValueError("Expected 1 <= SERVICE_WORKERS <= SERVICE_CAPACITY <= 64")
        if (
            not 0 < self.timeout_sec <= 270 or not math.isfinite(self.retention_sec)
            or self.retention_sec <= 0 or self.max_upload_bytes <= 0
        ):
            raise ValueError("Timeout must be in (0, 270]; retention and upload limit must be positive")
