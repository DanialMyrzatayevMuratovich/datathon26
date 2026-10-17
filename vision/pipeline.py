"""Entry point of the video pipeline. Real engine: TODO (danial). Mock engine: works today, same contract."""

import base64
import json
import os
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

from contract.schemas import EvidenceFrame, RuleRef, VideoAnalysis, ViolationDetail

ROOT = Path(__file__).resolve().parent.parent
CATALOG = ROOT / "data" / "laws" / "rules_catalog.json"
Progress = Callable[[float, str], None]


def probe_duration(path: str | Path) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True, check=True)
    return float(r.stdout.strip())


def frame_jpeg_base64(path: str | Path, t: float, max_side: int = 720) -> str:
    """One JPEG frame at second t, scaled so the long side <= max_side (keeps the API payload small)."""
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(path), "-frames:v", "1",
         "-vf", f"scale='if(gt(iw,ih),min({max_side},iw),-2)':'if(gt(iw,ih),-2,min({max_side},ih))'",
         "-f", "image2", "-c:v", "mjpeg", "-q:v", "4", "pipe:1"],
        capture_output=True, check=True)
    return base64.b64encode(r.stdout).decode("ascii")


def rules_for(type_id: str) -> tuple[str, list[RuleRef], str]:
    catalog = {t["type_id"]: t for t in json.loads(CATALOG.read_text(encoding="utf-8"))}
    t = catalog[type_id]
    rules = [RuleRef(**{k: r[k] for k in ("act", "approved", "ref", "quote", "url", "basis")}) for r in t["rules"]]
    return t["type"], rules, t["why"]


def describe_rule(rules: list[RuleRef]) -> str:
    r = rules[0]
    verb = "Нарушен" if r.basis == "direct" else "Может противоречить"
    short = r.approved.replace("постановление Правления Национального Банка Республики Казахстан", "пост. НБРК")
    return f"{verb} {r.ref} Правил ({short})."


def _mock(path: str | Path, progress: Progress | None) -> VideoAnalysis:
    started = time.perf_counter()
    dur = probe_duration(path)
    if progress:
        progress(0.5, "detect")
    t = round(min(5.0, dur / 2), 1)
    type_name, rules, why = rules_for("T1")
    v = ViolationDetail(
        type=type_name, type_id="T1", timecode_sec=t, start_sec=t, end_sec=min(dur, t + 3), confidence=0.5,
        description=f"[ЗАГЛУШКА] Кассир покидает рабочее место, наличные остаются на столе. {describe_rule(rules)}",
        evidence_frames=[EvidenceFrame(timestamp_sec=t, image_base64=frame_jpeg_base64(path, t))],
        rules=rules, why=why,
    )
    if progress:
        progress(1.0, "report")
    return VideoAnalysis(verdict="violation", violations=[v], duration_sec=round(dur, 2),
                         processing_sec=round(time.perf_counter() - started, 2), engine="mock")


def analyze(path: str | Path, progress: Progress | None = None) -> VideoAnalysis:
    """VISION_ENGINE=real -> detection + rules + local VLM (vision.engine); mock -> fixed demo result."""
    if os.getenv("VISION_ENGINE", "mock") == "real":
        from vision import engine

        return engine.run(path, progress)
    return _mock(path, progress)
