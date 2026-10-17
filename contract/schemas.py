"""CONTRACT (lead only): the public API JSON from the ТЗ (section 10) and the internal result of `vision.analyze`.

Public models are exactly the ТЗ schema — the jury parses them automatically, so no extra fields there.
Everything the UI needs on top (rule, quote, confidence) lives in the internal models and in /api/v1/jobs/{id}/details.
"""

from typing import Literal

from pydantic import BaseModel, Field

Status = Literal["processing", "done", "failed"]
Verdict = Literal["violation", "no_violation"]


# --- public: what the jury gets from GET /api/v1/jobs/{job_id} -----------------------------------------------
class EvidenceFrame(BaseModel):
    timestamp_sec: float
    image_base64: str  # JPEG, base64, no "data:" prefix


class Violation(BaseModel):
    type: str  # one of the fixed names in docs/VIOLATIONS.md, always spelled the same
    timecode_sec: float
    description: str  # what is seen and why it may be a violation; ends with the rule reference
    evidence_frames: list[EvidenceFrame] = Field(min_length=1, max_length=3)


class JobResult(BaseModel):
    job_id: str
    status: Status
    verdict: Verdict | None = None  # set when status == "done"
    violations: list[Violation] = Field(default_factory=list)


class AnalyzeAccepted(BaseModel):  # response of POST /api/v1/analyze
    job_id: str


# --- internal: vision.analyze(path) -> VideoAnalysis; UI reads it via /api/v1/jobs/{id}/details ----------------
class RuleRef(BaseModel):
    act: str  # full title of the act
    approved: str  # "постановление Правления НБРК от … № …"
    ref: str  # "п. 20" / "п. 24, подп. 3)"
    quote: str  # verbatim from data/laws/rules_catalog.json
    url: str  # adilet link
    basis: Literal["direct", "interpretation"]


class ViolationDetail(Violation):
    type_id: str  # T1..T12, T0 = «Иное нарушение кассовой дисциплины»
    confidence: float = Field(ge=0, le=1)
    start_sec: float | None = None
    end_sec: float | None = None
    rules: list[RuleRef] = Field(default_factory=list)
    why: str = ""  # how the rule relates to what is seen (from the catalog)


class VideoAnalysis(BaseModel):
    verdict: Verdict
    violations: list[ViolationDetail] = Field(default_factory=list)
    duration_sec: float
    processing_sec: float
    engine: str  # "mock" until the real pipeline lands, then e.g. "rules+qwen-vl-4b"


class JobDetails(BaseModel):  # GET /api/v1/jobs/{job_id}/details — for our UI, not for the jury
    job_id: str
    status: Status
    progress: float = 0.0  # 0..1
    stage: str = ""  # "decode", "detect", "verify", "report"
    error: str | None = None
    filename: str = ""
    analysis: VideoAnalysis | None = None


def to_public(job_id: str, status: Status, analysis: VideoAnalysis | None) -> JobResult:
    """Strip internal fields: the jury sees exactly the ТЗ schema."""
    if status != "done" or analysis is None:
        return JobResult(job_id=job_id, status=status)
    return JobResult(
        job_id=job_id, status="done", verdict=analysis.verdict,
        violations=[Violation(type=v.type, timecode_sec=v.timecode_sec, description=v.description,
                              evidence_frames=v.evidence_frames) for v in analysis.violations],
    )
