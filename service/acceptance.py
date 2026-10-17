"""Judge-facing schema checks. Inspect evidence locally; never print its contents."""

import base64
import io
import math
import urllib.error
import urllib.request

from PIL import Image

from contract.schemas import JobResult


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # X-API-Key must never follow redirects to another host or to plain HTTP.
        return None


def opener():
    return urllib.request.build_opener(NoRedirect())


def validate_public(data: dict, job_id: str) -> None:
    result = JobResult.model_validate(data, strict=True)
    if result.job_id != job_id:
        raise ValueError("Result job_id does not match the accepted job")
    fields = {"job_id", "status", "verdict", "violations"} if result.status == "done" else {"job_id", "status"}
    if set(data) != fields:
        raise ValueError("Public result has missing or extra fields")
    if result.status != "done":
        return
    if result.verdict not in {"violation", "no_violation"}:
        raise ValueError("Completed result is missing verdict")
    if (result.verdict == "violation") != bool(result.violations):
        raise ValueError("Verdict contradicts the violation list")
    for violation in data["violations"]:
        if set(violation) != {"type", "timecode_sec", "description", "evidence_frames"}:
            raise ValueError("Violation has missing or extra fields")
        if not math.isfinite(violation["timecode_sec"]) or violation["timecode_sec"] < 0:
            raise ValueError("Invalid violation timestamp")
        for frame in violation["evidence_frames"]:
            if set(frame) != {"timestamp_sec", "image_base64"}:
                raise ValueError("Evidence frame has missing or extra fields")
            if not math.isfinite(frame["timestamp_sec"]) or frame["timestamp_sec"] < 0:
                raise ValueError("Invalid frame timestamp")
            raw = base64.b64decode(frame["image_base64"], validate=True)
            with Image.open(io.BytesIO(raw)) as image:
                if image.format != "JPEG":
                    raise ValueError("Evidence must be JPEG")
                image.verify()


def check_auth(url: str, key: str) -> list[dict]:
    checks = []
    for path, method in [("/api/v1/analyze", "POST"), ("/api/v1/jobs/unknown", "GET"),
                         ("/api/v1/jobs/unknown/details", "GET"), ("/api/v1/jobs/unknown/video", "GET")]:
        for label, headers in [("missing", {}), ("invalid", {"X-API-Key": key + "-invalid"})]:
            request = urllib.request.Request(url + path, method=method, headers=headers)
            try:
                with opener().open(request, timeout=10) as response:
                    status = response.status
            except urllib.error.HTTPError as exc:
                status = exc.code
                exc.close()
            checks.append({"path": path, "key": label, "status": status, "ok": status == 401})
    return checks
