"""Local key setup and a repeatable concurrent API check (no cloud inference)."""

import argparse
import concurrent.futures
import json
import os
import secrets
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import replace
from pathlib import Path

from dotenv import dotenv_values, load_dotenv

from service.acceptance import check_auth, opener, validate_public
from service.config import ROOT, Settings
from service.preflight import inspect


def init_key(path: Path = ROOT / ".env.local") -> None:
    if path.exists() and dotenv_values(path).get("API_KEY"):
        print("API_KEY уже задан в .env.local; значение сохранено.")
        return
    text = path.read_text("utf-8") if path.exists() else ""
    # Replace an empty placeholder instead of leaving duplicate API_KEY entries.
    lines = [line for line in text.splitlines() if not line.strip().startswith("API_KEY=")]
    lines.append("API_KEY=" + secrets.token_urlsafe(32))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if os.name != "nt":
        path.chmod(0o600)
    print("API_KEY сохранён в .env.local. Передайте его жюри лично; не коммитьте этот файл.")


def check_one(url: str, key: str, path: Path, strict: bool = False) -> dict:
    boundary = "video-check-" + secrets.token_hex(16)
    body = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="check.mov"\r\n'
        "Content-Type: video/quicktime\r\n\r\n"
    ).encode() + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    started = time.monotonic()
    deadline = started + 300
    job_id = None
    accepted = None

    def request(path_url: str, data=None):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Acceptance deadline exceeded")
        headers = {"X-API-Key": key}
        if data is not None:
            headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
        req = urllib.request.Request(url + path_url, data=data, headers=headers)
        # Upload may legitimately take >40s. Polls retain a short idle timeout,
        # always bounded by the same overall acceptance budget.
        with opener().open(req, timeout=remaining if data is not None else min(40, remaining)) as response:
            if strict and response.status != 200:
                raise ValueError("Expected HTTP 200")
            value = json.load(response)
        # urllib's socket timeout is NOT a total wall deadline (slow drips).
        # Reject late responses even when individual reads did not time out.
        if time.monotonic() >= deadline:
            raise TimeoutError("Acceptance deadline exceeded")
        return value

    try:
        accepted_response = request("/api/v1/analyze", body)
        if strict and set(accepted_response) != {"job_id"}:
            raise ValueError("Upload response must contain only job_id")
        job_id = accepted_response["job_id"]
        if not isinstance(job_id, str) or not job_id or urllib.parse.quote(job_id, safe="") != job_id:
            raise ValueError("Invalid job identifier")
        del body
        accepted = time.monotonic() - started
        while time.monotonic() < deadline:
            public = request(f"/api/v1/jobs/{job_id}")
            if strict:
                validate_public(public, job_id)
            elapsed = time.monotonic() - started
            if elapsed >= 300:
                raise TimeoutError("Acceptance deadline exceeded")
            if public["status"] != "processing":
                row = {
                    "job_id": job_id, "status": public["status"], "accepted_sec": round(accepted, 3),
                    "elapsed_sec": round(elapsed, 3), "verdict": public.get("verdict"),
                    "violations": len(public.get("violations", [])),
                    "evidence_frames": sum(len(v["evidence_frames"]) for v in public.get("violations", [])),
                }
                try:
                    details = request(f"/api/v1/jobs/{job_id}/details")
                    analysis = details.get("analysis") or {}
                    row.update({
                        "engine": analysis.get("engine"), "error": details.get("error"),
                        "events": [{"type_id": v["type_id"], "timecode_sec": v["timecode_sec"]}
                                   for v in analysis.get("violations", [])],
                    })
                except (OSError, ValueError, KeyError) as exc:
                    # Batch consumers must not treat an unverified result with
                    # missing internal events as a complete model prediction.
                    row["public_status"] = row["status"]
                    row["status"] = "verification_failed"
                    row["verification_error"] = type(exc).__name__
                row["verification_elapsed_sec"] = round(time.monotonic() - started, 3)
                return row
            time.sleep(min(0.25, max(0, deadline - time.monotonic())))
    except TimeoutError:
        pass
    return {"job_id": job_id, "status": "timeout", "accepted_sec": accepted,
            "elapsed_sec": round(time.monotonic() - started, 3)}


def health_engine(url: str) -> str:
    with opener().open(url.rstrip("/") + "/api/v1/health", timeout=10) as response:
        health = json.load(response)
    if health.get("ok") is not True or health.get("engine") not in {"mock", "real"}:
        raise ValueError("Service health is not ready")
    return health["engine"]


def passed(report: dict) -> bool:
    if any(not check["ok"] for check in report["auth_checks"]):
        return False
    if len(report["results"]) != report["count"]:
        return False
    for row in report["results"]:
        if row["status"] != "done" or row["elapsed_sec"] >= 300 or row.get("verification_error"):
            return False
        if report.get("expected_engine") == "real" and (not row.get("engine") or row["engine"] == "mock"):
            return False
        if report.get("expected_engine") == "mock" and row.get("engine") != "mock":
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init", help="Create a private API key without printing it")
    doctor = sub.add_parser("doctor", help="Check tools, platform and local weights without loading models")
    doctor.add_argument("--engine", choices=["mock", "real"])
    doctor.add_argument("--require-ui", action="store_true")
    check = sub.add_parser("check", help="Upload videos concurrently and wait for results")
    check.add_argument("videos", type=Path, nargs="*", help="One video repeated, or --count distinct input files")
    check.add_argument("--synthetic", action="store_true", help="Generate a 2-second test pattern with FFmpeg")
    check.add_argument("--url", default="http://127.0.0.1:8000")
    check.add_argument("--count", type=int, default=5)
    check.add_argument("--report", type=Path)
    check.add_argument("--strict", action="store_true", help="Validate exact public fields, JPEG evidence and HTTP 401")
    check.add_argument("--expect-engine", choices=["mock", "real"], help="Reject the wrong server/analysis engine")
    args = parser.parse_args()
    if args.command == "init":
        init_key()
        return
    if args.command == "doctor":
        settings = Settings.from_env()
        if args.engine:
            settings = replace(settings, engine=args.engine)
        report = inspect(settings, require_ui=args.require_ui)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        raise SystemExit(0 if report["ok"] else 1)
    load_dotenv(ROOT / ".env.local", override=False)
    key = os.getenv("API_KEY", "").strip()
    if not key:
        parser.error("API_KEY is missing; run init first")
    if not 1 <= args.count <= 8:
        parser.error("--count must be between 1 and 8")
    parsed = urllib.parse.urlparse(args.url)
    if (
        parsed.scheme not in {"http", "https"} or not parsed.hostname
        or parsed.username or parsed.password or parsed.query or parsed.fragment
    ):
        parser.error("Use an HTTP(S) service URL without credentials, query or fragment")
    if parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        parser.error("Use HTTPS when sending an API key to a remote host")
    if bool(args.videos) == args.synthetic:
        parser.error("Choose video paths OR --synthetic")
    if len(args.videos) > 1 and len(args.videos) != args.count:
        parser.error("With multiple files, --count must equal the number of files")
    if args.expect_engine:
        try:
            current_engine = health_engine(args.url)
        except (OSError, ValueError, AttributeError) as exc:
            parser.error(f"Engine preflight failed ({type(exc).__name__})")
        if current_engine != args.expect_engine:
            parser.error(f"Expected {args.expect_engine} server, got {current_engine}; no videos uploaded")
    with tempfile.TemporaryDirectory(prefix="forte-video-check-") as temporary:
        paths = args.videos
        if args.synthetic:
            path = Path(temporary) / "check.mov"
            subprocess.run([
                "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=10",
                "-t", "2", "-an", "-c:v", "mpeg4", "-y", str(path),
            ], check=True, timeout=30)
            paths = [path]
        if any(not path.is_file() or path.stat().st_size > 200 * 1024 * 1024 for path in paths):
            parser.error("Video must exist and be at most 200 MiB")
        auth_checks = check_auth(args.url.rstrip("/"), key) if args.strict else []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.count) as pool:
            inputs = paths if len(paths) > 1 else paths * args.count
            futures = [pool.submit(check_one, args.url.rstrip("/"), key, path, args.strict) for path in inputs]
            rows = []
            for i, future in enumerate(futures):
                try:
                    row = future.result()
                except (OSError, ValueError, KeyError) as exc:
                    row = {"status": "request_failed", "error_type": type(exc).__name__}
                row.update({"input_index": i, "input_name": inputs[i].name})
                rows.append(row)
        report = {"synthetic": args.synthetic, "strict": args.strict, "count": args.count,
                  "expected_engine": args.expect_engine, "auth_checks": auth_checks, "results": rows}
        rendered = json.dumps(report, ensure_ascii=False, indent=2)
        print(rendered)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(rendered + "\n", encoding="utf-8")
        if not passed(report):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
