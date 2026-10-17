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

    def request(path_url: str, data=None):
        headers = {"X-API-Key": key}
        if data is not None:
            headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
        req = urllib.request.Request(url + path_url, data=data, headers=headers)
        with opener().open(req, timeout=40) as response:
            if strict and response.status != 200:
                raise ValueError("Expected HTTP 200")
            return json.load(response)

    accepted_response = request("/api/v1/analyze", body)
    if strict and set(accepted_response) != {"job_id"}:
        raise ValueError("Upload response must contain only job_id")
    job_id = accepted_response["job_id"]
    if not isinstance(job_id, str) or not job_id or urllib.parse.quote(job_id, safe="") != job_id:
        raise ValueError("Invalid job identifier")
    del body
    accepted = time.monotonic() - started
    while time.monotonic() - started < 300:
        public = request(f"/api/v1/jobs/{job_id}")
        if strict:
            validate_public(public, job_id)
        if public["status"] != "processing":
            details = request(f"/api/v1/jobs/{job_id}/details")
            return {
                "job_id": job_id, "status": public["status"], "accepted_sec": round(accepted, 3),
                "elapsed_sec": round(time.monotonic() - started, 3), "verdict": public.get("verdict"),
                "violations": len(public.get("violations", [])),
                "evidence_frames": sum(len(v["evidence_frames"]) for v in public.get("violations", [])),
                "engine": (details.get("analysis") or {}).get("engine"), "error": details.get("error"),
            }
        time.sleep(0.25)
    return {"job_id": job_id, "status": "timeout", "elapsed_sec": round(time.monotonic() - started, 3)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init", help="Create a private API key without printing it")
    doctor = sub.add_parser("doctor", help="Check tools, platform and local weights without loading models")
    doctor.add_argument("--engine", choices=["mock", "real"])
    doctor.add_argument("--require-ui", action="store_true")
    check = sub.add_parser("check", help="Upload videos concurrently and wait for results")
    check.add_argument("video", type=Path, nargs="?")
    check.add_argument("--synthetic", action="store_true", help="Generate a 2-second test pattern with FFmpeg")
    check.add_argument("--url", default="http://127.0.0.1:8000")
    check.add_argument("--count", type=int, default=5)
    check.add_argument("--report", type=Path)
    check.add_argument("--strict", action="store_true", help="Validate exact public fields, JPEG evidence and HTTP 401")
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
    if bool(args.video) == args.synthetic:
        parser.error("Choose a video path OR --synthetic")
    with tempfile.TemporaryDirectory(prefix="forte-video-check-") as temporary:
        path = args.video
        if args.synthetic:
            path = Path(temporary) / "check.mov"
            subprocess.run([
                "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=10",
                "-t", "2", "-an", "-c:v", "mpeg4", "-y", str(path),
            ], check=True, timeout=30)
        if not path.is_file() or path.stat().st_size > 200 * 1024 * 1024:
            parser.error("Video must exist and be at most 200 MiB")
        auth_checks = check_auth(args.url.rstrip("/"), key) if args.strict else []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.count) as pool:
            futures = [pool.submit(check_one, args.url.rstrip("/"), key, path, args.strict) for _ in range(args.count)]
            rows = []
            for future in futures:
                try:
                    rows.append(future.result())
                except (OSError, ValueError, KeyError) as exc:
                    rows.append({"status": "request_failed", "error_type": type(exc).__name__})
        report = {"synthetic": args.synthetic, "strict": args.strict, "count": args.count,
                  "auth_checks": auth_checks, "results": rows}
        rendered = json.dumps(report, ensure_ascii=False, indent=2)
        print(rendered)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(rendered + "\n", encoding="utf-8")
        if any(not check["ok"] for check in auth_checks) or any(
            row["status"] != "done" or row["elapsed_sec"] >= 300 for row in rows
        ):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
