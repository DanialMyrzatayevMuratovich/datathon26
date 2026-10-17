"""One local command: start isolated video API, check five inputs, stop our server.

Default is synthetic/mock. --real requires Apple Silicon, local weights and at
least five distinct MOV files; it includes the longest file in the supplied folder.
"""

import argparse
import json
import os
import secrets
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from service.cli import health_engine, passed
from vision.frames import info


def inputs(directory: Path):
    paths = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".mov")
    if len(paths) < 5:
        raise ValueError("Real acceptance needs at least five distinct MOV files")
    longest = max(paths, key=lambda p: info(p).duration)
    return [longest, *(p for p in paths if p != longest)][:5]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--videos", type=Path, default=Path("data/raw/open/videos"))
    parser.add_argument("--report", type=Path, default=Path("runs/jobs/smoke.json"))
    args = parser.parse_args()
    args.report = args.report.resolve()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    engine = "real" if args.real else "mock"
    # Do not kill or connect to an unrelated server on the agreed development port.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 8000))
    creation = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    with tempfile.TemporaryDirectory(prefix="video-smoke-", dir=args.report.parent) as temporary:
        directory = Path(temporary)
        videos = inputs(args.videos) if args.real else []
        if not args.real:
            for i in range(5):
                path = directory / f"synthetic-{i}.mov"
                subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                                f"testsrc2=size={320 + 16 * i}x240:rate=10", "-t", str(1 + .25 * i),
                                "-an", "-c:v", "mpeg4", "-threads", "2", "-y", str(path)],
                               check=True, timeout=30, creationflags=creation)
                videos.append(path)
        env = {**os.environ, "API_KEY": secrets.token_urlsafe(32), "VISION_ENGINE": engine,
               "JOBS_DIR": str(directory / "jobs")}
        env.pop("DT_CACHE", None)
        env.pop("DT_BATCH", None)
        log_path = args.report.with_suffix(".server.log")
        with log_path.open("w", encoding="utf-8") as log:
            server = subprocess.Popen([sys.executable, "-m", "uvicorn", "service.main:app", "--host",
                                       "127.0.0.1", "--port", "8000", "--workers", "1"],
                                      env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                      creationflags=creation)
            try:
                for _ in range(60):
                    if server.poll() is not None:
                        raise RuntimeError(f"Smoke server exited; see {log_path}")
                    try:
                        if health_engine("http://127.0.0.1:8000") == engine:
                            break
                    except (OSError, ValueError):
                        pass
                    time.sleep(.5)
                else:
                    raise TimeoutError(f"Smoke server startup failed; see {log_path}")
                subprocess.run([sys.executable, "-m", "service.cli", "check", *map(str, videos),
                                "--count", "5", "--strict", "--expect-engine", engine,
                                "--report", str(args.report)], env=env, check=True, timeout=360,
                               creationflags=creation, stdout=subprocess.DEVNULL)
                result = json.loads(args.report.read_text(encoding="utf-8"))
                result["synthetic"] = not args.real
                args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                if not passed(result):
                    raise RuntimeError("Acceptance checks failed")
                print(f"PASS: {engine}, 5 distinct videos; report {args.report}")
            finally:
                if server.poll() is None:
                    if os.name == "nt":
                        subprocess.run(["taskkill", "/PID", str(server.pid), "/T", "/F"],
                                       capture_output=True, creationflags=creation, check=False)
                    else:
                        server.terminate()
                    try:
                        server.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        server.kill()
                        server.wait()


if __name__ == "__main__":
    main()
