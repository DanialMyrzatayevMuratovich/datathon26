"""Bounded, local cold/warm experiment; no prediction cache or production changes.

Run only on an idle Apple Silicon host with installed weights:
  uv run python -m service.inference_profile path/to/video.mov
The report contains timings, counts and hashes, never frames or model text.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from contextlib import ExitStack, contextmanager
from functools import wraps
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class Timings:
    """Single-threaded nested wall timers; exclusive times do not overlap."""

    def __init__(self, clock=time.perf_counter):
        self.clock = clock
        self.stack = []
        self.values = defaultdict(lambda: {"calls": 0, "inclusive_sec": 0.0, "exclusive_sec": 0.0})
        self.counts = Counter()

    @contextmanager
    def span(self, label):
        frame = [self.clock(), 0.0]
        self.stack.append(frame)
        try:
            yield
        finally:
            elapsed = self.clock() - frame[0]
            self.stack.pop()
            value = self.values[label]
            value["calls"] += 1
            value["inclusive_sec"] += elapsed
            value["exclusive_sec"] += elapsed - frame[1]
            if self.stack:
                self.stack[-1][1] += elapsed

    def wrap(self, function, label, observe=None):
        @wraps(function)
        def call(*args, **kwargs):
            with self.span(label):
                result = function(*args, **kwargs)
                if observe:
                    observe(result)
                return result
        return call

    def report(self, total):
        accounted = sum(row["exclusive_sec"] for row in self.values.values())
        return {"engine_wall_sec": total, "stages": dict(self.values),
                "unattributed_sec": total - accounted, "counts": dict(self.counts)}


def result_digest(result):
    value = result.model_dump(mode="json")
    value.pop("processing_sec", None)
    # Include evidence and descriptions in the LOCAL comparison, emit only hash.
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def peak_rss_bytes():
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if sys.platform == "darwin" else value * 1024)
    except ImportError:
        return None


@contextmanager
def instrument(timing, engine, frames, people, scene, vlm):
    def sampled(result):
        timing.counts["sampled_frames"] += len(result)

    def answer(result):
        value = result[0]
        timing.counts["yes" if value is True else "no" if value is False else "unknown"] += 1

    def objects(result):
        for key, boxes in result.items():
            timing.counts[f"object_queries:{key}"] += 1
            if not boxes:
                timing.counts[f"empty_object_queries:{key}"] += 1

    hooks = [
        (engine, "info", "metadata_probe", None),
        (engine, "sample", "decode_to_scratch", sampled),
        (frames.SampledFrames, "__getitem__", "lazy_frame_decode", None),
        (people, "_model", "load_rtdetr", None),
        (people, "detect", "rtdetr", None),
        (scene, "_model", "load_dino", None),
        (scene, "detect", "dino", objects),
        (vlm, "load", "load_vlm", None),
        (vlm, "ask", "vlm_question", None),
        (vlm, "yes_no", "yes_no_parse", answer),
    ]
    with ExitStack() as stack:
        for module, name, label, observer in hooks:
            stack.enter_context(patch.object(module, name, timing.wrap(getattr(module, name), label, observer)))
        yield


def profile_pair(video, engine, frames, people, scene, vlm):
    loaders = {"rtdetr": people._model, "dino": scene._model, "vlm": vlm.load}
    initial_cache = {name: loader.cache_info().currsize for name, loader in loaders.items()}
    rows = []
    for phase in ("initial", "reuse"):
        # Never remove thermal protection; stop this bounded experiment if hot.
        if (ROOT / "runs/.hot").exists():
            raise RuntimeError("Thermal flag present; stop and let the host cool")
        before = {name: loader.cache_info().misses for name, loader in loaders.items()}
        timing = Timings()
        started = time.perf_counter()
        with instrument(timing, engine, frames, people, scene, vlm):
            result = engine.run(video)
        total = time.perf_counter() - started
        row = timing.report(total)
        row.update({"phase": phase, "duration_sec": result.duration_sec,
                    "events_count": len(result.violations), "output_sha256": result_digest(result),
                    "loader_cache_misses": {name: loader.cache_info().misses - before[name]
                                            for name, loader in loaders.items()},
                    "process_peak_rss_bytes_cumulative": peak_rss_bytes()})
        rows.append(row)
        del result  # do not retain evidence between runs; keep only model caches
    return {"status": "complete", "initial_model_cache_entries": initial_cache,
            "process_cold_start": all(value == 0 for value in initial_cache.values()),
            "prediction_cache_enabled": False, "runs": rows,
            "outputs_identical": rows[0]["output_sha256"] == rows[1]["output_sha256"],
            "initial_minus_reuse_sec": rows[0]["engine_wall_sec"] - rows[1]["engine_wall_sec"],
            "limitations": ["Two engine calls on one input, not API acceptance or independent quality",
                            "OS file caches not controlled; first inference may include lazy initialization",
                            "Load accessor timings include cache hits; inspect loader_cache_misses",
                            "Inclusive stage times overlap; sum only exclusive times",
                            "Peak RSS is cumulative process high-water, not total GPU memory or swap",
                            "Warm delta includes OS/kernel caches; not pure weight-load savings"]}


def child(video, output):
    # One watchdog for this entire experiment, never one fixed-deadline guardian per run.
    if os.name != "nt" and os.getenv("PROFILE_PARENT_PID") and os.getenv("PROFILE_DEADLINE"):
        subprocess.Popen([sys.executable, "-m", "service.watchdog", os.environ["PROFILE_PARENT_PID"],
                          str(os.getpid()), os.environ["PROFILE_DEADLINE"]],
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.environ.pop("DT_CACHE", None)
    os.environ.pop("DT_BATCH", None)
    os.environ.update(VISION_ENGINE="real", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    started = time.perf_counter()
    from vision import engine, frames, people, scene, vlm
    import_seconds = time.perf_counter() - started
    from service.worker import write_json
    from vision.devcache import model_identity

    if engine.BATCH:
        raise RuntimeError("Unexpected batch mode; start a fresh profiling process")
    sources = [*sorted((ROOT / "vision").rglob("*.py")), Path(__file__), ROOT / "uv.lock",
               ROOT / "contract/schemas.py", ROOT / "data/laws/rules_catalog.json"]
    source_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    with video.open("rb") as stream:
        video_digest = hashlib.file_digest(stream, "sha256").hexdigest()
    report = profile_pair(video, engine, frames, people, scene, vlm)
    report.update({"imports_sec": import_seconds, "video_sha256": video_digest,
                   "source_sha256": source_hashes, "fps": [engine.BASE_FPS, engine.LONG_FPS],
                   "python": platform.python_version(), "platform": platform.platform(),
                   "packages": {name: importlib.metadata.version(name)
                                for name in ("torch", "transformers", "mlx-vlm", "Pillow")},
                   "threads": os.getenv("DT_THREADS", "2"), "dino_device": scene.DEVICE,
                   "model_file_stats_not_content_proof": [model_identity(name) for name in (
                       vlm.DEFAULT_MODEL, "rtdetr-v2-r50", "grounding-dino-tiny")]})
    write_json(output, report)


def supervise(command_factory, output, timeout, environment):
    """Run our own isolated child; clean only the temporary directory we allocated."""
    from service.jobs import stop_process
    from service.worker import write_json

    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, {"status": "running"})  # never leave an old successful report as this run's result
    with tempfile.TemporaryDirectory(prefix="inference-profile-", dir=output.parent) as temporary:
        scratch = Path(temporary).resolve()
        if scratch.parent != output.parent.resolve():
            raise ValueError("Unexpected profiler scratch location")
        result_path = scratch / "result.json"
        deadline = time.monotonic() + timeout
        env = {**environment, "TEMP": str(scratch), "TMP": str(scratch), "TMPDIR": str(scratch),
               "PROFILE_PARENT_PID": str(os.getpid()), "PROFILE_DEADLINE": str(deadline)}
        env.pop("API_KEY", None)
        if env.get("MODELS_DIR"):
            env["MODELS_DIR"] = str(Path(env["MODELS_DIR"]).expanduser().resolve())
        with output.with_suffix(".log").open("wb") as log:
            process = subprocess.Popen(command_factory(result_path), cwd=ROOT, env=env,
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=os.name != "nt",
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            try:
                try:
                    code = process.wait(timeout=max(0, deadline - time.monotonic()))
                    report = {"status": "failed", "exit_code": code}
                    if time.monotonic() >= deadline:
                        report = {"status": "timeout", "exit_code": code}
                    elif code == 0:
                        try:
                            candidate = json.loads(result_path.read_text("utf-8"))
                            if isinstance(candidate, dict) and candidate.get("status") == "complete":
                                report = candidate
                        except (OSError, ValueError):
                            pass  # corrupt/absent output is a failed run, never stale successful metadata
                except subprocess.TimeoutExpired:
                    report = {"status": "timeout"}
            finally:
                stop_process(process)
        write_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/jobs/inference-profile.json")
    parser.add_argument("--timeout-sec", type=float, default=300,
                        help="Budget for BOTH runs and imports, max 600s; not an API SLA test")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    video, output = args.video.resolve(), args.out.resolve()
    if not video.is_file() or video.suffix.lower() not in {".mov", ".mp4"}:
        parser.error("Need a local MOV/MP4 file")
    if not output.is_relative_to((ROOT / "runs/jobs").resolve()):
        parser.error("Write the private experiment report under runs/jobs")
    if not math.isfinite(args.timeout_sec) or not 0 < args.timeout_sec <= 600:
        parser.error("Timeout must be in (0, 600]")
    if args.child:
        child(video, output)
        return
    from dotenv import load_dotenv

    from service.preflight import real_engine_errors
    load_dotenv(ROOT / ".env.local", override=False)
    errors = real_engine_errors()
    if errors:
        parser.error("; ".join(errors))
    if (ROOT / "runs/.hot").exists():
        parser.error("Thermal flag present; do not start another GPU workload")
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "VISION_ENGINE": "real"}
    env.pop("DT_CACHE", None)
    env.pop("DT_BATCH", None)
    report = supervise(lambda result: [sys.executable, "-m", "service.inference_profile", str(video),
                                      "--out", str(result), "--child"], output, args.timeout_sec, env)
    print(f"Profile {report['status']}; metadata: {output}; no bank frames or text exported")
    raise SystemExit(0 if report["status"] == "complete" and report.get("outputs_identical") else 1)


if __name__ == "__main__":
    main()
