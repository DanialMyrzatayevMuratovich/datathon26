"""Bounded local queue, process deadlines and restart-safe job storage."""

import json
import logging
import math
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path

from contract.schemas import JobDetails, VideoAnalysis
from service.config import ROOT, Settings
from service.worker import write_json

LOG = logging.getLogger(__name__)
JOB_ID = re.compile(r"[0-9a-f]{32}\Z")


def stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True, timeout=10, check=False,
        )
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


class JobManager:
    def __init__(self, settings: Settings, command: Callable[[Path], list[str]] | None = None):
        self.settings = settings
        self.command = command or (lambda path: [sys.executable, "-m", "service.worker", str(path)])
        self.lock = threading.RLock()
        self.slots = threading.BoundedSemaphore(settings.capacity)
        self.pending: queue.Queue[str] = queue.Queue()
        self.stopping = threading.Event()
        self.records: dict[str, dict] = {}
        self.threads: list[threading.Thread] = []
        self.lock_file = None
        self.uploads: set[str] = set()
        self.unsaved: set[str] = set()

    def start(self) -> None:
        root = self.settings.jobs_dir
        root.mkdir(parents=True, exist_ok=True)
        # Protect uploads even on a fresh checkout before the lead adds the global ignore rule.
        (root / ".gitignore").write_text("*\n", encoding="utf-8")
        # Only one API process may own this queue. uvicorn must use --workers 1.
        self.lock_file = (root / ".service.lock").open("a+b")
        if self.lock_file.tell() == 0:
            self.lock_file.write(b"0")
            self.lock_file.flush()
        self.lock_file.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.lock_file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.lock_file.close()
            self.lock_file = None
            raise RuntimeError("Another service owns JOBS_DIR; use one uvicorn worker") from exc
        for directory in root.iterdir():
            if not JOB_ID.fullmatch(directory.name) or directory.is_symlink() or not directory.is_dir():
                continue
            try:
                record = json.loads((directory / "result.json").read_text(encoding="utf-8"))
                details = JobDetails.model_validate(record["details"])
                if not all(math.isfinite(float(record[key])) for key in ("created_at", "updated_at")):
                    continue
                for key in ("created_at", "updated_at"):
                    record[key] = float(record[key])
                record["details"] = details.model_dump()
                if details.job_id != directory.name:
                    continue
                self.records[directory.name] = record
                if details.status == "processing":
                    self.fail(directory.name, "Сервис был перезапущен.")
            except (OSError, ValueError, KeyError, TypeError):
                LOG.warning("Ignored incomplete job %s", directory.name)
        self.cleanup()
        for index in range(self.settings.workers):
            thread = threading.Thread(target=self._work, name=f"video-worker-{index}", daemon=True)
            thread.start()
            self.threads.append(thread)
        janitor = threading.Thread(target=self._clean_loop, name="job-cleanup", daemon=True)
        janitor.start()
        self.threads.append(janitor)

    def close(self) -> None:
        self.stopping.set()
        for thread in self.threads:
            thread.join(timeout=20)
        with self.lock:
            for job_id, record in self.records.items():
                if record["details"]["status"] == "processing":
                    self.fail(job_id, "Сервис остановлен.")
        if self.lock_file is not None:
            self.lock_file.close()
            self.lock_file = None

    def reserve(self) -> bool:
        return self.ready and self.slots.acquire(blocking=False)

    @property
    def ready(self) -> bool:
        with self.lock:
            return (
                not self.stopping.is_set() and not self.unsaved
                and len(self.threads) >= self.settings.workers
                and all(thread.is_alive() for thread in self.threads[:self.settings.workers])
            )

    def allocate(self) -> tuple[str, Path]:
        job_id = uuid.uuid4().hex
        directory = self.settings.jobs_dir / job_id
        with self.lock:
            directory.mkdir()
            self.uploads.add(job_id)
        return job_id, directory

    def discard_upload(self, job_id: str) -> None:
        with self.lock:
            try:
                self._remove_directory(self.settings.jobs_dir / job_id)
            finally:
                self.uploads.discard(job_id)
                self.unsaved.discard(job_id)

    def submit(self, job_id: str, filename: str) -> None:
        now = time.time()
        details = JobDetails(job_id=job_id, status="processing", filename=filename, stage="queued")
        with self.lock:
            self.records[job_id] = {"created_at": now, "updated_at": now, "details": details.model_dump()}
            try:
                self.update(job_id)
            except OSError:
                del self.records[job_id]
                self.unsaved.discard(job_id)
                raise
            self.uploads.discard(job_id)
        self.pending.put(job_id)

    def get(self, job_id: str) -> JobDetails | None:
        if not JOB_ID.fullmatch(job_id):
            return None
        with self.lock:
            record = self.records.get(job_id)
            return JobDetails.model_validate(record["details"]) if record else None

    def update(self, job_id: str, **changes) -> None:
        with self.lock:
            record = self.records[job_id]
            record["details"].update(changes)
            record["updated_at"] = time.time()
            try:
                write_json(self.settings.jobs_dir / job_id / "result.json", record)
            except OSError:
                self.unsaved.add(job_id)
                raise
            self.unsaved.discard(job_id)

    def fail(self, job_id: str, error: str) -> None:
        # update() also changes the in-memory record. A full disk must not kill the queue thread.
        try:
            self.update(job_id, status="failed", stage="failed", error=error, analysis=None)
        except OSError:
            LOG.exception("Could not persist failed job %s; uploads paused until storage recovers", job_id)

    def flush_pending(self) -> None:
        with self.lock:
            for job_id in list(self.unsaved):
                try:
                    self.update(job_id)
                except OSError:
                    return

    def _work(self) -> None:
        while not self.stopping.is_set():
            try:
                job_id = self.pending.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._run(job_id)
            except Exception:
                LOG.exception("Job failed: %s", job_id)
                self.fail(job_id, "Ошибка обработки видео; проверьте доступность диска и журнал сервера.")
            finally:
                self.slots.release()
                self.pending.task_done()

    def _run(self, job_id: str) -> None:
        directory = self.settings.jobs_dir / job_id
        with self.lock:
            remaining = self.settings.timeout_sec - (time.time() - self.records[job_id]["created_at"])
        if remaining <= 0:
            self.update(job_id, status="failed", stage="failed", error="Истекло время ожидания в очереди.")
            return
        deadline = time.monotonic() + remaining
        self.update(job_id, stage="decode")
        env = os.environ.copy()
        env.update(VISION_ENGINE=self.settings.engine, PYTHONIOENCODING="utf-8")
        env["MODELS_DIR"] = str(Path(env.get("MODELS_DIR", ROOT.parent / "models")).expanduser().resolve())
        env["SERVICE_PARENT_PID"] = str(os.getpid())
        env["SERVICE_JOB_DEADLINE"] = str(deadline)
        with (directory / "worker.log").open("wb") as log:
            process = subprocess.Popen(
                self.command(directory), cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=os.name != "nt",
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            try:
                while process.poll() is None:
                    if self.stopping.is_set() or time.monotonic() >= deadline:
                        reason = "Сервис остановлен." if self.stopping.is_set() else "Превышен лимит обработки (270 с)."
                        stop_process(process)
                        self.update(job_id, status="failed", stage="failed", error=reason)
                        return
                    self._progress(job_id, directory)
                    self.stopping.wait(0.1)
                if process.returncode != 0:
                    raise RuntimeError(f"Worker exit code {process.returncode}")
                if time.monotonic() >= deadline:
                    self.update(job_id, status="failed", stage="failed", error="Превышен лимит обработки (270 с).")
                    return
                analysis = VideoAnalysis.model_validate_json((directory / "analysis.json").read_text("utf-8"))
                self.update(job_id, status="done", progress=1.0, stage="report", analysis=analysis.model_dump())
                LOG.info("Job done: %s engine=%s", job_id, analysis.engine)
            finally:
                stop_process(process)

    def _progress(self, job_id: str, directory: Path) -> None:
        try:
            value = json.loads((directory / "progress.json").read_text("utf-8"))
            fraction = float(value["progress"])
            if not math.isfinite(fraction):
                return
            current = self.get(job_id)
            fraction = max(current.progress, min(0.99, max(0, fraction)))
            stage = str(value["stage"])[:80]
            if fraction != current.progress or stage != current.stage:
                self.update(job_id, progress=fraction, stage=stage)
        except (OSError, ValueError, KeyError, TypeError):
            pass

    def cleanup(self) -> None:
        cutoff = time.time() - self.settings.retention_sec
        with self.lock:
            for job_id, record in list(self.records.items()):
                if record["details"]["status"] == "processing" or record["updated_at"] >= cutoff:
                    continue
                if self._remove_directory(self.settings.jobs_dir / job_id):
                    del self.records[job_id]
                    self.unsaved.discard(job_id)
            # Uploads interrupted before result.json was written still contain private video.
            for directory in self.settings.jobs_dir.iterdir():
                if (
                    not JOB_ID.fullmatch(directory.name)
                    or directory.name in self.records or directory.name in self.uploads
                ):
                    continue
                if directory.is_symlink() or not directory.is_dir():
                    continue
                newest = max([directory.stat().st_mtime] + [p.stat().st_mtime for p in directory.iterdir()])
                if newest < cutoff:
                    self._remove_directory(directory)

    def _remove_directory(self, directory: Path) -> bool:
        if (
            not JOB_ID.fullmatch(directory.name) or directory.is_symlink()
            or directory.resolve().parent != self.settings.jobs_dir.resolve()
        ):
            return False
        if directory.exists():
            shutil.rmtree(directory)
        return True

    def _clean_loop(self) -> None:
        last_cleanup = time.monotonic()
        while not self.stopping.wait(1):
            try:
                self.flush_pending()
                if time.monotonic() - last_cleanup >= 60:
                    self.cleanup()
                    last_cleanup = time.monotonic()
            except OSError:
                LOG.exception("Could not clean expired jobs")
