"""ForteBank video API. Run one ASGI process; the job manager owns two workers."""

import asyncio
import json
import logging
import math
import secrets
import subprocess
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.formparsers import MultiPartException
from starlette.staticfiles import StaticFiles

from contract.schemas import AnalyzeAccepted, JobDetails, to_public
from service.config import Settings
from service.jobs import JobManager
from service.preflight import real_engine_errors

LOG = logging.getLogger(__name__)
CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data: blob:; media-src 'self' blob:; font-src 'self'; connect-src 'self'; "
    "object-src 'none'; base-uri 'self'; frame-ancestors 'none'"
)


def check_budget(deadline: float | None) -> None:
    if deadline is not None and time.monotonic() >= deadline:
        raise HTTPException(408, "Истекло время приёма видео. Повторите попытку.")


def validate_video(path: Path, *, deadline: float | None = None) -> None:
    """Only parse a local QuickTime/MP4 container; never follow playlists or URLs."""
    try:
        check_budget(deadline)
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-protocol_whitelist", "file,pipe", "-format_whitelist", "mov",
                "-show_entries", "stream=codec_type,width,height:format=duration,format_name",
                "-of", "json", str(path),
            ], capture_output=True,
            timeout=10 if deadline is None else max(0.001, min(10, deadline - time.monotonic())),
            check=True,
        )
        info = json.loads(result.stdout)
        duration = float(info["format"]["duration"])
        video = any(
            stream.get("codec_type") == "video" and stream.get("width", 0) > 0 and stream.get("height", 0) > 0
            for stream in info.get("streams", [])
        )
        if not video or not math.isfinite(duration) or duration <= 0:
            raise ValueError("No video stream")
    except FileNotFoundError as exc:
        raise HTTPException(503, "FFmpeg/ffprobe не установлен на сервере.") from exc
    except (subprocess.SubprocessError, ValueError, KeyError, TypeError) as exc:
        check_budget(deadline)
        raise HTTPException(400, "Файл не является доступным для чтения видео MOV.") from exc


def save_upload(upload: UploadFile, destination: Path, limit: int, deadline: float | None = None) -> None:
    size = 0
    with destination.open("xb") as output:
        while chunk := upload.file.read(1024 * 1024):
            check_budget(deadline)
            size += len(chunk)
            if size > limit:
                raise HTTPException(413, "Видео превышает лимит загрузки.")
            output.write(chunk)
    if size == 0:
        raise HTTPException(400, "Пустой файл.")


class ApiGuard:
    """Authenticate before multipart parsing and bound streaming request bodies."""

    def __init__(self, app, settings: Settings):
        self.app = app
        self.settings = settings

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        started = time.monotonic()
        path = scope["path"]
        if path.startswith("/api/") and path != "/api/v1/health":
            headers = dict(scope["headers"])
            supplied = headers.get(b"x-api-key", b"")
            if not self.settings.api_key or not secrets.compare_digest(supplied, self.settings.api_key.encode()):
                response = JSONResponse({"detail": "Неверный или отсутствующий X-API-Key."}, 401)
                return await response(scope, receive, send)
        original_send = send

        async def private_send(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers += [(b"x-content-type-options", b"nosniff"), (b"referrer-policy", b"no-referrer")]
                if path.startswith("/api/"):
                    headers = [(k, v) for k, v in headers if k.lower() != b"cache-control"]
                    headers.append((b"cache-control", b"no-store"))
                else:
                    headers.append((b"content-security-policy", CSP.encode("ascii")))
                message = {**message, "headers": headers}
            await original_send(message)

        send = private_send
        if path != "/api/v1/analyze" or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        limit = self.settings.max_upload_bytes + 1024 * 1024  # multipart envelope
        try:
            declared = int(dict(scope["headers"]).get(b"content-length", b"0"))
            if declared < 0:
                raise ValueError
        except ValueError:
            return await JSONResponse({"detail": "Некорректный Content-Length."}, 400)(scope, receive, send)
        if declared > limit:
            return await JSONResponse({"detail": "Видео превышает лимит загрузки."}, 413)(scope, receive, send)
        total = 0
        exceeded = False
        timed_out = False
        # One server budget includes upload, validation, queue and inference.
        analysis_deadline = started + self.settings.timeout_sec
        scope.setdefault("state", {})["analysis_deadline"] = analysis_deadline
        deadline = min(analysis_deadline, started + self.settings.upload_timeout_sec)

        async def limited_receive():
            nonlocal total, exceeded, timed_out
            remaining = deadline - time.monotonic()
            try:
                if remaining <= 0:
                    raise TimeoutError
                message = await asyncio.wait_for(receive(), timeout=min(remaining, self.settings.upload_idle_sec))
            except TimeoutError as exc:
                timed_out = True
                raise MultiPartException("Истекло время загрузки видео. Повторите попытку.") from exc
            if message["type"] == "http.disconnect":
                raise MultiPartException("Загрузка видео прервана.")
            if message["type"] == "http.request":
                total += len(message.get("body", b""))
                if total > limit:
                    exceeded = True
                    # Starlette closes temporary upload files on this exception.
                    raise MultiPartException("Видео превышает лимит загрузки.")
            return message

        async def limited_send(message):
            if exceeded and message["type"] == "http.response.start":
                message = {**message, "status": 413}
            elif timed_out and message["type"] == "http.response.start":
                message = {**message, "status": 408}
            await send(message)

        await self.app(scope, limited_receive, limited_send)


class TeamUI(StaticFiles):
    async def check_config(self):
        if Path(self.directory).is_dir():
            await super().check_config()

    async def get_response(self, path, scope):
        if not Path(self.directory).is_dir():
            return JSONResponse({"detail": "Интерфейс ещё не подключён. API работает."}, status_code=503)
        return await super().get_response(path, scope)


def create_app(settings: Settings | None = None, manager: JobManager | None = None, probe=validate_video) -> FastAPI:
    settings = settings or Settings.from_env()
    jobs = manager or JobManager(settings)

    @asynccontextmanager
    async def lifespan(app):
        settings.validate()
        if settings.engine == "real" and manager is None:
            errors = real_engine_errors()
            if errors:
                raise RuntimeError("Real engine preflight failed: " + " ".join(errors))
        jobs.start()
        app.state.jobs = jobs
        try:
            yield
        finally:
            await run_in_threadpool(jobs.close)

    app = FastAPI(title="ForteBank Video API", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(ApiGuard, settings=settings)

    @app.get("/api/v1/health")
    def health():
        return JSONResponse({"ok": jobs.ready, "engine": settings.engine}, status_code=200 if jobs.ready else 503)

    @app.post("/api/v1/analyze", response_model=AnalyzeAccepted)
    async def analyze(request: Request):
        deadline = request.state.analysis_deadline
        if not request.headers.get("content-type", "").lower().startswith("multipart/form-data"):
            raise HTTPException(400, "Нужен multipart/form-data с полем file.")
        if not jobs.ready:
            raise HTTPException(503, "Сервис временно не принимает задачи: проверьте диск и воркеры.")
        if not jobs.reserve():
            raise HTTPException(429, "Очередь заполнена. Повторите загрузку позже.", headers={"Retry-After": "5"})
        directory = None
        submitted = False
        try:
            async with request.form(max_files=1, max_fields=0) as form:
                check_budget(deadline)
                upload = form.get("file")
                if not isinstance(upload, UploadFile) or len(form.multi_items()) != 1:
                    raise HTTPException(400, "Нужно одно видео в поле file.")
                filename = (upload.filename or "").replace("\\", "/").rsplit("/", 1)[-1][:240]
                if Path(filename).suffix.lower() != ".mov":
                    raise HTTPException(400, "Поддерживаются файлы .mov.")
                if upload.size is not None and upload.size > settings.max_upload_bytes:
                    raise HTTPException(413, "Видео превышает лимит загрузки.")
                job_id, directory = jobs.allocate()
                await run_in_threadpool(save_upload, upload, directory / "source.mov",
                                        settings.max_upload_bytes, deadline)
                check_budget(deadline)
                probe_options = {"deadline": deadline} if probe is validate_video else {}
                await run_in_threadpool(probe, directory / "source.mov", **probe_options)
                check_budget(deadline)
                jobs.submit(job_id, filename, deadline=deadline)
                submitted = True
                return AnalyzeAccepted(job_id=job_id)
        except OSError as exc:
            LOG.exception("Could not save upload")
            raise HTTPException(503, "Не удалось сохранить видео. Проверьте свободное место на сервере.") from exc
        finally:
            if not submitted:
                try:
                    if directory is not None:
                        try:
                            jobs.discard_upload(directory.name)
                        except OSError:
                            LOG.exception("Could not remove interrupted upload %s", directory.name)
                finally:
                    jobs.slots.release()

    def get_job(job_id: str) -> JobDetails:
        details = jobs.get(job_id)
        if details is None:
            raise HTTPException(404, "Задача не найдена или срок хранения истёк.")
        return details

    @app.get("/api/v1/jobs/{job_id}")
    def result(job_id: str):
        details = get_job(job_id)
        return to_public(job_id, details.status, details.analysis).model_dump(exclude_unset=True)

    @app.get("/api/v1/jobs/{job_id}/details", response_model=JobDetails)
    def details(job_id: str):
        return get_job(job_id)

    @app.api_route("/api/v1/jobs/{job_id}/video", methods=["GET", "HEAD"])
    def video(job_id: str):
        get_job(job_id)
        path = settings.jobs_dir / job_id / "source.mov"
        if not path.is_file() or path.is_symlink():
            raise HTTPException(404, "Исходное видео не найдено.")
        return FileResponse(path, media_type="video/quicktime", headers={"Cache-Control": "no-store"})

    @app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"])
    def unknown_api(path: str):
        raise HTTPException(404, "Метод API не найден.")

    app.mount("/", TeamUI(directory=str(settings.ui_dir), html=True, check_dir=False), name="ui")
    return app


app = create_app()
