"""Contract, access control, real subprocess deadlines and restart tests."""

import base64
import json
import shutil
import subprocess
import sys
import time
from contextlib import ExitStack
from dataclasses import replace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from service.config import Settings
from service.jobs import JobManager
from service.main import create_app, validate_video
from service.worker import write_json

KEY = "local-test-key"
AUTH = {"X-API-Key": KEY}
ANALYSIS = {
    "verdict": "violation", "duration_sec": 2, "processing_sec": 0.1, "engine": "test",
    "violations": [{
        "type": "Test event", "timecode_sec": 1.0, "description": "Synthetic test only",
        "evidence_frames": [{"timestamp_sec": 1, "image_base64": base64.b64encode(b"test-jpeg").decode()}],
        "type_id": "T1", "confidence": 0.5, "rules": [], "why": "Internal field",
    }],
}


def worker_command(delay=0.1, exit_code=0):
    code = (
        "import json, pathlib, sys, time; p=pathlib.Path(sys.argv[1]); "
        "(p/'progress.json').write_text(json.dumps({'progress': 0.5, 'stage': 'detect'})); "
        f"time.sleep({delay!r}); "
        f"(p/'analysis.json').write_text({json.dumps(ANALYSIS)!r}); "
        f"sys.exit({exit_code!r})"
    )
    return lambda path: [sys.executable, "-c", code, str(path)]


@pytest.fixture
def service(tmp_path):
    with ExitStack() as stack:
        def factory(**kwargs):
            settings = Settings(
                api_key=KEY, jobs_dir=tmp_path / f"jobs-{len(stack._exit_callbacks)}",
                ui_dir=tmp_path / "ui", timeout_sec=5, max_upload_bytes=1024,
            )
            settings = replace(settings, **kwargs.pop("settings", {}))
            manager = JobManager(settings, command=kwargs.pop("command", worker_command()))
            app = create_app(settings, manager, probe=kwargs.pop("probe", lambda path: None))
            client = stack.enter_context(TestClient(app))
            return client, manager, settings
        yield factory


def upload(client, content=b"synthetic video", filename="test.mov"):
    return client.post("/api/v1/analyze", headers=AUTH, files={"file": (filename, content, "video/quicktime")})


def wait_for_job(client, job_id, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/jobs/{job_id}", headers=AUTH)
        assert response.status_code == 200
        if response.json()["status"] != "processing":
            return response.json()
        time.sleep(0.03)
    pytest.fail("Job remained processing beyond the test deadline")


def test_contract_progress_video_and_private_fields(service):
    client, _, settings = service(command=worker_command(delay=0.6))
    accepted = upload(client, filename="../../unsafe.MOV")
    assert accepted.status_code == 200
    assert set(accepted.json()) == {"job_id"}
    job_id = accepted.json()["job_id"]
    assert client.get(f"/api/v1/jobs/{job_id}", headers=AUTH).json() == {
        "job_id": job_id, "status": "processing",
    }
    result = wait_for_job(client, job_id)
    assert set(result) == {"job_id", "status", "verdict", "violations"}
    assert result["status"] == "done"
    assert set(result["violations"][0]) == {"type", "timecode_sec", "description", "evidence_frames"}
    details = client.get(f"/api/v1/jobs/{job_id}/details", headers=AUTH).json()
    assert details["progress"] == 1
    assert details["filename"] == "unsafe.MOV"
    assert details["analysis"]["violations"][0]["why"] == "Internal field"
    video = client.get(f"/api/v1/jobs/{job_id}/video", headers={**AUTH, "Range": "bytes=0-3"})
    assert video.status_code == 206
    assert video.content == b"synt"
    assert (settings.jobs_dir / job_id / "result.json").is_file()
    assert client.head(f"/api/v1/jobs/{job_id}/video", headers=AUTH).status_code == 200


@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "wrong"}])
def test_key_is_required_before_parsing_and_for_original_video(service, headers):
    client, _, _ = service()
    for path in ["/api/v1/analyze", "/api/v1/jobs/unknown", "/api/v1/jobs/unknown/details",
                 "/api/v1/jobs/unknown/video"]:
        response = client.post(path, headers=headers, content=b"invalid") if path.endswith("analyze") else client.get(
            path, headers=headers,
        )
        assert response.status_code == 401
    assert client.get("/api/v1/health").json() == {"ok": True, "engine": "mock"}


def test_invalid_uploads_release_capacity_and_remove_files(service):
    client, manager, settings = service(settings={"capacity": 2})
    for content, filename, code in [(b"x", "bad.txt", 400), (b"", "empty.mov", 400), (b"x" * 1025, "big.mov", 413)]:
        assert upload(client, content, filename).status_code == code
    assert client.post("/api/v1/analyze", headers=AUTH, json={}).status_code == 400
    assert client.post("/api/v1/analyze", headers=AUTH, files={"wrong": ("x.mov", b"x")}).status_code == 400
    assert not [p for p in settings.jobs_dir.iterdir() if p.is_dir()]
    assert manager.reserve() and manager.reserve()
    manager.slots.release()
    manager.slots.release()


def test_probe_error_and_unknown_jobs(service):
    def reject(path):
        raise HTTPException(400, "Not a video")
    client, _, _ = service(probe=reject)
    assert upload(client).status_code == 400
    for path in ["/api/v1/jobs/not-an-id", "/api/v1/jobs/" + "f" * 32, "/api/v1/unknown"]:
        assert client.get(path, headers=AUTH).status_code == 404


def test_chunked_body_limit_without_content_length(service):
    client, _, _ = service()
    body = b'--test\r\nContent-Disposition: form-data; name="file"; filename="x.mov"\r\n\r\n'
    headers = {**AUTH, "Content-Type": "multipart/form-data; boundary=test"}
    response = client.post("/api/v1/analyze", headers=headers, content=iter([body, b"x" * (2 * 1024 * 1024)]))
    assert response.status_code == 413


def test_queue_is_bounded_and_timeout_kills_worker(service):
    client, manager, _ = service(
        command=worker_command(delay=30), settings={"workers": 1, "capacity": 1, "timeout_sec": 1},
    )
    job_id = upload(client).json()["job_id"]
    full = upload(client)
    assert full.status_code == 429 and full.headers["Retry-After"] == "5"
    result = wait_for_job(client, job_id)
    assert result == {"job_id": job_id, "status": "failed"}
    assert "лимит" in client.get(f"/api/v1/jobs/{job_id}/details", headers=AUTH).json()["error"]
    manager.command = worker_command(delay=0)
    # Completion releases its slot immediately after storing the final status.
    deadline = time.monotonic() + 2
    while (next_response := upload(client)).status_code == 429 and time.monotonic() < deadline:
        time.sleep(0.03)
    assert next_response.status_code == 200
    assert wait_for_job(client, next_response.json()["job_id"])["status"] == "done"


def test_worker_failure_is_not_a_no_violation_result(service):
    client, _, _ = service(command=worker_command(exit_code=3))
    job_id = upload(client).json()["job_id"]
    result = wait_for_job(client, job_id)
    assert result == {"job_id": job_id, "status": "failed"}
    assert client.get(f"/api/v1/jobs/{job_id}/details", headers=AUTH).json()["error"]


def test_second_service_cannot_own_the_same_queue(service):
    _, _, settings = service()
    second = JobManager(settings)
    with pytest.raises(RuntimeError, match="Another service"):
        second.start()


def test_failed_persistence_does_not_leave_a_queued_job(service, monkeypatch):
    client, manager, settings = service()
    def disk_full(*args, **kwargs):
        raise OSError("Disk full")
    monkeypatch.setattr("service.jobs.write_json", disk_full)
    with pytest.raises(OSError, match="Disk full"):
        upload(client)
    assert manager.records == {}
    assert not [p for p in settings.jobs_dir.iterdir() if p.is_dir()]
    assert manager.reserve()
    manager.slots.release()


def test_done_result_survives_restart_and_interrupted_job_fails(tmp_path):
    settings = Settings(api_key=KEY, jobs_dir=tmp_path / "jobs", ui_dir=tmp_path / "ui")
    first = JobManager(settings, command=worker_command())
    with TestClient(create_app(settings, first, probe=lambda path: None)) as client:
        job_id = upload(client).json()["job_id"]
        result = wait_for_job(client, job_id)
    interrupted = "a" * 32
    (settings.jobs_dir / interrupted).mkdir()
    write_json(settings.jobs_dir / interrupted / "result.json", {
        "created_at": time.time(), "updated_at": time.time(),
        "details": {"job_id": interrupted, "status": "processing"},
    })
    with TestClient(create_app(settings)) as client:
        assert client.get(f"/api/v1/jobs/{job_id}", headers=AUTH).json() == result
        assert client.get(f"/api/v1/jobs/{interrupted}", headers=AUTH).json()["status"] == "failed"


def test_retention_and_static_ui(service):
    client, manager, settings = service()
    assert client.get("/").status_code == 503
    settings.ui_dir.mkdir()
    (settings.ui_dir / "index.html").write_text("<p>Team UI</p>")
    assert "Team UI" in client.get("/").text
    job_id = upload(client).json()["job_id"]
    wait_for_job(client, job_id)
    manager.records[job_id]["updated_at"] = 0
    manager.cleanup()
    assert client.get(f"/api/v1/jobs/{job_id}", headers=AUTH).status_code == 404
    assert not (settings.jobs_dir / job_id).exists()


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg is not installed")
def test_ffprobe_accepts_real_mov_and_rejects_fake(tmp_path):
    path = tmp_path / "sample.mov"
    subprocess.run([
        "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=size=64x64:rate=2", "-t", "1",
        "-c:v", "mpeg4", "-y", str(path),
    ], check=True, timeout=30)
    validate_video(path)
    path.write_bytes(b"not a real movie")
    with pytest.raises(HTTPException) as error:
        validate_video(path)
    assert error.value.status_code == 400
