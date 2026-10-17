"""Acceptance must reject invalid evidence and never forward a key through redirects."""

import base64
import io
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from PIL import Image

from service.acceptance import opener, validate_public
from service.config import Settings
from service.main import create_app


def public_result():
    image = io.BytesIO()
    Image.new("RGB", (8, 8)).save(image, format="JPEG")
    return {"job_id": "example", "status": "done", "verdict": "violation", "violations": [{
        "type": "Synthetic", "timecode_sec": 1.0, "description": "Synthetic check",
        "evidence_frames": [{"timestamp_sec": 1.0, "image_base64": base64.b64encode(image.getvalue()).decode()}],
    }]}


def test_exact_public_contract_and_real_jpeg():
    result = public_result()
    validate_public(result, "example")
    result["violations"][0]["evidence_frames"][0]["image_base64"] = base64.b64encode(b"fake jpeg").decode()
    with pytest.raises((ValueError, OSError)):
        validate_public(result, "example")


@pytest.mark.parametrize("bad_timestamp", ["1.0", True, float("nan"), -1.0])
def test_strict_rejects_invalid_timestamps(bad_timestamp):
    result = public_result()
    result["violations"][0]["timecode_sec"] = bad_timestamp
    with pytest.raises(ValueError):
        validate_public(result, "example")


def test_redirect_does_not_forward_credentials():
    seen = []
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append(self.path)
            self.send_response(302 if self.path == "/start" else 200)
            self.send_header("Location", "/receiver")
            self.end_headers()

        def log_message(self, *args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            request = urllib.request.Request(
                f"http://127.0.0.1:{server.server_port}/start", headers={"X-API-Key": "synthetic-test-key"},
            )
            with pytest.raises(urllib.error.HTTPError) as error:
                opener().open(request, timeout=2)
            assert error.value.code == 302
            error.value.close()
            assert seen == ["/start"]
        finally:
            server.shutdown()
            thread.join(timeout=2)


def test_real_service_refuses_start_when_preflight_fails(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.setattr("service.main.real_engine_errors", lambda: ["Missing local weights"])
    settings = Settings(api_key="synthetic-key", engine="real", jobs_dir=tmp_path / "jobs")
    with pytest.raises(RuntimeError, match="Missing local weights"), TestClient(create_app(settings)):
        pass
    assert not settings.jobs_dir.exists()
