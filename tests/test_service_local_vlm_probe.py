"""Private-image transport is tested only with generated solid-color images."""

import base64
import io
import json
import urllib.error
from types import SimpleNamespace

import pytest
from PIL import Image

from service.local_vlm_probe import MODEL, LocalVLM, NoRedirect, keep_awake


@pytest.mark.parametrize("model", [MODEL, "gemma-4-e4b-it-qat-q4_0"])
def test_local_adapter_preserves_image_order_and_generation_controls(model, tmp_path):
    trace_path = tmp_path / "inputs.jsonl"
    client = LocalVLM(model=model, action_codes=True, trace_path=trace_path)
    captured = []

    def respond(request, timeout):
        captured.append((request, timeout))
        return io.BytesIO(json.dumps({"choices": [{"message": {"content": " A "}}],
                                     "usage": {"completion_tokens": 1}}).encode())

    client.opener = SimpleNamespace(open=respond)
    frames = [Image.new("RGB", (16, 16), color) for color in ("red", "green", "blue")]
    answer, seconds = client.ask(frames, "Choose A/B/C/D", max_tokens=4)
    request, timeout = captured[0]
    assert request.full_url == "http://127.0.0.1:8091/v1/chat/completions"
    assert 0 < timeout <= 60 and seconds >= 0
    payload = json.loads(request.data)
    assert payload["temperature"] == 0 and payload["max_tokens"] == 4
    assert payload["cache_prompt"] is False and payload["chat_template_kwargs"]["enable_thinking"] is False
    assert payload["model"] == model
    assert payload["grammar"] == 'root ::= [ABCD]'
    parts = payload["messages"][0]["content"]
    for channel, part in enumerate(parts[:3]):
        data = part["image_url"]["url"]
        assert data.startswith("data:image/jpeg;base64,")
        pixel = Image.open(io.BytesIO(base64.b64decode(data.split(",")[1]))).getpixel((0, 0))
        assert pixel[channel] == max(pixel)
    assert parts[3] == {"type": "text", "text": "Choose A/B/C/D"}
    assert answer == "A" and client.calls == client.completion_tokens == 1
    assert len(client.inputs[0]["jpeg_sha256"]) == 3
    assert len(set(client.inputs[0]["jpeg_sha256"])) == 3
    saved = trace_path.read_text("utf-8")
    assert json.loads(saved) == client.inputs[0]
    assert "Choose A/B/C/D" not in saved and "base64" not in saved


def test_redirects_are_rejected_before_images_can_leave_loopback():
    request = SimpleNamespace(full_url="http://127.0.0.1:8091/v1/chat/completions")
    with pytest.raises(urllib.error.HTTPError, match="forbidden"):
        NoRedirect().redirect_request(request, None, 307, "redirect", {}, "https://example.com")


def test_deadline_and_model_mismatch_prevent_requests():
    client = LocalVLM()
    client.opener = SimpleNamespace(open=lambda *args, **kwargs: pytest.fail("must not send request"))
    frames = [Image.new("RGB", (1, 1))]
    with pytest.raises(ValueError):
        client.ask(frames, "test", model_name="different")
    client.deadline = 0
    with pytest.raises(TimeoutError):
        client.ask(frames, "test")


def test_server_errors_never_export_body_or_turn_into_negative_prediction():
    client = LocalVLM()

    def fail(*args, **kwargs):
        raise urllib.error.URLError("synthetic private prompt must not be propagated")

    client.opener = SimpleNamespace(open=fail)
    with pytest.raises(RuntimeError) as error:
        client.ask([Image.new("RGB", (1, 1))], "test")
    assert "private prompt" not in str(error.value)
    assert client.calls == 0


def test_malformed_server_response_is_failure_not_no():
    client = LocalVLM()
    client.opener = SimpleNamespace(open=lambda *args, **kwargs: io.BytesIO(b'{"choices":[]}'))
    with pytest.raises(ValueError, match="Invalid local VLM response"):
        client.ask([Image.new("RGB", (1, 1))], "test")


def test_windows_awake_request_restored_after_failure(monkeypatch):
    import ctypes

    import service.local_vlm_probe as probe

    states = []

    def set_state(value):
        states.append(value)
        return 0x80000000

    monkeypatch.setattr(probe.os, "name", "nt")
    monkeypatch.setattr(ctypes, "windll", SimpleNamespace(kernel32=SimpleNamespace(
        SetThreadExecutionState=set_state)), raising=False)
    with pytest.raises(RuntimeError, match="synthetic interruption"), keep_awake():
        raise RuntimeError("synthetic interruption")
    assert states == [0x80000003, 0x80000000]
