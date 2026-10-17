"""Visual-state orchestration tests use generated colors, never banking videos."""

import hashlib
import json
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from PIL import Image

from service.local_state_probe import QUESTIONS, prepare_frames, probe_frames, state_code, uniform_windows


class FakeClient:
    def __init__(self, answers):
        self.answers = iter(answers)
        self.inputs = []
        self.requests = []

    def ask(self, images, question, *, max_tokens):
        self.requests.append((images, question, max_tokens))
        self.inputs.append({"jpeg_sha256": [hashlib.sha256(im.tobytes()).hexdigest() for im in images],
                            "question_sha256": hashlib.sha256(question.encode()).hexdigest()})
        return next(self.answers), 0.1


def frames(count=21):
    return [(i / 2, Image.new("RGB", (100, 80), (i, i * 2, i * 3))) for i in range(count)]


@pytest.mark.parametrize("count", [9, 10, 21, 240])
def test_uniform_windows_are_distinct_bounded_and_cover_clip(count):
    windows = uniform_windows(count)
    assert len(windows) == len(set(windows)) == 5
    assert windows[0][0] == 0 and windows[-1][-1] == count - 1
    assert all(0 <= a < b < c < count and c - a == 4 for a, b, c in windows)


@pytest.mark.parametrize("count", [0, 1, 4, 5, 8])
def test_short_clip_never_duplicates_windows(count):
    windows = uniform_windows(count)
    assert len(windows) < 5 and len(windows) == len(set(windows))


@pytest.mark.parametrize("mode,attribute,expected_roi", [
    ("door", "door", [0, 0, 100, 80]), ("key", "safe", [5, 0, 95, 75]),
])
def test_probe_uses_uniform_images_factual_prompt_roi_and_input_hashes(mode, attribute, expected_roi):
    client = FakeClient(["A", "B", "C", "D", "A"])
    checks = []
    scene_inputs = []

    def find_scene(value):
        scene_inputs.append(value)
        return SimpleNamespace(**{attribute: [20, 10, 80, 62]})

    sampled = frames()
    report = probe_frames(sampled, mode=mode, client=client, find_scene=find_scene,
                          wait_if_hot=lambda: checks.append(True))
    assert report["complete"] and report["actual_vlm_calls"] == 5
    assert scene_inputs == [sampled] and len(checks) == 11
    assert report["roi"]["box_pixels"] == expected_roi
    assert report["windows"][0]["sample_timestamps_sec"] == [0, 1, 2]
    assert report["windows"][-1]["sample_timestamps_sec"] == [8, 9, 10]
    for number, (images, question, max_tokens) in enumerate(client.requests):
        assert question == QUESTIONS[mode] and max_tokens == 4
        assert len(images) == 3 and all(max(image.size) == 640 for image in images)
        assert len(set(report["windows"][number]["jpeg_sha256"])) == 3
        assert report["windows"][number]["question_sha256"] == hashlib.sha256(question.encode()).hexdigest()
    encoded = json.dumps(report).lower()
    assert all(word not in encoded for word in ("violation", "accuracy", "annotation", "base64", "data:image"))


@pytest.mark.parametrize("box", [None, [20, 10, 20, 50], [0, 0, float("nan"), 50], [1.1, 2.1, 1.9, 2.9]])
def test_missing_or_invalid_roi_is_incomplete_without_vlm_calls(box):
    client = FakeClient([])
    report = probe_frames(frames(), mode="door", client=client,
                          find_scene=lambda _: SimpleNamespace(door=box), wait_if_hot=lambda: None)
    assert not report["complete"] and report["actual_vlm_calls"] == 0
    assert report["reason"] in ("roi_not_found", "invalid_roi")
    assert report["windows"] == [] and client.requests == []


def test_bad_timestamps_stop_before_scene_or_requests():
    sampled = frames()
    sampled[2] = (0, sampled[2][1])
    client = FakeClient([])
    report = probe_frames(sampled, mode="key", client=client,
                          find_scene=lambda _: pytest.fail("invalid temporal input must not reach scene finder"),
                          wait_if_hot=lambda: None)
    assert not report["complete"] and report["reason"] == "invalid_sample_timestamps"
    assert client.requests == []


@pytest.mark.parametrize("text,expected", [
    (" A ", "A"), ("d", "D"), ("", "unknown"), ("A.", "unknown"),
    ("A because secret detail", "unknown"), ("B or D", "unknown"), (None, "unknown"),
])
def test_state_codes_do_not_parse_free_text_or_invent_negative(text, expected):
    assert state_code(text) == expected


def test_invalid_answer_is_unknown_and_not_exported():
    client = FakeClient(["private detail", "B", "D", "D", "D"])
    report = probe_frames(frames(), mode="key", client=client,
                          find_scene=lambda _: SimpleNamespace(safe=[20, 10, 80, 62]),
                          wait_if_hot=lambda: None)
    assert not report["complete"] and report["reason"] == "invalid_model_output"
    assert report["windows"][0]["state"] == "unknown"
    assert "private detail" not in json.dumps(report)


def test_preparation_retains_pil_triples_without_vlm_calls_or_predictions(monkeypatch):
    from service.local_vlm_probe import LocalVLM

    monkeypatch.setattr(LocalVLM, "ask", lambda *args, **kwargs: pytest.fail("preparation must not call VLM"))
    report, triples = prepare_frames(frames(), mode="door",
                                     find_scene=lambda _: SimpleNamespace(door=[20, 10, 80, 62]),
                                     wait_if_hot=lambda: None)
    assert report["prepared"] and not report["complete"] and report["reason"] == "prepared"
    assert report["actual_vlm_calls"] == 0 and report["request_seconds"] == 0
    assert len(triples) == len(report["windows"]) == 5
    assert all(len(images) == 3 and all(isinstance(image, Image.Image) for image in images) for images in triples)
    assert all(set(window) == {"sample_timestamps_sec"} for window in report["windows"])
    assert report["windows"][0]["sample_timestamps_sec"] == [0, 1, 2]
    assert report["windows"][-1]["sample_timestamps_sec"] == [8, 9, 10]


def test_client_failure_preserves_completed_windows_and_stops_without_private_error_text():
    class FailingClient(FakeClient):
        def ask(self, *args, **kwargs):
            if self.requests:
                raise RuntimeError("private prompt and server response")
            return super().ask(*args, **kwargs)

    client = FailingClient(["A"])
    report = probe_frames(frames(), mode="door", client=client,
                          find_scene=lambda _: SimpleNamespace(door=[20, 10, 80, 62]), wait_if_hot=lambda: None)
    assert not report["complete"] and report["actual_vlm_calls"] == 1
    assert report["reason"] == "request_failed" and report["error_type"] == "RuntimeError"
    assert report["windows"][0]["state"] == "A"
    assert len(report["windows"][0]["jpeg_sha256"]) == 3
    assert all("state" not in row for row in report["windows"][1:])
    assert "private" not in json.dumps(report) and len(client.requests) == 1


@pytest.mark.parametrize("error", [ValueError("private decoder detail"), RuntimeError("private detector detail")])
def test_cli_writes_sanitized_pipeline_failure_atomically(monkeypatch, tmp_path, capsys, error):
    import service.local_state_probe as module
    from service.worker import write_json

    video = tmp_path / "synthetic.mov"
    video.write_bytes(b"synthetic file; probe is replaced")
    output = tmp_path / "runs/jobs/result.json"
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "keep_awake", nullcontext)
    monkeypatch.setattr(sys, "argv", ["local_state_probe", "--mode", "door", "--video", str(video),
                                     "--out", str(output)])

    def fail(*args, **kwargs):
        raise error

    writes = []

    def write(path, value):
        writes.append(value["status"])
        write_json(path, value)

    monkeypatch.setattr(module, "probe", fail)
    monkeypatch.setattr(module, "write_json", write)
    with pytest.raises(SystemExit) as stopped:
        module.main()
    report = json.loads(output.read_text("utf-8"))
    assert stopped.value.code == 1 and writes == ["running", "failed"]
    assert report["status"] == "failed" and report["complete"] is False
    assert report["error_type"] == type(error).__name__ and report["actual_vlm_calls"] == 0
    assert "private" not in output.read_text("utf-8") + capsys.readouterr().out
    assert len(report["video_sha256"]) == 64
