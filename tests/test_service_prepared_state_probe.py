"""Prepared replay tests contain generated colors only."""

import hashlib

import pytest
from PIL import Image

from service.prepared_state_probe import QUESTIONS, digest, infer, jpeg_digest
from service.worker import write_json


def prepared(directory):
    directory.mkdir()
    windows = []
    for number in range(5):
        window = {"sample_timestamps_sec": [number, number + 1, number + 2], "images": []}
        for index in range(3):
            image = Image.new("RGB", (12, 12), (number * 20, index * 30, 40))
            path = directory / f"{number}-{index}.png"
            image.save(path)
            window["images"].append({"file": path.name, "png_sha256": digest(path),
                                     "jpeg_sha256": jpeg_digest(image)})
        windows.append(window)
    manifest = {"prepared": True, "complete": False, "mode": "key",
                "question_sha256": hashlib.sha256(QUESTIONS["key"].encode()).hexdigest(),
                "videos": [{"id": "synthetic", "windows": windows}]}
    path = directory / "manifest.json"
    write_json(path, manifest)
    (directory / "manifest.sha256").write_text(digest(path), encoding="ascii")


class Client:
    model = "synthetic-test"

    def __init__(self, fail_after=None):
        self.inputs = []
        self.calls = 0
        self.fail_after = fail_after

    def ask(self, images, question, **kwargs):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("private image description")
        self.inputs.append({"jpeg_sha256": [jpeg_digest(image) for image in images],
                            "question_sha256": hashlib.sha256(question.encode()).hexdigest()})
        return "D", 0.1


@pytest.mark.parametrize("damage", ["png", "manifest", "jpeg", "question"])
def test_replay_hash_mismatch_prevents_every_request(tmp_path, damage):
    import json

    directory = tmp_path / "prepared"
    prepared(directory)
    path = directory / "manifest.json"
    if damage == "png":
        (directory / "4-2.png").write_bytes(b"tampered")  # last input checked before first call
    elif damage == "manifest":
        path.write_bytes(path.read_bytes() + b" ")
    else:
        manifest = json.loads(path.read_text("utf-8"))
        if damage == "jpeg":
            manifest["videos"][0]["windows"][4]["images"][2]["jpeg_sha256"] = "wrong"
        else:
            manifest["question_sha256"] = "wrong"
        write_json(path, manifest)
        (directory / "manifest.sha256").write_text(digest(path), encoding="ascii")
    client = Client()
    report = infer(directory, tmp_path / "result.json", client)
    assert report["status"] == "failed" and not report["complete"]
    assert client.calls == 0 and report["actual_vlm_calls"] == 0


def test_failed_replay_preserves_completed_windows_and_sanitizes_exception(tmp_path):
    import json

    directory = tmp_path / "prepared"
    prepared(directory)
    output = tmp_path / "result.json"
    client = Client(fail_after=1)
    report = infer(directory, output, client)
    assert report["status"] == "failed" and report["complete"] is False
    assert report["error_type"] == "RuntimeError" and client.calls == 2
    assert report["actual_vlm_calls"] == len(report["windows"]) == 1
    assert report["windows"][0]["state"] == "D"
    assert json.loads(output.read_text("utf-8")) == report
    assert "private image" not in output.read_text("utf-8")


def test_successful_replay_performs_actual_calls(tmp_path):
    directory = tmp_path / "prepared"
    prepared(directory)
    client = Client()
    report = infer(directory, tmp_path / "result.json", client)
    assert report["status"] == "complete" and report["complete"]
    assert report["actual_vlm_calls"] == client.calls == 5
    assert all(window["state"] == "D" for window in report["windows"])


@pytest.mark.parametrize("after_first", [False, True])
def test_thermal_flag_stops_replay_and_preserves_completed_calls(tmp_path, monkeypatch, after_first):
    import service.prepared_state_probe as module

    monkeypatch.setattr(module, "ROOT", tmp_path)
    (tmp_path / "runs").mkdir()
    flag = tmp_path / "runs/.hot"
    directory = tmp_path / "prepared"
    prepared(directory)

    class HeatingClient(Client):
        def ask(self, *args, **kwargs):
            result = super().ask(*args, **kwargs)
            flag.touch()
            return result

    if not after_first:
        flag.touch()
    client = HeatingClient()
    report = infer(directory, tmp_path / "result.json", client)
    assert report["status"] == "failed" and not report["complete"]
    assert client.calls == report["actual_vlm_calls"] == len(report["windows"]) == int(after_first)
