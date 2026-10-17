"""Regression checks use synthetic pixels and fake model answers, never bank video."""

import io
import json
import os
import shutil
import subprocess

import pytest
from PIL import Image

from vision import devcache, evaluate, frames, limits


def test_lossless_sample_is_lazy_and_matches_ffmpeg(tmp_path):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg missing")
    video = tmp_path / "synthetic.mov"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=96x64:rate=30",
                    "-t", "4", "-c:v", "mpeg4", str(video)], check=True)
    reference = subprocess.check_output([
        "ffmpeg", "-v", "error", "-threads", limits.FFMPEG_THREADS, "-i", str(video), "-vf", "fps=2.0",
        "-f", "image2pipe", "-c:v", "png", "pipe:1"])
    chunks = reference.split(b"\x89PNG\r\n\x1a\n")[1:]
    with frames.sample(video) as sampled:
        assert len(sampled) == len(chunks) == 8
        assert not sampled._decoded
        assert sampled.timestamps == [i / 2 for i in range(8)]
        for i, chunk in enumerate(chunks):
            expected = Image.open(io.BytesIO(b"\x89PNG\r\n\x1a\n" + chunk)).convert("RGB")
            assert sampled[i][1].tobytes() == expected.tobytes()
            assert len(sampled._decoded) <= 4
        assert sampled[-1][0] == 3.5
        assert len(sampled.images()[2:5]) == 3
        assert sampled[1:3][0][0] == .5
        with pytest.raises(IndexError):
            sampled[-9]
    assert sampled._file.closed


def test_bad_sampling_and_truncated_stream():
    for fps in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            frames.sample("not-read.mov", fps=fps)
    with pytest.raises(ValueError, match="Truncated"):
        frames._read_exact(io.BytesIO(b"abc"), 5)


def test_bad_detector_boxes_are_rejected_and_crops_clipped():
    for box in ((float("nan"), 0, 4, 4), (2, 2, 1, 1), (0, 0, .1, .1), (20, 0, 30, 8)):
        assert frames.clipped_box(box, 10, 10) is None
    assert frames.clipped_box((-3, -3, 20, 20), 10, 10) == (0, 0, 10, 10)
    image = Image.new("RGB", (10, 10))
    assert frames.crop(image, (-2, -2, 12, 12)).size == (448, 448)
    with pytest.raises(ValueError, match="valid pixels"):
        frames.crop(image, (20, 0, 30, 8))


def test_cache_model_prompt_pixels_order_and_fps(tmp_path, monkeypatch):
    monkeypatch.setenv("DT_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("MODELS_DIR", str(tmp_path / "models"))
    video = tmp_path / "video.mov"
    video.write_bytes(b"synthetic video identity")
    first = devcache.DevCache(video)
    red, blue = Image.new("RGB", (2, 2), "red"), Image.new("RGB", (2, 2), "blue")
    assert first.get("dets", lambda: ["original"]) == ["original"]
    assert first.vlm_many([red, blue], "action?", lambda: "A", max_tokens=4) == "A"
    first.save()
    same = devcache.DevCache(video)
    assert same.get("dets", lambda: pytest.fail("cache miss")) == ["original"]
    assert same.vlm_many([red, blue], "action?", lambda: pytest.fail("cache miss"), max_tokens=4) == "A"
    assert same.vlm_many([blue, red], "action?", lambda: "B", max_tokens=4) == "B"
    assert same.vlm_many([red, blue], "new question", lambda: "C", max_tokens=4) == "C"
    assert same.vlm_many([red, blue], "action?", lambda: "longer", max_tokens=8) == "longer"
    assert devcache.DevCache(video, fps=1).get("dets", lambda: ["new fps"]) == ["new fps"]
    monkeypatch.setenv("VLM_MODEL", "another-model")
    changed = devcache.DevCache(video)
    assert changed.vlm_many([red, blue], "action?", lambda: "new model", max_tokens=4) == "new model"
    assert changed.get("dets", lambda: pytest.fail("VLM change must preserve detector cache")) == ["original"]
    assert not list((tmp_path / "cache").glob("*.tmp"))


def test_stale_heat_marker_fails_without_resuming(tmp_path, monkeypatch):
    hot = tmp_path / ".hot"
    hot.write_text("123")
    os.utime(hot, (1, 1))
    monkeypatch.setattr(limits, "HOT", hot)
    with pytest.raises(RuntimeError, match="heartbeat"):
        limits.wait_if_hot(log=lambda *_: None)
    assert hot.exists()


def test_live_heat_wait_is_bounded_and_clear_resumes(tmp_path, monkeypatch):
    hot = tmp_path / ".hot"
    hot.write_text("123")
    monkeypatch.setattr(limits, "HOT", hot)
    with pytest.raises(TimeoutError):
        limits.wait_if_hot(log=lambda *_: None, max_wait_s=0)
    monkeypatch.setattr(limits.time, "sleep", lambda _: hot.unlink())
    limits.wait_if_hot(log=lambda *_: None)


def test_evaluation_forces_real_and_explicit_cache(monkeypatch):
    monkeypatch.setenv("VISION_ENGINE", "mock")
    monkeypatch.setenv("DT_CACHE", "inherited")
    evaluate.configure(False)
    assert os.environ["VISION_ENGINE"] == "real" and "DT_CACHE" not in os.environ
    evaluate.configure(True)
    assert os.environ["DT_CACHE"] == str(evaluate.ROOT / "runs/cache")


def test_evaluation_joint_hits_and_incomplete_coverage():
    truth = {"X-1": {"violation": True, "events": [{"type_id": "T1", "t": 10}]},
             "X-2": {"violation": False, "events": []}}
    pred = {"X-1": {"verdict": "violation", "processing_sec": 1, "violations": [
        {"type_id": "T1", "timecode_sec": 30}, {"type_id": "T2", "timecode_sec": 10}]}}
    result = evaluate.metrics(truth, pred)
    assert result["type_hit"] == result["time_hit_2s"] == "1/1"
    assert result["joint_events"]["type_and_time_2s"]["tp"] == 0
    assert result["accuracy"] is None
    assert result["coverage"]["missing"] == ["X-2"]


def test_final_compare_rejects_legacy_cached_partial_and_changed_predictions(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "RUNS", tmp_path)
    assert evaluate.compare("a", "b") == 2
    manifest = {"status": "complete", "engine": "real", "cache_enabled": False, "subset": False,
                "expected": ["X-1"], "completed": ["X-1"], "labels_sha256": "labels", "source_sha256": {},
                "models": [], "video_sha256": {"X-1": "video"}}
    for name in ("a", "b"):
        directory = tmp_path / name
        directory.mkdir()
        pred = directory / "predictions.json"
        pred.write_text(json.dumps({"X-1": {"verdict": "no_violation", "violations": []}}))
        manifest["predictions_sha256"] = evaluate.sha256(pred)
        (directory / "manifest.json").write_text(json.dumps(manifest))
    assert evaluate.compare("a", "b") == 0
    path = tmp_path / "b/manifest.json"
    for key in ("cache_enabled", "subset"):
        path.write_text(json.dumps({**manifest, key: True}))
        assert evaluate.compare("a", "b") == 2
    path.write_text(json.dumps(manifest))
    (tmp_path / "b/predictions.json").write_text("{}")
    assert evaluate.compare("a", "b") == 2
