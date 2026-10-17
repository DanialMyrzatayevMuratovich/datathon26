"""Synthetic checks of profiling bookkeeping, isolation and bounded cleanup."""

import json
import os
import subprocess
import sys
from copy import deepcopy
from functools import cache
from types import SimpleNamespace

import pytest

from service import inference_profile as profile


class Result:
    duration_sec = 2
    violations = []

    def __init__(self, **changes):
        self.value = {"verdict": "no_violation", "violations": [], "processing_sec": 1, **changes}

    def model_dump(self, **kwargs):
        return deepcopy(self.value)


def fake_pipeline():
    frames = SimpleNamespace()
    people = SimpleNamespace(_model=cache(lambda: object()))
    scene = SimpleNamespace(_model=cache(lambda: object()))
    vlm = SimpleNamespace(load=cache(lambda: object()))
    state = {"closed": 0}

    class SampledFrames:
        def __len__(self):
            return 2

        def __getitem__(self, index):
            return index

        def __enter__(self):
            return self

        def __exit__(self, *args):
            state["closed"] += 1

    frames.SampledFrames = SampledFrames
    engine = SimpleNamespace(info=lambda path: None, sample=lambda path: SampledFrames())

    def detect_people(images):
        people._model()
        return [images[0], images[1]]

    def detect_scene(images):
        scene._model()
        return {"desk": [images[0]], "safe": []}

    def ask(*args):
        vlm.load()
        return "unrecognized", 0.1

    def yes_no(*args):
        text, seconds = vlm.ask(*args)
        return None, text, seconds

    def run(path):
        engine.info(path)
        with engine.sample(path) as images:
            people.detect(images)
            scene.detect(images)
            vlm.yes_no(images, "synthetic question")
        return Result()

    people.detect, scene.detect, vlm.ask, vlm.yes_no = detect_people, detect_scene, ask, yes_no
    engine.run = run
    return engine, frames, people, scene, vlm, state


def test_nested_timers_do_not_subtract_model_load_twice():
    now = [0.0]
    timings = profile.Timings(clock=lambda: now[0])
    with timings.span("question"):
        now[0] = 1
        with timings.span("load"):
            now[0] = 4
        now[0] = 6
    report = timings.report(8)
    assert report["stages"]["question"]["inclusive_sec"] == 6
    assert report["stages"]["question"]["exclusive_sec"] == 3
    assert report["stages"]["load"]["exclusive_sec"] == 3
    assert report["unattributed_sec"] == 2
    with pytest.raises(RuntimeError), timings.span("failure"):
        now[0] = 9
        raise RuntimeError("synthetic failure")
    assert timings.stack == []


def test_two_runs_reuse_models_preserve_results_and_restore_hooks(tmp_path, monkeypatch):
    monkeypatch.setattr(profile, "ROOT", tmp_path)
    engine, frames, people, scene, vlm, state = fake_pipeline()
    originals = people._model, vlm.load, vlm.ask, frames.SampledFrames.__getitem__
    report = profile.profile_pair(tmp_path / "synthetic.mov", engine, frames, people, scene, vlm)
    assert report["process_cold_start"] and report["outputs_identical"]
    assert report["runs"][0]["loader_cache_misses"] == {"rtdetr": 1, "dino": 1, "vlm": 1}
    assert report["runs"][1]["loader_cache_misses"] == {"rtdetr": 0, "dino": 0, "vlm": 0}
    for row in report["runs"]:
        assert row["counts"]["unknown"] == 1
        assert row["counts"]["sampled_frames"] == 2
        assert row["counts"]["empty_object_queries:safe"] == 1
        assert row["unattributed_sec"] >= 0
    assert state["closed"] == 2
    assert originals == (people._model, vlm.load, vlm.ask, frames.SampledFrames.__getitem__)
    with pytest.raises(RuntimeError):
        with profile.instrument(profile.Timings(), engine, frames, people, scene, vlm):
            raise RuntimeError("restore on failure")
    assert originals == (people._model, vlm.load, vlm.ask, frames.SampledFrames.__getitem__)
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs/.hot").touch()
    with pytest.raises(RuntimeError, match="Thermal"):
        profile.profile_pair(tmp_path / "synthetic.mov", engine, frames, people, scene, vlm)
    assert state["closed"] == 2


def test_result_hash_ignores_timing_but_includes_prediction_and_evidence():
    assert profile.result_digest(Result()) == profile.result_digest(Result(processing_sec=99))
    base = {"type_id": "T1", "timestamp_sec": 1, "evidence": "synthetic", "description": "example"}
    reference = profile.result_digest(Result(violations=[base]))
    for key, value in [("type_id", "T2"), ("timestamp_sec", 2), ("evidence", "changed"),
                       ("description", "changed")]:
        assert profile.result_digest(Result(violations=[{**base, key: value}])) != reference
    assert profile.result_digest(Result(verdict="violation")) != profile.result_digest(Result())


def test_supervisor_deadline_kills_child_and_cleans_only_its_scratch(tmp_path):
    marker = tmp_path / "preserve.txt"
    marker.write_text("keep", encoding="utf-8")
    output = tmp_path / "profile.json"
    code = "import tempfile, time; tempfile.NamedTemporaryFile(delete=False); time.sleep(30)"
    report = profile.supervise(lambda result: [sys.executable, "-c", code], output, 0.5, os.environ)
    assert report["status"] == "timeout"
    assert json.loads(output.read_text("utf-8"))["status"] == "timeout"
    assert not list(tmp_path.glob("inference-profile-*"))
    assert marker.read_text("utf-8") == "keep"


def test_supervisor_success_never_forwards_api_key(tmp_path):
    code = ("import json, os, pathlib, sys; "
            "assert 'API_KEY' not in os.environ; "
            "assert os.environ['TMPDIR'] == str(pathlib.Path(sys.argv[1]).parent); "
            "assert pathlib.Path(os.environ['MODELS_DIR']).is_absolute(); "
            "pathlib.Path(sys.argv[1]).write_text(json.dumps({'status':'complete'}))")
    report = profile.supervise(lambda result: [sys.executable, "-c", code, str(result)],
                               tmp_path / "profile.json", 10,
                               {**os.environ, "API_KEY": "synthetic", "MODELS_DIR": "synthetic-models"})
    assert report["status"] == "complete"
    assert not list(tmp_path.glob("inference-profile-*"))


def test_corrupt_child_output_replaces_previous_success(tmp_path):
    output = tmp_path / "profile.json"
    output.write_text('{"status":"complete"}', encoding="utf-8")
    code = "import pathlib, sys; pathlib.Path(sys.argv[1]).write_text('{broken')"
    report = profile.supervise(lambda result: [sys.executable, "-c", code, str(result)],
                               output, 10, os.environ)
    assert report["status"] == "failed"
    assert json.loads(output.read_text("utf-8")) == report
    assert not list(tmp_path.glob("inference-profile-*"))


def test_import_does_not_load_models_or_mutate_environment():
    code = ("import os, sys; before=dict(os.environ); import service.inference_profile; "
            "assert before == dict(os.environ); assert 'vision' not in sys.modules; "
            "assert 'torch' not in sys.modules; assert 'mlx_vlm' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], cwd=profile.ROOT,
                   env={**os.environ, "DT_CACHE": "synthetic", "DT_BATCH": "1"},
                   check=True, timeout=10, capture_output=True)
