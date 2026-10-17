"""A development score must not silently certify a candidate for production."""

import json

from service.quality_gate import compare, digest

TRUTH = {
    "X-1": {"violation": True, "events": [{"type_id": "T1", "t": 1}]},
    "X-2": {"violation": False, "events": []},
}


def write_run(path, *, missing=False, cache=False, wrong_type=False, false_alarm=False):
    path.mkdir()
    pred = {"X-1": {"verdict": "violation", "violations": [
        {"type_id": "T9" if wrong_type else "T1", "timecode_sec": 1}]}}
    if not missing:
        pred["X-2"] = {"verdict": "violation" if false_alarm else "no_violation",
                       "violations": [{"type_id": "T1", "timecode_sec": 1}] if false_alarm else []}
    file = path / "predictions.json"
    file.write_text(json.dumps(pred), encoding="utf-8")
    manifest = {
        "status": "complete", "engine": "real", "cache_enabled": cache, "subset": missing,
        "expected": list(TRUTH), "requested": list(pred), "completed": list(pred),
        "labels_sha256": "fixed-labels", "predictions_sha256": digest(file),
        "video_sha256": {vid: f"hash-{vid}" for vid in pred}, "source_sha256": {"vision/engine.py": "source"},
        "runtime": {"python": "test"}, "models": [{"name": "model", "files": {"weights": "hash"}}],
    }
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_full_equal_runs_only_allow_independent_evaluation(tmp_path):
    a, b = write_run(tmp_path / "a"), write_run(tmp_path / "b")
    out = compare(a, b, TRUTH, "fixed-labels")
    assert out["decision"] == "eligible_for_independent_evaluation"
    assert out["automatic_promotion"] is False
    assert out["independent_evaluation"] == "not_assessed"


def test_cached_perfect_subset_is_not_a_quality_claim(tmp_path):
    out = compare(write_run(tmp_path / "a"), write_run(tmp_path / "b", missing=True, cache=True),
                  TRUTH, "fixed-labels")
    assert out["decision"] == "insufficient_evidence"
    assert out["candidate"]["binary"]["accuracy"] is None
    assert out["candidate"]["binary"]["accuracy_observed_only"] == 1
    assert out["transitions_on_observed_videos"] == {}
    assert "candidate:cached_or_subset_run" in out["blockers"]


def test_correct_verdict_does_not_hide_wrong_event_type(tmp_path):
    out = compare(write_run(tmp_path / "a"), write_run(tmp_path / "b", wrong_type=True), TRUTH, "fixed-labels")
    assert out["candidate"]["binary"]["accuracy"] == 1
    assert out["decision"] == "regression_requires_review"
    assert out["joint_event_regression"]


def test_false_alarm_and_tampered_prediction_are_reported(tmp_path):
    a, b = write_run(tmp_path / "a"), write_run(tmp_path / "b", false_alarm=True)
    out = compare(a, b, TRUTH, "fixed-labels")
    assert out["regressed_videos"] == ["X-2"]
    assert out["decision"] == "regression_requires_review"
    file = b / "predictions.json"
    file.write_text(file.read_text() + " ", encoding="utf-8")
    out = compare(a, b, TRUTH, "different-labels")
    assert "candidate:prediction_hash_mismatch" in out["blockers"]
    assert "candidate:label_hash_mismatch" in out["blockers"]


def change_event(run, type_id, time):
    file = run / "predictions.json"
    prediction = json.loads(file.read_text())
    prediction["X-1"]["violations"] = [{"type_id": type_id, "timecode_sec": time}]
    file.write_text(json.dumps(prediction), encoding="utf-8")
    manifest_file = run / "manifest.json"
    manifest = json.loads(manifest_file.read_text())
    manifest["predictions_sha256"] = digest(file)
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")


def test_event_gains_cannot_cancel_lost_matches(tmp_path):
    truth = {**TRUTH, "X-1": {"violation": True, "events": [
        {"type_id": "T1", "t": 1}, {"type_id": "T9", "t": 10}]}}
    a, b = write_run(tmp_path / "a"), write_run(tmp_path / "b")
    change_event(b, "T9", 10)
    out = compare(a, b, truth, "fixed-labels")
    assert out["baseline"]["joint_events"] == out["candidate"]["joint_events"]
    assert out["decision"] == "regression_requires_review"
    assert out["event_regressions_on_observed_videos"][0]["lost_truth_indices"] == [0]


def test_one_second_timing_regression_is_visible(tmp_path):
    a, b = write_run(tmp_path / "a"), write_run(tmp_path / "b")
    change_event(b, "T1", 2.5)
    out = compare(a, b, TRUTH, "fixed-labels")
    assert out["baseline"]["joint_events"]["type_and_time_2s"] == out["candidate"]["joint_events"]["type_and_time_2s"]
    assert out["decision"] == "regression_requires_review"
    assert {r["tolerance_sec"] for r in out["event_regressions_on_observed_videos"]} == {1}
