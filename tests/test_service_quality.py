"""Diagnostic scores cannot be improved by duplicate events, missing clips or crossed matches."""

import pytest

from service.quality import audit, labels, match_events


def prediction(*events):
    return {"verdict": "violation" if events else "no_violation", "violations": [
        {"type_id": kind, "timecode_sec": time} for kind, time in events
    ]}


def test_wrong_type_at_right_time_does_not_count_as_joint_hit():
    truth = {"X-1": {"violation": True, "events": [{"type_id": "T1", "t": 10.0}]}}
    result = audit(truth, {"X-1": prediction(("T1", 30), ("T2", 10))})
    assert result["binary"]["accuracy"] == 1
    assert result["joint_events"]["type_and_time_2s"] == {
        "tp": 0, "fp": 2, "fn": 1, "precision": 0, "recall": 0, "f1": 0,
    }


def test_one_prediction_cannot_explain_two_events_and_duplicates_count_as_fp():
    truth = [{"type_id": "T1", "t": 1}, {"type_id": "T1", "t": 2}]
    assert len(match_events(truth, prediction(("T1", 1.5))["violations"], 2)) == 1
    result = audit({"X-1": {"violation": True, "events": truth[:1]}},
                   {"X-1": prediction(("T1", 1), ("T1", 1))})
    assert result["joint_events"]["type_and_time_1s"]["fp"] == 1


def test_matching_reassigns_greedy_choice_to_find_both_events():
    truth = [{"type_id": "T1", "t": 2}, {"type_id": "T1", "t": 0}]
    assert match_events(truth, prediction(("T1", 1), ("T1", 3))["violations"], 1) == [(0, 1), (1, 0)]


def test_missing_video_cannot_produce_perfect_full_score():
    truth = {"X-1": {"violation": False, "events": []},
             "X-2": {"violation": True, "events": [{"type_id": "T9", "t": 3}]}}
    result = audit(truth, {"X-1": prediction()})
    assert result["coverage"]["missing"] == ["X-2"]
    assert result["binary"]["accuracy"] is None
    assert result["joint_events"]["type_and_time_2s"]["fn"] == 1


def test_duplicate_video_alias_and_inconsistent_verdict_are_rejected():
    truth = {"X-4": {"violation": False, "events": []}}
    with pytest.raises(ValueError, match="Duplicate"):
        audit(truth, {"X-4": prediction(), "Х-47": prediction()})
    with pytest.raises(ValueError, match="disagrees"):
        audit(truth, {"X-4": {"verdict": "violation", "violations": []}})


@pytest.mark.parametrize("value", [float("nan"), -1, True, "1"])
def test_bad_timestamps_are_rejected(value):
    with pytest.raises(ValueError):
        audit({"X-1": {"violation": False, "events": []}}, {"X-1": prediction(("T1", value))})


def test_labels_merge_known_duplicate_clips_without_losing_second_event(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text("video,label,type_id,timecode_sec,confidence\n"
                    "X-4,violation,T9,1,high\nХ-47,violation,T9,5,medium\n", encoding="utf-8")
    result = labels(path)
    assert list(result) == ["X-4"]
    assert [e["t"] for e in result["X-4"]["events"]] == [1, 5]
