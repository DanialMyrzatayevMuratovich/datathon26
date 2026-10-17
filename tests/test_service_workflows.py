import json
from types import SimpleNamespace

from PIL import Image

from service.review_packet import build
from vision.experiments import t9_action


def test_review_packet_excludes_data_predictions_and_extra_report_fields(tmp_path):
    (tmp_path / "README.md").write_text("Supported project claim", encoding="utf-8")
    (tmp_path / "data/raw").mkdir(parents=True)
    (tmp_path / "data/raw/private.mov").write_bytes(b"PRIVATE_VIDEO_SENTINEL")
    report = tmp_path / "service/reports/quality-latest.json"
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({"binary": {"accuracy": .85},
                                  "per_video": {"description": "PRIVATE_DESCRIPTION_SENTINEL"},
                                  "new_private_field": "PRIVATE_EXTRA_SENTINEL"}), encoding="utf-8")
    packet = build(tmp_path)
    assert "Supported project claim" in packet and '"accuracy": 0.85' in packet
    assert "PRIVATE_" not in packet


def test_action_parser_rejects_explanations_and_window_order():
    assert t9_action.action_code("A.") == "A"
    assert t9_action.action_code("Actually I cannot tell") == "unknown"
    assert t9_action.action_code("A or B") == "unknown"
    assert t9_action.window_indices(3, 20, 2) == [3, 5, 7]


def test_t9_probe_stops_at_budget_without_claiming_complete(monkeypatch):
    import vision
    class Frames(list):
        timestamps = [i / 2 for i in range(12)]
        def __enter__(self):
            return self
        def __exit__(self, *_):
            pass
    sampled = Frames([(i / 2, Image.new("RGB", (8, 8), (i, 0, 0))) for i in range(12)])
    monkeypatch.delenv("DT_CACHE", raising=False)
    engine = SimpleNamespace(FPS=2, find_scene=lambda _: SimpleNamespace(desk=(0, 0, 8, 8)))
    monkeypatch.setattr(vision, "engine", engine, raising=False)
    monkeypatch.setattr(t9_action, "sample", lambda *a, **kw: sampled)
    monkeypatch.setattr(t9_action.limits, "wait_if_hot", lambda: None)
    monkeypatch.setattr(t9_action.vlm, "ask", lambda *a, **kw: ("A", 0))
    result = t9_action.probe("unused.mov", every=1, max_calls=2, refine=True)
    assert not result["complete"]
    assert result["actual_vlm_calls"] == len(result["windows"]) == 2
