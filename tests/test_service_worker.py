"""Live analysis must not inherit the developer's cached detections or VLM answers."""

import json
import os
import sys
from types import SimpleNamespace

from service.worker import main


def test_live_worker_disables_tuning_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("DT_CACHE", str(tmp_path / "dev-cache"))
    monkeypatch.setenv("DT_BATCH", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")
    monkeypatch.delenv("SERVICE_PARENT_PID", raising=False)
    monkeypatch.delenv("SERVICE_JOB_DEADLINE", raising=False)
    monkeypatch.setattr(sys, "argv", ["service.worker", str(tmp_path)])
    def analyze(path, progress):
        assert path == tmp_path / "source.mov"
        assert "DT_CACHE" not in os.environ
        assert "DT_BATCH" not in os.environ
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
        progress(1, "report")
        return SimpleNamespace(model_dump=lambda **kwargs: {"engine": "fresh-test"})
    monkeypatch.setitem(sys.modules, "vision", SimpleNamespace(analyze=analyze))
    main()
    assert json.loads((tmp_path / "analysis.json").read_text())["engine"] == "fresh-test"
