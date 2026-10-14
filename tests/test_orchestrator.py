import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from app import bundle, demo
from app.agent.log import RunLog
from app.agent.orchestrator import TaskError, run_auto
from app.models import Adapter


def test_binary_end_to_end(root, bank, fast_spec):
    train, test = bank
    s = run_auto(fast_spec, train_df=train, test_df=test, root=root, author="t", echo=False)
    assert s.task == "binary" and s.metric == "roc_auc" and s.cv == "stratified"
    assert s.best_model in {"linear", "hgb"} and s.submission_score > 0.6
    run_dir = root / "runs" / s.run_id
    for name in ("report.md", "metrics.json", "agent_log.jsonl", "submission.csv", "importance.png"):
        assert (run_dir / name).exists(), name
    sub = pd.read_csv(run_dir / "submission.csv")
    assert list(sub.columns) == ["client_id", "default"] and len(sub) == len(test)
    assert sub["default"].between(0, 1).all()
    tools = [st.tool for st in RunLog.read(run_dir)]
    assert tools[:5] == ["plan", "load", "validate", "cv", "features"]
    assert tools[-1] == "report"
    assert json.loads((root / "models" / "best.json").read_text())["run_id"] == s.run_id
    assert s.run_id in (root / "docs" / "EXPERIMENTS.md").read_text()


def test_saved_bundle_predicts_like_the_submission(root, bank, fast_spec):
    train, test = bank
    s = run_auto(fast_spec, train_df=train, test_df=test, root=root, author="t", echo=False)
    b = bundle.load(models_dir=root / "models")
    sub = pd.read_csv(root / "runs" / s.run_id / "submission.csv")
    assert np.allclose(b.predict(test), sub["default"], atol=1e-6)


def test_broken_model_does_not_stop_the_run(root, bank, fast_spec, monkeypatch):
    train, test = bank
    original = Adapter.fit

    def flaky(self, *args, **kwargs):
        if self.name == "hgb":
            raise RuntimeError("simulated crash")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Adapter, "fit", flaky)
    s = run_auto(fast_spec, train_df=train, test_df=test, root=root, author="t", echo=False)
    status = {m.model: m.status for m in s.leaderboard}
    assert status["hgb"] == "fail" and s.best_model == "linear"


def test_regression_and_labels(root, regression_frame, fast_spec):
    train, test = regression_frame.iloc[:600], regression_frame.iloc[600:].drop(columns=["y"])
    s = run_auto(fast_spec, train_df=train, test_df=test, root=root, author="t", echo=False)
    assert s.task == "regression" and s.metric == "rmse" and s.cv == "kfold"
    assert s.submission_score < 1.0


def test_multiclass_labels_in_submission(root, fast_spec):
    df = demo.clients(1500, seed=5).drop(columns=["default"])
    train, test = df.iloc[:1200], df.iloc[1200:].drop(columns=["segment"])
    s = run_auto(replace(fast_spec, target="segment"), train_df=train, test_df=test, root=root,
                 author="t", echo=False)
    assert s.task == "multiclass" and s.metric == "f1_macro"
    sub = pd.read_csv(root / s.submission)
    assert set(sub["segment"]) <= {"premium", "risk", "mass"}


def test_time_split_never_validates_on_the_past(root, bank, fast_spec):
    train, test = bank
    s = run_auto(replace(fast_spec, time="open_date"), train_df=train, test_df=test, root=root,
                 author="t", echo=False)
    assert s.cv == "time"


def test_missing_target_is_a_clear_error(root, bank, fast_spec):
    train, _ = bank
    with pytest.raises(TaskError, match="target"):
        run_auto(fast_spec, train_df=train, test_df=None, root=root, author="t", echo=False)


def test_too_good_score_is_flagged(root, bank, fast_spec):
    train, test = bank
    train = train.assign(leak=train["default"] * 10 + np.random.default_rng(0).normal(size=len(train)))
    test = test.assign(leak=0.0)
    s = run_auto(fast_spec, train_df=train, test_df=test, root=root, author="t", echo=False)
    assert any("утечку" in w for w in s.warnings)
