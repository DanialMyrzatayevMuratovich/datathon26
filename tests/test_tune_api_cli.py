import tomllib

from fastapi.testclient import TestClient

from app import demo, main
from app.agent.orchestrator import run_auto
from app.agent.tune import run_tune
from app.cli import main as cli


def test_tune_writes_params_only_when_better(root, fast_spec):
    paths = demo.write(root / "data", "binary", n=900)
    spec = fast_spec.__class__(train=str(paths["train"]), test=str(paths["test"]), folds=3)
    r = run_tune(spec, "hgb", trials=3, minutes=1, root=root, echo=False)
    params = root / "params.toml"
    assert r["trials"] >= 1
    if r["improved"]:
        assert tomllib.loads(params.read_text())["hgb"] == r["params"]
    else:
        assert not params.exists()


def test_api_serves_runs_and_predicts(root, bank, fast_spec, monkeypatch):
    train, test = bank
    s = run_auto(fast_spec, train_df=train, test_df=test, root=root, author="t", echo=False)
    monkeypatch.setattr(main, "RUNS_DIR", root / "runs")
    monkeypatch.setattr(main, "OUTPUTS", root / "outputs")
    monkeypatch.setattr(main, "MODELS_DIR", root / "models")
    main._BUNDLES.clear()
    client = TestClient(main.app)
    assert client.get("/api/health").json()["best"]["run_id"] == s.run_id
    assert client.get("/api/runs").json()[0]["run_id"] == s.run_id
    detail = client.get(f"/api/runs/{s.run_id}").json()
    assert detail["steps"] and "importance.png" in detail["figures"]
    assert client.get(f"/api/runs/{s.run_id}/files/importance.png").status_code == 200
    assert client.get("/api/runs/..%2F..%2Fetc").status_code in {400, 404}
    records = test.head(3).to_dict(orient="records")
    preds = client.post("/api/predict", json={"records": records}).json()["predictions"]
    assert len(preds) == 3 and all(0 <= p <= 1 for p in preds)
    assert client.get("/").status_code == 200


def test_cli_profile(tmp_path, capsys):
    path = tmp_path / "x.csv"
    demo.clients(30).to_csv(path, index=False)
    assert cli(["profile", str(path), "--target", "default"]) == 0
    out = capsys.readouterr().out
    assert "client_id" in out and "target" in out
