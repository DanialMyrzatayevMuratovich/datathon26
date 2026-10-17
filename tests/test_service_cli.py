"""A real acceptance run must reject mock output and exercise each requested input."""

import json
import sys

import pytest

from service import cli


def test_real_check_rejects_mock_before_uploading(tmp_path, monkeypatch):
    monkeypatch.setenv("API_KEY", "synthetic-test-key")
    monkeypatch.setattr(sys, "argv", ["cli", "check", str(tmp_path / "unused.mov"), "--expect-engine", "real"])
    monkeypatch.setattr(cli, "health_engine", lambda url: "mock")
    calls = []
    monkeypatch.setattr(cli, "check_one", lambda *args: calls.append(args))
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    assert not calls


def test_mixed_files_are_each_submitted_once(tmp_path, monkeypatch, capsys):
    paths = [tmp_path / name for name in ("clean.mov", "event.mov", "long.mov")]
    for path in paths:
        path.write_bytes(b"synthetic")
    monkeypatch.setenv("API_KEY", "synthetic-test-key")
    monkeypatch.setattr(sys, "argv", ["cli", "check", *map(str, paths), "--count", "3", "--expect-engine", "real"])
    monkeypatch.setattr(cli, "health_engine", lambda url: "real")
    seen = []
    def check(url, key, path, strict):
        seen.append(path)
        return {"status": "done", "elapsed_sec": 1.0, "engine": "rules+test"}
    monkeypatch.setattr(cli, "check_one", check)
    cli.main()
    report = json.loads(capsys.readouterr().out)
    assert sorted(seen) == sorted(paths)
    assert [row["input_name"] for row in report["results"]] == [p.name for p in paths]
    assert cli.passed(report)


@pytest.mark.parametrize("row", [
    {"status": "done", "elapsed_sec": 1, "engine": "mock"},
    {"status": "done", "elapsed_sec": 1, "engine": None},
    {"status": "done", "elapsed_sec": 300, "engine": "rules+test"},
    {"status": "request_failed"},
])
def test_real_acceptance_rejects_wrong_engine_slow_or_failed_result(row):
    assert not cli.passed({"count": 1, "auth_checks": [], "expected_engine": "real", "results": [row]})
