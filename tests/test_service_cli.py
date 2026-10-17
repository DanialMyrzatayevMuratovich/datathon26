"""A real acceptance run must reject mock output and exercise each requested input."""

import json
import sys
from io import BytesIO
from types import SimpleNamespace

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


def fake_requests(monkeypatch, responses):
    clock = [0.0]
    timeouts = []
    pending = iter(responses)

    def open_request(request, timeout):
        duration, value = next(pending)
        timeouts.append(timeout)
        clock[0] += duration
        stream = BytesIO(json.dumps(value).encode())
        stream.status = 200
        return stream

    monkeypatch.setattr(cli, "time", SimpleNamespace(
        monotonic=lambda: clock[0], sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    ))
    monkeypatch.setattr(cli, "opener", lambda: SimpleNamespace(open=open_request))
    return timeouts


DONE = {"job_id": "example", "status": "done", "verdict": "no_violation", "violations": []}
DETAILS = {"analysis": {"engine": "rules+test", "violations": []}}


def test_slow_upload_allowed_and_public_timing_excludes_details(tmp_path, monkeypatch):
    path = tmp_path / "synthetic.mov"
    path.write_bytes(b"synthetic")
    timeouts = fake_requests(monkeypatch, [(60, {"job_id": "example"}), (10, DONE), (15, DETAILS)])
    row = cli.check_one("http://127.0.0.1:8000", "synthetic-key", path, strict=True)
    assert timeouts[0] > 60
    assert row["status"] == "done" and row["accepted_sec"] == 60
    assert row["elapsed_sec"] == 70 and row["verification_elapsed_sec"] == 85
    assert cli.passed({"count": 1, "auth_checks": [], "expected_engine": "real", "results": [row]})


def test_late_done_response_fails_and_poll_uses_remaining_budget(tmp_path, monkeypatch):
    path = tmp_path / "synthetic.mov"
    path.write_bytes(b"synthetic")
    timeouts = fake_requests(monkeypatch, [(299, {"job_id": "example"}), (2, DONE)])
    row = cli.check_one("http://127.0.0.1:8000", "synthetic-key", path)
    assert timeouts[-1] == 1
    assert row["status"] == "timeout" and row["elapsed_sec"] == 301


def test_late_details_does_not_hide_timely_public_result_but_blocks_verification(tmp_path, monkeypatch):
    path = tmp_path / "synthetic.mov"
    path.write_bytes(b"synthetic")
    fake_requests(monkeypatch, [(1, {"job_id": "example"}), (298, DONE), (2, DETAILS)])
    row = cli.check_one("http://127.0.0.1:8000", "synthetic-key", path)
    assert row["status"] == "verification_failed" and row["public_status"] == "done"
    assert row["elapsed_sec"] == 299
    assert row["verification_error"] == "TimeoutError"
    assert not cli.passed({"count": 1, "auth_checks": [], "results": [row]})
