"""Deployment credential boundaries and worker crash containment."""

import os
import plistlib
import subprocess
import sys
import time

import pytest

from service.deploy.render import render


def test_deploy_keeps_key_out_of_plist_and_api_unprivileged(tmp_path):
    output = tmp_path / "runs/jobs/deploy"
    models = tmp_path / "custom models"
    render("video.example.com", tmp_path, models, "operator", tmp_path / "home", output)
    api = plistlib.loads((output / "kz.fortebank.api.plist").read_bytes())
    assert api["UserName"] == "operator"
    assert api["EnvironmentVariables"]["MODELS_DIR"] == str(models)
    assert "API_KEY" not in api["EnvironmentVariables"]
    assert api["ProgramArguments"][0] == "/usr/bin/caffeinate"
    assert "127.0.0.1" in api["ProgramArguments"]
    assert "admin off" in (output / "Caddyfile").read_text()
    with pytest.raises(ValueError):
        render("video.example.com\nmalicious", tmp_path, models, "operator", tmp_path, output)
    with pytest.raises(ValueError):
        render("video.example.com", tmp_path, models, "operator", tmp_path, tmp_path / "tracked")


@pytest.mark.skipif(os.name == "nt", reason="Real deployment uses POSIX process groups on Apple Silicon")
@pytest.mark.parametrize("crash_parent", [False, True])
def test_watchdog_stops_worker_after_deadline_or_parent_crash(crash_parent):
    # A separate sleep process represents the API. Killing it must not leave the
    # inference process alive, even when inference itself never returns.
    parent = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    code = (
        "import os, subprocess, sys, time; "
        "subprocess.Popen([sys.executable, '-m', 'service.watchdog', sys.argv[1], str(os.getpid()), sys.argv[2]]); "
        "time.sleep(60)"
    )
    deadline = time.monotonic() + (30 if crash_parent else 1)
    worker = subprocess.Popen([sys.executable, "-c", code, str(parent.pid), str(deadline)], start_new_session=True)
    try:
        if crash_parent:
            parent.kill()
            parent.wait(timeout=3)
        assert worker.wait(timeout=5) != 0
    finally:
        if worker.poll() is None:
            import signal
            os.killpg(worker.pid, signal.SIGKILL)
        worker.wait(timeout=3)
        parent.kill()
        parent.wait(timeout=3)
