"""One disposable process per video: a hung model can be stopped at the deadline."""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def write_json(path: Path, value: dict) -> None:
    """Replace a complete JSON document; readers never see a half-written result."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    # Rule-tuning cache is never evidence of fresh inference in the live API.
    # It also uses one temp path per video, unsuitable for concurrent jobs.
    os.environ.pop("DT_CACHE", None)
    # launchd restarts the API after a crash, but our independent process groups
    # must not leave old GPU workers alive. A separate process can also interrupt
    # a native model call that holds the Python GIL.
    if os.name != "nt" and os.getenv("SERVICE_PARENT_PID") and os.getenv("SERVICE_JOB_DEADLINE"):
        subprocess.Popen([
            sys.executable, "-m", "service.watchdog", os.environ["SERVICE_PARENT_PID"],
            str(os.getpid()), os.environ["SERVICE_JOB_DEADLINE"],
        ], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # Model weights must be prepared locally before starting the service.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from vision import analyze

    directory = Path(sys.argv[1]).resolve()

    def progress(value: float, stage: str) -> None:
        write_json(directory / "progress.json", {"progress": value, "stage": stage})

    result = analyze(directory / "source.mov", progress)
    write_json(directory / "analysis.json", result.model_dump(mode="json"))


if __name__ == "__main__":
    main()
