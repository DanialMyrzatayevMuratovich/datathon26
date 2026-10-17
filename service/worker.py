"""One disposable process per video: a hung model can be stopped at the deadline."""

import json
import os
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
