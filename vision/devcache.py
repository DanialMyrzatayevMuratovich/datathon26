"""Development cache: detections and VLM answers per video, so rule tuning does not re-run the heavy models.

Off unless DT_CACHE=<dir> is set (vision.evaluate --cache sets runs/cache). The service and the final double run never
use it. VLM answers are keyed by the exact pixels of the crop + the question, so a changed crop or question is asked
again; detections are keyed by video content + stage name.
"""

import hashlib
import importlib.metadata
import json
import os
import pickle
import tempfile
import warnings
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE_VERSION = 2


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def model_identity(name: str) -> dict:
    """Local file identity, not a claim that weight contents were cryptographically verified."""
    directory = Path(os.getenv("MODELS_DIR", ROOT.parent.parent / "models")) / name
    files = []
    if directory.is_dir():
        for path in sorted(directory.rglob("*")):
            if path.is_file() and ".cache" not in path.parts:
                stat = path.stat()
                files.append((str(path.relative_to(directory)), stat.st_size, stat.st_mtime_ns))
    return {"name": name, "files_stat_sha256": _digest(files), "files": len(files)}


@cache
def _backend_versions():
    versions = {}
    for name in ("torch", "transformers", "mlx-vlm", "Pillow"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def fingerprints(fps=2.0) -> dict:
    source = lambda name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()  # noqa: E731
    decoding = {"frames": source("frames.py"), "fps": fps, "max_side": None}
    versions = _backend_versions()
    return {
        "dets": _digest([decoding, source("people.py"), model_identity("rtdetr-v2-r50"), versions]),
        "scene": _digest([decoding, source("scene.py"), model_identity("grounding-dino-tiny"), versions,
                          "first/middle/last;threshold=.25;desk,laptop,safe,door"]),
        "vlm": _digest([source("vlm.py"), model_identity(os.getenv("VLM_MODEL", "qwen3-vl-4b-instruct-mlx-4bit")),
                        versions]),
    }

def _file_key(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


class DevCache:
    def __init__(self, video: str | Path, *, fps=2.0):
        d = os.getenv("DT_CACHE")
        # Legacy pickle files remain untouched; their model/preprocessing identity is unknown.
        self.file = Path(d) / f"{_file_key(Path(video))}.v{CACHE_VERSION}.pkl" if d else None
        self.fingerprints = fingerprints(fps) if d else {}
        self.data: dict = {}
        self.dirty = False
        if self.file and self.file.exists():
            try:
                self.data = pickle.loads(self.file.read_bytes())
                if not isinstance(self.data, dict):
                    raise ValueError("Invalid cache object")
            except (OSError, ValueError, EOFError, pickle.UnpicklingError):
                warnings.warn("Unreadable local development cache; recomputing", stacklevel=2)
                self.data = {}

    def get(self, key, fn):
        if self.file is None:
            return fn()
        stage = "dets" if key == "dets" else "scene"
        key = (self.fingerprints[stage], key)
        if key not in self.data:
            self.data[key] = fn()
            self.dirty = True
        return self.data[key]

    def vlm(self, image, question: str, fn):
        return self.vlm_many([image], question, fn, max_tokens=8)

    def vlm_many(self, images, question: str, fn, *, max_tokens=64):
        if self.file is None:
            return fn()
        pixels = [(im.mode, im.size, hashlib.sha256(im.tobytes()).hexdigest()) for im in images]
        k = ("vlm", self.fingerprints["vlm"], _digest([pixels, question, max_tokens, 0.0]))
        if k not in self.data:
            self.data[k] = fn()
            self.dirty = True
        return self.data[k]

    def save(self) -> None:
        if self.file and self.dirty:
            self.file.parent.mkdir(parents=True, exist_ok=True)
            tmp = None
            try:
                with tempfile.NamedTemporaryFile(dir=self.file.parent, suffix=".tmp", delete=False) as stream:
                    tmp = Path(stream.name)
                    pickle.dump(self.data, stream)
                tmp.replace(self.file)
            finally:
                if tmp is not None:
                    tmp.unlink(missing_ok=True)
            self.dirty = False
