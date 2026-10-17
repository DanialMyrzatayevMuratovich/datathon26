"""Development cache: detections and VLM answers per video, so rule tuning does not re-run the heavy models.

Off unless DT_CACHE=<dir> is set (vision.evaluate --cache sets runs/cache). The service and the final double run never
use it. VLM answers are keyed by the exact pixels of the crop + the question, so a changed crop or question is asked
again; detections are keyed by video content + stage name.
"""

import hashlib
import os
import pickle
from pathlib import Path


def _file_key(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


class DevCache:
    def __init__(self, video: str | Path):
        d = os.getenv("DT_CACHE")
        self.file = Path(d) / f"{_file_key(Path(video))}.pkl" if d else None
        self.data: dict = {}
        self.dirty = False
        if self.file and self.file.exists():
            self.data = pickle.loads(self.file.read_bytes())

    def get(self, key, fn):
        if self.file is None:
            return fn()
        if key not in self.data:
            self.data[key] = fn()
            self.dirty = True
        return self.data[key]

    def vlm(self, image, question: str, fn):
        if self.file is None:
            return fn()
        k = ("vlm", hashlib.sha1(image.tobytes() + question.encode()).hexdigest()[:20])
        return self.get(k, fn)

    def save(self) -> None:
        if self.file and self.dirty:
            self.file.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.file.with_suffix(".tmp")
            tmp.write_bytes(pickle.dumps(self.data))
            tmp.replace(self.file)
            self.dirty = False
