"""T8 "counting after the client left": single 3-frame answer vs 2-of-3 vote, original and mirrored copies.
    DT_T8_VOTE=1 uv run python -m vision.experiments.t8_vote
"""

import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("DT_CACHE", str(ROOT / "runs" / "cache"))

from vision import engine  # noqa: E402
from vision.evaluate import video_path  # noqa: E402
from vision.limits import wait_if_hot  # noqa: E402

VIDEOS = [("X-34", "T8"), ("X-38", "нарушение"), ("X-35", "чисто"), ("X-39", "чисто"), ("X-5", "чисто"),
          ("X-19", "чисто")]


def main():
    print("vote:", engine.T8_VOTE, flush=True)
    with tempfile.TemporaryDirectory() as td:
        for vid, lab in VIDEOS:
            src = video_path(vid)
            mir = Path(td) / f"{vid}_mirror.mov"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-threads", "2", "-i", str(src), "-vf", "hflip", "-an",
                            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", str(mir)], check=True)
            for name, p in (("original", src), ("mirror", mir)):
                wait_if_hot()
                a = engine.run(p)
                t8 = [v.timecode_sec for v in a.violations if v.type_id == "T8"]
                allv = [(v.type_id, v.timecode_sec) for v in a.violations]
                print(f"{vid:5} ({lab:9}) {name:8} T8 at {t8} | all {allv}", flush=True)


if __name__ == "__main__":
    main()
