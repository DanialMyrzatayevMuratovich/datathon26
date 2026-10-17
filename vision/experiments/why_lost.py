"""Why a finding disappears on a mirrored copy: log every VLM answer and the bag-on-desk detector flags, original vs
mirror (no cache).  uv run python -m vision.experiments.why_lost X-40,X-38
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.pop("DT_CACHE", None)

from vision import engine, people  # noqa: E402
from vision.evaluate import video_path  # noqa: E402
from vision.frames import sample  # noqa: E402


class LogChecker(engine.Checker):
    def _ask(self, key, images, q):
        ans = super()._ask(key, images, q)
        self.log.append((key[0], round(self.frames.timestamps[key[1]], 1) if isinstance(key[1], int) else key[1], ans))
        return ans


def run(path):
    with sample(path, fps=engine.FPS) as frames:
        sc = engine.find_scene(frames)
        dets = people.detect(frames.images())
        tr = engine.tracks(frames, dets, sc)
        H = frames[0][1].height
        bags = [round(frames.timestamps[i], 1) for i, ds in enumerate(dets)
                if any(d.label in engine.BAGS and d.score >= 0.5 for d in ds)]
        chk = LogChecker(frames, sc)
        chk.log = []
        ev = engine.detect_events(frames, dets, sc, chk)
        return ([(e.type_id, e.t) for e in ev], bags[:12], len(bags), round(tr.depth / H, 2) if tr.depth else None,
                [x for x in chk.log if x[0] in ("items", "items_left", "cash_hands", "action3", "cash")][:14])


def main():
    for vid in sys.argv[1].split(","):
        src = video_path(vid)
        with tempfile.TemporaryDirectory() as td:
            mir = Path(td) / "m.mov"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vf", "hflip", "-an", "-c:v", "libx264",
                            "-preset", "veryfast", "-crf", "18", str(mir)], check=True)
            for name, p in (("original", src), ("mirror", mir)):
                ev, bags, nbags, depth, log = run(p)
                print(f"{vid} {name}: events={ev} | bag frames={nbags} first={bags} | depth={depth}\n   answers={log}",
                      flush=True)


if __name__ == "__main__":
    main()
