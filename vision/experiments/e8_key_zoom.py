"""Э8: key in the safe lock after the cashier left — double zoom (whole safe + 3x zoom of the door's lock side with
CLAHE contrast) and a forced choice (idea from the Gemini review). 12 videos with a leave; only X-22 has a key.
    uv run python -m vision.experiments.e8_key_zoom
"""

import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("DT_CACHE", str(ROOT / "runs" / "cache"))

from vision import engine, people, vlm  # noqa: E402
from vision.devcache import DevCache  # noqa: E402
from vision.evaluate import video_path  # noqa: E402
from vision.frames import crop, sample  # noqa: E402

Q = ("Image 1: a grey metal safe. Image 2: a 3x zoom of the left part of its door, where the keyhole is. Look at "
     "image 2. Is a key inserted in the keyhole (a small light key head sticking out), or is the keyhole empty "
     "(a small dark hole)? Choose one: KEY or EMPTY.")
VIDEOS = ["X-22", "X-11", "X-12", "X-13", "X-15", "X-23", "X-24", "X-25", "X-26", "X-27", "X-28", "X-29"]


def clahe(im: Image.Image) -> Image.Image:
    lab = cv2.cvtColor(np.asarray(im.convert("RGB")), cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4)).apply(lab[..., 0])
    return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


def main() -> None:
    k = lambda s: max(1, round(s * engine.FPS))  # noqa: E731
    for vid in VIDEOS:
        path = video_path(vid)
        with sample(path, fps=engine.FPS) as frames:
            dc = DevCache(path, fps=engine.FPS)
            sc = dc.get("scene", lambda: engine.find_scene(frames))
            dets = dc.get("dets", lambda: people.detect(frames.images()))
            tr = engine.tracks(frames, dets, sc)
            n, out = len(frames), []
            for s, e in engine._runs(tr.present, False, 2):
                if not s or not (e - s >= k(engine.LEAVE_MIN_S) or e == n) or sc.safe is None:
                    continue
                im = frames[min(n - 1, s + k(2.0))][1]
                x0, y0, x1, y1 = sc.safe
                lock = [x0, y0 + 0.25 * (y1 - y0), x0 + 0.45 * (x1 - x0), y0 + 0.7 * (y1 - y0)]
                imgs = [crop(im, sc.safe, margin=0.05, out_side=448), clahe(crop(im, lock, margin=0.0, out_side=448))]
                ans = dc.get(("e8key", vid, s), lambda imgs=imgs: vlm.ask(imgs, Q, max_tokens=3)[0])
                out.append((frames.timestamps[s], ans.strip()))
            dc.save()
        print(vid, out, flush=True)


if __name__ == "__main__":
    main()
