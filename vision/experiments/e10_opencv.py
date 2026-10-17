"""Э10: two pure-OpenCV checks suggested by another assistant (no neural nets):
  door  — optical-flow motion in the door box after the entering person reached the desk (open door = motion?);
  key   — bright elongated contour in the lock part of the safe door, 3x Lanczos zoom, 2 s after the cashier left.
    uv run python -m vision.experiments.e10_opencv
"""

import os
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("DT_CACHE", str(ROOT / "runs" / "cache"))

from vision import engine, people  # noqa: E402
from vision.devcache import DevCache  # noqa: E402
from vision.evaluate import video_path  # noqa: E402
from vision.frames import sample  # noqa: E402


def gray(im, box):
    x0, y0, x1, y1 = (int(v) for v in box)
    return cv2.cvtColor(np.asarray(im.convert("RGB"))[max(0, y0):y1, max(0, x0):x1], cv2.COLOR_RGB2GRAY)


def door(vid):
    with sample(video_path(vid), fps=engine.FPS) as fr:
        dc = DevCache(video_path(vid), fps=engine.FPS)
        sc = dc.get("scene", lambda: engine.find_scene(fr))
        if sc.door is None:
            return "no door box"
        n = len(fr)
        mags = []
        for i in range(min(n - 1, 2 * 6), min(n - 1, 2 * 16)):  # 6–16 s: after entering and sitting down
            a, b = gray(fr[i][1], sc.door), gray(fr[i + 1][1], sc.door)
            flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mags.append(float(cv2.magnitude(flow[..., 0], flow[..., 1]).mean()))
        return round(float(np.median(mags)), 3), sum(m > 2.0 for m in mags)


def key(vid):
    path = video_path(vid)
    with sample(path, fps=engine.FPS) as fr:
        dc = DevCache(path, fps=engine.FPS)
        sc = dc.get("scene", lambda: engine.find_scene(fr))
        dets = dc.get("dets", lambda: people.detect(fr.images()))
        tr = engine.tracks(fr, dets, sc)
        n, k = len(fr), lambda s: max(1, round(s * engine.FPS))  # noqa: E731
        out = []
        for s, e in engine._runs(tr.present, False, 2):
            if not s or not (e - s >= k(engine.LEAVE_MIN_S) or e == n) or sc.safe is None:
                continue
            x0, y0, x1, y1 = sc.safe
            lock = [x0, y0 + 0.25 * (y1 - y0), x0 + 0.45 * (x1 - x0), y0 + 0.7 * (y1 - y0)]
            g = gray(fr[min(n - 1, s + k(2.0))][1], lock)
            g = cv2.resize(g, (g.shape[1] * 3, g.shape[0] * 3), interpolation=cv2.INTER_LANCZOS4)
            thr = max(200, int(np.percentile(g, 99)))
            _, th = cv2.threshold(g, thr - 1, 255, cv2.THRESH_BINARY)
            cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            keys = 0
            for c in cnts:
                if 10 < cv2.contourArea(c) < 500:
                    (w, h) = cv2.minAreaRect(c)[1]
                    if max(w, h) / (min(w, h) + 1e-5) > 2.5:
                        keys += 1
            out.append((fr.timestamps[s], keys))
        return out


def main():
    print("DOOR (median flow, frames > 2.0) after entering:")
    for v, lab in [("X-31", "закрыла"), ("X-32", "НЕ закрыла"), ("X-33", "НЕ закрыла"), ("X-40", "в порядке")]:
        print(f"  {v} {lab}: {door(v)}", flush=True)
    print("KEY (elongated bright contours in the lock area) 2 s after leaving:")
    for v in ["X-22", "X-11", "X-12", "X-13", "X-15", "X-23", "X-24", "X-25", "X-26", "X-27", "X-28", "X-29"]:
        print(f"  {v} {'КЛЮЧ' if v == 'X-22' else '-'}: {key(v)}", flush=True)


if __name__ == "__main__":
    main()
