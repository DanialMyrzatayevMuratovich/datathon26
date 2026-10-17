"""Э12: key left in the safe lock, measured on pixels (the VLM answers "empty" even on X-22, Э8). The keyhole is a
dark dot on the light safe door; an inserted key adds a light head just left of it (seen by eye: X-22 after the
cashier left; X-24 at the start, X-8 — key in while she works). Locate the dot (darkest blob in the lock area of the
safe box), then compare the brightness just left of it with the door to its right.
    uv run python -m vision.experiments.e12_key_pixels
"""

import os
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("DT_CACHE", str(ROOT / "runs" / "cache"))

from vision import engine  # noqa: E402
from vision.devcache import DevCache  # noqa: E402
from vision.evaluate import ground_truth, video_path  # noqa: E402
from vision.frames import sample  # noqa: E402

LEAVES = ["X-22", "X-11", "X-12", "X-13", "X-15", "X-23", "X-24", "X-25", "X-26", "X-27", "X-28", "X-29"]
KEY_IN = {"X-22": [3.0, 5.0, 15.0, 17.0, 19.0], "X-24": [1.0, 2.0, 3.0], "X-8": [5.0, 20.0, 37.5]}  # by eye


def _nope():
    raise RuntimeError("not cached")


def key_score(im, safe):
    """(score, dot x, dot y in % of the safe box). The keyhole: the darkest small dot in the lock area of the box
    (5-30 % of its width, 38-68 % of its height). An empty keyhole leaves the door edge and the door next to it the
    same as above and below; a key breaks them (light head, grey metal, shadow): score = mean |pixel - the same
    column's median above/below| on the keyhole rows between the edge and the dot."""
    g = np.asarray(im.convert("L"), dtype=np.float32)
    x0, y0, x1, y1 = safe
    w, h = x1 - x0, y1 - y0
    sx0, sx1 = int(x0 + 0.05 * w), int(x0 + 0.30 * w)
    sy0, sy1 = int(y0 + 0.38 * h), int(y0 + 0.68 * h)
    c = cv2.blur(g, (3, 3))
    # a dot, not the door edge: darker than above, below and to the right (a vertical line is as dark above/below)
    d = 3
    win = np.minimum(np.minimum(c[sy0 - d:sy1 - d, sx0:sx1], c[sy0 + d:sy1 + d, sx0:sx1]),
                     c[sy0:sy1, sx0 + d:sx1 + d]) - c[sy0:sy1, sx0:sx1]
    dy, dx = np.unravel_index(int(np.argmax(win)), win.shape)
    cy, cx = sy0 + dy, sx0 + dx
    cols = slice(cx - 6, cx - 1)
    ref = np.median(np.concatenate([g[cy - 8:cy - 2, cols], g[cy + 4:cy + 9, cols]]), axis=0)
    score = np.abs(g[cy - 1:cy + 3, cols] - ref).mean()
    return float(score), round(100 * (cx - x0) / w), round(100 * (cy - y0) / h)


def scores(vid, times=None):
    path = video_path(vid)
    with sample(path, fps=2.0) as frames:
        dc = DevCache(path, fps=2.0)
        sc = dc.get("scene", _nope)
        ts = list(frames.timestamps)
        if times is None:  # after the cashier left: 1, 2 and 3 s into the empty workplace (clips often end soon)
            tr = engine.tracks(frames, dc.get("dets", _nope), sc)
            runs = [(s, e) for s, e in engine._runs(tr.present, False, 2) if s > 0 and (e - s >= 6 or e == len(ts))]
            if not runs:
                return []
            s = runs[0][0]
            times = sorted({ts[min(len(ts) - 1, s + d)] for d in (2, 4, 6)})
        return [(t, *key_score(frames[ts.index(t)][1], sc.safe)) for t in times]


def main():
    gt = ground_truth()
    print("after the cashier left:")
    for vid in LEAVES:
        lab = ",".join(e["type_id"] for e in gt[vid]["events"]) or "clean"
        print(f"  {vid:5} {lab:6} " + "  ".join(f"{t:.1f}s:{s:+.0f} @({x},{y})" for t, s, x, y in scores(vid)),
              flush=True)
    print("key in the lock (seen by eye):")
    for vid, times in KEY_IN.items():
        print(f"  {vid:5}        " + "  ".join(f"{t:.1f}s:{s:+.0f} @({x},{y})" for t, s, x, y in scores(vid, times)),
              flush=True)


if __name__ == "__main__":
    main()
