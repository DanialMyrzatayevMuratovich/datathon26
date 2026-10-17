"""Э3: two targeted VLM checks on the cached layout (runs/cache), one video at a time.

  leave_items — at every leave of the cashier (2 s after the desk got empty): desk tiles, "drink cans, bottles, cups or
                food on the desk?" (personal items left behind, X-24) — should be no on clean leaves;
  after_client — 3 ordered frames after a client leaves (T8): "counting at the machine" vs "putting cash away".
    uv run python -m vision.experiments.e3_probe leave_items
    uv run python -m vision.experiments.e3_probe after_client
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("DT_CACHE", str(ROOT / "runs" / "cache"))

from vision import engine, vlm  # noqa: E402
from vision.devcache import DevCache  # noqa: E402
from vision.evaluate import video_path  # noqa: E402
from vision.frames import crop, sample  # noqa: E402
from vision.limits import wait_if_hot  # noqa: E402

Q_ITEMS2 = ("Are there drink cans, bottles, cups, food or other personal belongings lying on this desk? Answer no for "
            "a laptop, a desk phone, a banknote counting machine, a currency detector and papers.")
Q_ACTION = ("These are three frames in time order from a camera above a bank cash desk, just after the client left. "
            "What is the cashier doing? Answer with one letter: A — feeding or counting banknotes in the counting "
            "machine; B — putting the cash away (into a safe, a drawer or a bag) or tidying up; C — something else.")
LEAVES = ["X-11", "X-12", "X-13", "X-15", "X-22", "X-23", "X-24", "X-25", "X-26", "X-27", "X-28", "X-29"]
AFTER = ["X-34", "X-35", "X-38", "X-39", "X-37", "X-3", "X-5", "X-19"]


class _F:
    width, height = 464, 848


def layout(vid):
    path = video_path(vid)
    frames = sample(path, fps=engine.FPS)
    dc = DevCache(path, fps=engine.FPS)
    sc = dc.get("scene", lambda: engine.find_scene(frames))
    tr = engine.tracks(frames, dc.get("dets", lambda: None), sc)
    return frames, dc, sc, tr


def leave_items():
    for vid in LEAVES:
        wait_if_hot()
        frames, dc, sc, tr = layout(vid)
        n, k = len(tr.t), lambda s: max(1, round(s * engine.FPS))  # noqa: E731
        out = []
        for s, e in engine._runs(tr.present, False, 2):
            if not s or not (e - s >= k(engine.LEAVE_MIN_S) or e == n) or not sc.desk:
                continue
            j = min(n - 1, s + k(2.0))
            ans = any(dc.vlm(t, Q_ITEMS2, lambda t=t: vlm.yes_no([t], Q_ITEMS2)[0])
                      for t in engine._tiles(frames[j][1], sc.desk))
            out.append((tr.t[j], ans))
        dc.save()
        print(vid, out, flush=True)


def after_client():
    for vid in AFTER:
        wait_if_hot()
        frames, dc, sc, tr = layout(vid)
        n, k = len(tr.t), lambda s: max(1, round(s * engine.FPS))  # noqa: E731
        out = []
        for _s, e in engine._runs(tr.client_here, True, k(5.0)):
            if e >= n - 1:
                continue
            idx = [min(n - 1, e + k(x)) for x in (1.0, 2.5, 4.0)]
            zone = engine._grow(sc.desk, 0.15) if sc.desk else [0, 0, frames[0][1].width, frames[0][1].height]
            ims = [crop(frames[i][1], zone, margin=0.05, out_side=448) for i in idx]
            ans = dc.vlm_many(ims, Q_ACTION, lambda ims=ims: vlm.ask(ims, Q_ACTION, max_tokens=4)[0], max_tokens=4)
            out.append((tr.t[e], ans))
        dc.save()
        print(vid, out, flush=True)


if __name__ == "__main__":
    {"leave_items": leave_items, "after_client": after_client}[sys.argv[1]]()
