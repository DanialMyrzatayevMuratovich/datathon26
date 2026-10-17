"""Э5: analyst pass (vision/analyst.py) on the 6 missed clips and all 19 clean ones — does it find the misses
without new false alarms? One video at a time; answers cached in runs/cache.
    uv run python -m vision.experiments.e5_analyst                 # default list
    uv run python -m vision.experiments.e5_analyst X-18,X-39 --sheets   # also save the strips to data/processed/e5
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("DT_CACHE", str(ROOT / "runs" / "cache"))

from vision import analyst, engine, people  # noqa: E402
from vision.devcache import DevCache  # noqa: E402
from vision.evaluate import video_path  # noqa: E402
from vision.frames import crop, sample  # noqa: E402
from vision.limits import wait_if_hot  # noqa: E402

MISSED = ["X-2", "X-18", "X-22", "X-32", "X-33", "X-44"]
CLEAN = ["X-3", "X-5", "X-8", "X-10", "X-12", "X-14", "X-16", "X-19", "X-21", "X-23", "X-25", "X-27", "X-29", "X-31",
         "X-35", "X-37", "X-39", "X-43", "X-45"]
OUT = ROOT / "data" / "processed" / "e5"


def spread(a: int, b: int, m: int) -> list[int]:
    if b <= a:
        return [a]
    return sorted({round(a + (b - a) * x / (m - 1)) for x in range(m)})


def moments(tr, n: int):
    k = lambda s: max(1, round(s * engine.FPS))  # noqa: E731
    out = []
    for s, e in engine._runs(tr.client_here, True, k(5.0)):
        out.append(("visit", spread(s, min(n - 1, e + k(2.0)), 8)))
    if not any(tr.persons[i] for i in range(min(n, k(1.0)))):  # empty room at the start: somebody enters
        first = next((i for i in range(n) if tr.persons[i]), None)
        if first is not None:
            out.append(("entry", spread(first, min(n - 1, first + k(10.0)), 8)))
    for s, e in engine._runs(tr.present, False, 2):
        if s and (e - s >= k(engine.LEAVE_MIN_S) or e == n):
            out.append(("leave", [min(n - 1, s + k(2.0))]))
    return out


def run(vid: str, sheets: bool) -> list:
    path = video_path(vid)
    with sample(path, fps=engine.FPS) as frames:
        dc = DevCache(path, fps=engine.FPS)
        sc = dc.get("scene", lambda: engine.find_scene(frames))
        dets = dc.get("dets", lambda: people.detect(frames.images()))
        tr = engine.tracks(frames, dets, sc)
        res = []
        for kind, idx in moments(tr, len(frames)):
            wait_if_hot()
            if kind == "visit":
                im = analyst.strip(frames, idx)
                ans = analyst.parse_visit(analyst.ask([im], analyst.Q_VISIT, dc))
            elif kind == "entry":
                im = analyst.strip(frames, idx)
                ans = analyst.ask([im], analyst.Q_ENTRY, dc, max_tokens=4)
            else:
                if sc.safe is None:
                    continue
                room = frames[idx[0]][1]
                im = crop(room, sc.safe, margin=0.15, out_side=672)
                ans = analyst.ask([im, room], analyst.Q_KEY, dc, max_tokens=4)
            res.append((kind, frames.timestamps[idx[0]], ans))
            if sheets:
                OUT.mkdir(parents=True, exist_ok=True)
                im.save(OUT / f"{vid}_{kind}_{frames.timestamps[idx[0]]:.1f}.jpg", quality=85)
        dc.save()
    return res


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    vids = args[0].split(",") if args else MISSED + CLEAN
    for vid in vids:
        tag = "MISSED" if vid in MISSED else "clean " if vid in CLEAN else ""
        print(f"{vid:5} {tag} {run(vid, '--sheets' in sys.argv)}", flush=True)


if __name__ == "__main__":
    main()
