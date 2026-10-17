"""Per-frame trace of one video for rule tuning: where people are relative to the scene objects.

    uv run python -m vision.experiments.trace X-28            # uses runs/cache (computes and stores if missing)
    uv run python -m vision.experiments.trace X-28 --from 4 --to 20

Columns: t | people in room (box bottom/height as % of frame height) | at_work | near_cash | behind | seat | safe.
"""

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("DT_CACHE", str(ROOT / "runs" / "cache"))

from vision import engine, people  # noqa: E402
from vision.devcache import DevCache  # noqa: E402
from vision.evaluate import video_path  # noqa: E402
from vision.frames import sample  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--from", dest="t0", type=float, default=0.0)
    ap.add_argument("--to", dest="t1", type=float, default=1e9)
    a = ap.parse_args()
    path = video_path(a.video)
    with sample(path, fps=engine.FPS) as frames:
        trace(path, frames, a)


def trace(path, frames, a):
    dc = DevCache(path, fps=engine.FPS)
    sc = dc.get("scene", lambda: engine.find_scene(frames))
    dets = dc.get("dets", lambda: people.detect(frames.images()))
    dc.save()
    H = frames[0][1].height
    tr = engine.tracks(frames, dets, sc)
    pct = lambda v: round(100 * v / H)  # noqa: E731
    print(f"{a.video}: H={H} desk={sc.desk} laptop={sc.laptop} safe={sc.safe} door={sc.door}")
    print(f"desk_top={pct(tr.desk_top)}%  angle={'B (camera behind cashier)' if tr.cashier_below else 'A'}")
    for i, t in enumerate(tr.t):
        if not a.t0 <= t <= a.t1:
            continue
        room = " ".join(f"{pct(p.bottom)}/{pct(p.height)}@{round(100 * p.cx / frames[0][1].width)}"
                        for p in tr.persons[i])
        print(f"{t:5.1f} | {room:34} | w{len(tr.at_work[i])} n{len(tr.near_cash[i])} b{len(tr.behind[i])} "
              f"{'S' if tr.at_seat[i] else '.'}{'F' if tr.at_safe[i] else '.'}")
    n, k = len(tr.t), lambda sec: max(1, round(sec * engine.FPS))  # noqa: E731
    runs = [(s, e) for s, e in engine._runs(tr.present, False, 2) if (e - s >= k(engine.LEAVE_MIN_S) or e == n) and s]
    print("leave:", [(tr.t[s], tr.t[engine.leave_start(tr, s, frames[0][1].width)]) for s, _ in runs])
    t7 = engine._runs([len(ps) >= 2 for ps in tr.near_cash], True, k(engine.TWO_AT_DESK_MIN_S))
    print("two on the cashier side:", [(tr.t[s], tr.t[e - 1]) for s, e in t7])
    print("client arrivals:", [tr.t[i] for i in engine.client_arrivals(tr)])


if __name__ == "__main__":
    main()
