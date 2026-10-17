"""Э9: door left open — behaviour, not glass. After entering, does the person stop by the door for a few seconds
(turning back to close it, X-31) or walk straight to the desk (X-32, X-33)? Raw person boxes (no glass-wall filter),
the entering person followed frame to frame; prints her track (x centre, box bottom, height in % of the frame).
    uv run python -m vision.experiments.e9_door_dwell
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("DT_CACHE", str(ROOT / "runs" / "cache"))

from vision import engine, people  # noqa: E402
from vision.devcache import DevCache  # noqa: E402
from vision.evaluate import ground_truth, video_path  # noqa: E402
from vision.frames import sample  # noqa: E402


def track(vid):
    path = video_path(vid)
    with sample(path, fps=engine.FPS) as frames:
        dc = DevCache(path, fps=engine.FPS)
        dets = dc.get("dets", lambda: people.detect(frames.images()))
        dc.save()
        W, H = frames[0][1].width, frames[0][1].height
        ts = list(frames.timestamps)
    raw = [[d for d in ds if d.label == "person" and d.score >= 0.5 and d.height >= 0.12 * H] for ds in dets]
    if any(raw[i] for i in range(min(len(raw), 2))):
        return None
    first = next((i for i, ps in enumerate(raw) if ps), None)
    if first is None:
        return None
    cur, out = max(raw[first], key=lambda p: p.score), []
    for i in range(first, min(len(raw), first + 40)):
        if raw[i]:
            nxt = min(raw[i], key=lambda p: abs(p.cx - cur.cx) + abs(p.bottom - cur.bottom))
            if abs(nxt.cx - cur.cx) + abs(nxt.bottom - cur.bottom) < 0.25 * H:
                cur = nxt
                out.append((ts[i], round(100 * cur.cx / W), round(100 * cur.bottom / H), round(100 * cur.height / H)))
                continue
        out.append((ts[i], None, None, None))
    return out


def main():
    gt = ground_truth()
    for vid in sorted(gt, key=lambda v: int(v.split("-")[1])):
        tr = track(vid)
        if tr is None:
            continue
        lab = ",".join(e["type_id"] for e in gt[vid]["events"]) or "-"
        print(f"{vid} gt={lab}: " + " ".join(f"{t:.1f}:{x},{b},{h}" if x is not None else f"{t:.1f}:-"
                                            for t, x, b, h in tr[:30]))


if __name__ == "__main__":
    main()


def dwell(tr, step=0.5, skip=1.5, near=6):
    """Seconds spent by the door: the door = the point of the path farthest from where the person ends up (her
    place at the desk), after the first `skip` s (still behind the glass); time within `near` % of the frame width."""
    pts = [(t, x, b, h) for t, x, b, h in tr if x is not None]
    if len(pts) < 8:
        return None, None
    t0 = pts[0][0]
    tail = pts[-6:]
    x_end = sorted(x for _t, x, _b, _h in tail)[len(tail) // 2]
    body = [q for q in pts[:-6] if q[0] - t0 >= skip]
    if not body:
        return None, None
    door = max(body, key=lambda q: abs(q[1] - x_end))
    at_door = [q for q in body if abs(q[1] - door[1]) <= near]
    left_door = max(q[0] for q in at_door)
    return len(at_door) * step, left_door + step


def summary():
    gt = ground_truth()
    for vid in sorted(gt, key=lambda v: int(v.split("-")[1])):
        tr = track(vid)
        if tr is None:
            continue
        d, t = dwell(tr)
        lab = ",".join(e["type_id"] for e in gt[vid]["events"]) or "-"
        print(f"{vid} gt={lab}: by the door {d} s, leaves the door at {t} s")


if __name__ == "__main__" and os.getenv("SUMMARY"):
    summary()
