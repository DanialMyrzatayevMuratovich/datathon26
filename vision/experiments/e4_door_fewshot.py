"""Э4: is the cash-room door left open? Few-shot: reference crops OPEN / CLOSED from OTHER training videos + the query.

Honest check (leave-one-video-out): a video is never its own reference. Query moments: the cashier arrives at an empty
workplace (entered the room) -> 6 s and 8 s later (time to close the door, X-31 closes it at 8-11 s).
    uv run python -m vision.experiments.e4_door_fewshot
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("DT_CACHE", str(ROOT / "runs" / "cache"))

from vision import engine, people, vlm  # noqa: E402
from vision.devcache import DevCache  # noqa: E402
from vision.evaluate import video_path  # noqa: E402
from vision.frames import crop, frame_at, sample  # noqa: E402
from vision.limits import wait_if_hot  # noqa: E402

Q = ("Image 1 shows an office glass door that is OPEN (the door leaf is swung away from the frame). Image 2 shows the "
     "same kind of door CLOSED (the door leaf is flat in its frame). Look at image 3. Is the door in image 3 open like "
     "image 1 or closed like image 2? Answer with one digit: 1 or 2.")
# references: (video, second) — open door / closed door, as seen in the training clips (docs/VIOLATIONS.md, T6)
OPEN = {"X-32": 9.0, "X-33": 10.0}
CLOSED = {"X-31": 14.0, "X-40": 12.0}
VIDEOS = ["X-30", "X-31", "X-32", "X-33", "X-40"]


def door_crop(vid: str, t: float, door) -> object:
    im = frame_at(video_path(vid), t)
    return crop(im, engine._grow(door, 0.4, 0.05), margin=0.0, out_side=448)


def main() -> None:
    boxes = {}
    for vid in sorted(set(VIDEOS) | set(OPEN) | set(CLOSED)):
        path = video_path(vid)
        frames = sample(path, fps=engine.FPS)
        dc = DevCache(path, fps=engine.FPS)
        sc = dc.get("scene", lambda frames=frames: engine.find_scene(frames))
        dets = dc.get("dets", lambda frames=frames: people.detect(frames.images()))
        dc.save()
        boxes[vid] = (sc, engine.tracks(frames, dets, sc))
    for vid in VIDEOS:
        wait_if_hot()
        sc, tr = boxes[vid]
        if sc.door is None:
            print(vid, "no door box")
            continue
        ref_open = next((door_crop(v, t, boxes[v][0].door) for v, t in OPEN.items() if v != vid), None)
        ref_closed = next((door_crop(v, t, boxes[v][0].door) for v, t in CLOSED.items() if v != vid), None)
        n = len(tr.t)
        arrivals = [s for s, _e in engine._runs(tr.present, True, 2) if s > 0 and not any(tr.present[:s])]
        out = []
        for s in arrivals[:1]:
            for dt in (6.0, 8.0):
                t = tr.t[min(n - 1, s + round(dt * engine.FPS))]
                q = door_crop(vid, t, sc.door)
                ans, _ = vlm.ask([ref_open, ref_closed, q], Q, max_tokens=3)
                out.append((t, ans.strip()))
        print(f"{vid}: arrival at {[tr.t[s] for s in arrivals[:1]]} -> {out}", flush=True)


if __name__ == "__main__":
    main()
