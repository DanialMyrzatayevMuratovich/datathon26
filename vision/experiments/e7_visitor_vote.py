"""Э7: a second visitor in the cash room during a client visit — majority vote over 3 parts of the visit (2 of 3
"yes"), Qwen3-VL-8B on time-ordered strips (idea from the Gemini review). Goal: keep X-18, drop the single-strip false
alarms of Э5 on clean X-5 and X-19.
    uv run python -m vision.experiments.e7_visitor_vote
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("DT_CACHE", str(ROOT / "runs" / "cache"))

from vision import analyst, engine, people  # noqa: E402
from vision.devcache import DevCache  # noqa: E402
from vision.evaluate import video_path  # noqa: E402
from vision.frames import sample  # noqa: E402
from vision.limits import wait_if_hot  # noqa: E402

Q = ("These frames are in time order (seconds in the corner) from a camera above a bank cash desk. Count the people "
     "inside the room in each frame: the cashier, the seated client, and anybody else. Is there a THIRD person inside "
     "the room (standing or walking, not seen through the glass wall) in at least two of these frames? "
     "Reply with one word: yes or no.")
VIDEOS = ["X-18", "X-5", "X-19", "X-39", "X-3", "X-35", "X-37", "X-16"]


def main() -> None:
    k = lambda s: max(1, round(s * engine.FPS))  # noqa: E731
    for vid in VIDEOS:
        wait_if_hot()
        path = video_path(vid)
        with sample(path, fps=engine.FPS) as frames:
            dc = DevCache(path, fps=engine.FPS)
            sc = dc.get("scene", lambda: engine.find_scene(frames))
            dets = dc.get("dets", lambda: people.detect(frames.images()))
            tr = engine.tracks(frames, dets, sc)
            n, res = len(frames), []
            for s, e in engine._runs(tr.client_here, True, k(5.0)):
                e = min(n - 1, e + k(2.0))
                parts = [(s + (e - s) * p // 3, s + (e - s) * (p + 1) // 3) for p in range(3)]
                votes = []
                for a, b in parts:
                    idx = sorted({round(a + (b - a) * x / 5) for x in range(6)})
                    votes.append(analyst.ask([analyst.strip(frames, idx, cols=3)], Q, dc, max_tokens=3)[:3])
                res.append((frames.timestamps[s], votes, sum(v.startswith("yes") for v in votes) >= 2))
            dc.save()
        print(vid, res, flush=True)


if __name__ == "__main__":
    main()
