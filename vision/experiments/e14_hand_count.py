"""Э14: partial count (T9, X-44) — Danial's reading: after the machine, the clean cashiers (X-43, X-45) recount the
notes by hand, X-44 only holds them and puts them down. Per frame, a crop of the cashier goes to Qwen: "counting
banknotes by hand right now?". Prints the timeline (C = counting). One video at a time, answers cached.
    uv run python -m vision.experiments.e14_hand_count X-44,X-43,X-45
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("DT_CACHE", str(ROOT / "runs" / "cache"))

from vision import engine, vlm  # noqa: E402
from vision.devcache import DevCache  # noqa: E402
from vision.evaluate import ground_truth, video_path  # noqa: E402
from vision.frames import crop, sample  # noqa: E402
from vision.limits import wait_if_hot  # noqa: E402

Q_COUNT = ("Is this person counting banknotes by hand right now — holding a stack of notes and flipping through them "
           "one by one with the fingers? Answer no if the notes only lie on the desk, are just held or put down, or "
           "the hands are empty.")


def _nope():
    raise RuntimeError("not cached")


def run(vid):
    path = video_path(vid)
    with sample(path, fps=engine.FPS) as frames:
        dc = DevCache(path, fps=engine.FPS)
        sc = dc.get("scene", _nope)
        tr = engine.tracks(frames, dc.get("dets", _nope), sc)
        rows = []
        for i in range(len(frames)):
            if not tr.at_work[i]:
                rows.append((frames.timestamps[i], None))
                continue
            wait_if_hot()
            p = max(tr.at_work[i], key=lambda q: q.height)
            im = crop(frames[i][1], p.box, margin=0.1, out_side=448)
            rows.append((frames.timestamps[i], dc.vlm(im, Q_COUNT, lambda im=im: vlm.yes_no([im], Q_COUNT)[0])))
        dc.save()
    return rows


def main():
    gt = ground_truth()
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    for vid in args[0].split(",") if args else ["X-44"]:
        rows = run(vid)
        lab = ",".join(e["type_id"] for e in gt[vid]["events"]) or "clean"
        line = "".join("C" if a else "." if a is False else "_" for _, a in rows)
        hits = [t for t, a in rows if a]
        print(f"{vid:5} {lab:6} counting {len(hits)}/{len(rows)} frames  {line}  at {hits}", flush=True)


if __name__ == "__main__":
    main()
