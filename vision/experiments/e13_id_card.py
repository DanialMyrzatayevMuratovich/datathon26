"""Э13: money taken without the client's ID (T13, X-2). Earlier "is the client showing a document?" on whole frames
was noise (X-2 1 yes, clean X-35/X-37 2 each). Here: zoomed crops of the client (arms and hands) and of the desk,
frame by frame from the client's arrival for 10 s, with an ID-card question. One video at a time, answers cached.
    uv run python -m vision.experiments.e13_id_card X-2
    uv run python -m vision.experiments.e13_id_card X-35 --sheets
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("DT_CACHE", str(ROOT / "runs" / "cache"))

from PIL import Image  # noqa: E402

from vision import engine, vlm  # noqa: E402
from vision.devcache import DevCache  # noqa: E402
from vision.evaluate import ground_truth, video_path  # noqa: E402
from vision.frames import crop, sample  # noqa: E402
from vision.limits import wait_if_hot  # noqa: E402

WINDOW_S = 10.0
Q_HAND = ("Look at this person's hands. Is the person holding or handing over an identity card or an ID document "
          "(a small plastic card or a passport booklet)? Answer no for banknotes, a phone or nothing.")
Q_DESK = ("Is there an identity card or an ID document (a small plastic card with a photo, or a passport) on this "
          "desk? Answer no for banknotes, papers, a phone or a laptop.")
OUT = Path(os.getenv("E13_OUT", ROOT / "runs" / "e13"))


def _nope():
    raise RuntimeError("not cached")


def run(vid, sheets):
    path = video_path(vid)
    with sample(path, fps=engine.FPS) as frames:
        dc = DevCache(path, fps=engine.FPS)
        sc = dc.get("scene", _nope)
        tr = engine.tracks(frames, dc.get("dets", _nope), sc)
        n = len(frames)
        arrivals = engine.client_arrivals(tr)
        if not arrivals:
            return []
        i0 = arrivals[0]
        rows, crops = [], []
        for i in range(i0, min(n, i0 + round(WINDOW_S * engine.FPS))):
            if not tr.behind[i]:
                continue
            wait_if_hot()
            p = max(tr.behind[i], key=lambda q: q.height)  # the client
            im_h = crop(frames[i][1], p.box, margin=0.2, out_side=448)
            hand = dc.vlm(im_h, Q_HAND, lambda im=im_h: vlm.yes_no([im], Q_HAND)[0])
            desk = any(dc.vlm(im, Q_DESK, lambda im=im: vlm.yes_no([im], Q_DESK)[0])
                       for im in engine._tiles(frames[i][1], sc.desk)[:3])  # the client's half (far row)
            rows.append((frames.timestamps[i], hand, desk))
            crops.append(im_h.resize((112, round(112 * im_h.height / im_h.width))))
        dc.save()
    if sheets and crops:
        OUT.mkdir(parents=True, exist_ok=True)
        sheet = Image.new("RGB", (112 * len(crops), max(c.height for c in crops)), "white")
        for k, c in enumerate(crops):
            sheet.paste(c, (112 * k, 0))
        sheet.save(OUT / f"{vid}_id.jpg", quality=85)
    return rows


def main():
    gt = ground_truth()
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    for vid in args[0].split(",") if args else ["X-2"]:
        rows = run(vid, "--sheets" in sys.argv)
        lab = ",".join(e["type_id"] for e in gt[vid]["events"]) or "clean"
        cells = " ".join(f"{t:.1f}:{'H' if h else 'h'}{'D' if d else 'd'}" for t, h, d in rows)
        print(f"{vid} {lab}: {cells}  | hand-ID {sum(bool(h) for _, h, _ in rows)}/{len(rows)}, "
              f"desk-ID {sum(d for _, _, d in rows)}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
