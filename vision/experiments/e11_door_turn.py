"""Э11: door left open — does the entering person ever turn back to the door? Closing a door means facing it, i.e.
showing the camera the opposite side from the one seen while walking in (X-31, X-40 turn their back to the camera at
the door; X-32, X-33 walk in facing the camera and never turn). The entering person is followed frame to frame on raw
boxes (as in Э9); each crop of her goes to Qwen: "turned away from the camera?". One video at a time, answers cached.
    uv run python -m vision.experiments.e11_door_turn                 # X-31, X-32, X-33, X-40
    uv run python -m vision.experiments.e11_door_turn X-32 --sheets   # also save the crop strip to the scratch dir
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("DT_CACHE", str(ROOT / "runs" / "cache"))

from PIL import Image  # noqa: E402

from vision import engine, people, vlm  # noqa: E402
from vision.devcache import DevCache  # noqa: E402
from vision.evaluate import ground_truth, video_path  # noqa: E402
from vision.frames import crop, sample  # noqa: E402
from vision.limits import wait_if_hot  # noqa: E402

VIDS = ["X-31", "X-32", "X-33", "X-40"]
WINDOW_S = 10.0
Q_BACK = ("Look at the person in the picture. Is the person turned away from the camera, so that we see the back of "
          "the head and the back, not the face?")
OUT = Path(os.getenv("E11_OUT", ROOT / "runs" / "e11"))


def entering(dets, H, n):
    """[(frame index, box)] of the person who walks into an empty room, or None when somebody is there at the start."""
    raw = [[d for d in ds if d.label == "person" and d.score >= 0.5 and d.height >= 0.12 * H] for ds in dets]
    if any(raw[i] for i in range(min(n, 2))):
        return None
    first = next((i for i, ps in enumerate(raw) if ps), None)
    if first is None:
        return None
    cur, out = max(raw[first], key=lambda p: p.score), []
    for i in range(first, min(n, first + round(WINDOW_S * engine.FPS))):
        if raw[i]:
            nxt = min(raw[i], key=lambda p: abs(p.cx - cur.cx) + abs(p.bottom - cur.bottom))
            if abs(nxt.cx - cur.cx) + abs(nxt.bottom - cur.bottom) < 0.25 * H:
                cur = nxt
                out.append((i, cur))
    return out


def run(vid, sheets):
    path = video_path(vid)
    with sample(path, fps=engine.FPS) as frames:
        dc = DevCache(path, fps=engine.FPS)
        dets = dc.get("dets", lambda: people.detect(frames.images()))
        W, H = frames[0][1].width, frames[0][1].height
        tr = entering(dets, H, len(frames))
        if tr is None:
            return None
        rows, crops = [], []
        for i, p in tr:
            wait_if_hot()
            im = crop(frames[i][1], p.box, margin=0.15, out_side=448)
            ans = dc.vlm(im, Q_BACK, lambda im=im: vlm.yes_no([im], Q_BACK)[0])
            rows.append((frames.timestamps[i], round(100 * p.cx / W), round(100 * p.bottom / H), ans))
            crops.append(im.resize((112, round(112 * im.height / im.width))))
        dc.save()
    if sheets and crops:
        OUT.mkdir(parents=True, exist_ok=True)
        h = max(c.height for c in crops)
        sheet = Image.new("RGB", (112 * len(crops), h), "white")
        for k, c in enumerate(crops):
            sheet.paste(c, (112 * k, 0))
        sheet.save(OUT / f"{vid}_turn.jpg", quality=85)
    return rows


def main():
    gt = ground_truth()
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    for vid in args[0].split(",") if args else VIDS:
        rows = run(vid, "--sheets" in sys.argv)
        lab = ",".join(e["type_id"] for e in gt[vid]["events"]) or "-"
        if rows is None:
            print(f"{vid} gt={lab}: nobody enters an empty room", flush=True)
            continue
        cells = " ".join(f"{t:.1f}:{x},{b}{'B' if a else 'f' if a is False else '?'}" for t, x, b, a in rows)
        print(f"{vid} gt={lab}: {cells}", flush=True)


if __name__ == "__main__":
    main()
