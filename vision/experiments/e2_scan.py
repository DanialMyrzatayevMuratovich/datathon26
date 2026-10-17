"""Э2: scan one video with one VLM question every N seconds — for new rules (T6 door, T9 manual cash).

    uv run python -m vision.experiments.e2_scan X-2,X-13 machine,hands --every 2
    uv run python -m vision.experiments.e2_scan X-32 door --from 3 --to 14

Prints the answer per time and saves a review sheet of the crops to data/processed/e2/ (git-ignored).
Answers are cached in runs/cache like the engine's, so a rerun costs nothing.
"""

import argparse
import os
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("DT_CACHE", str(ROOT / "runs" / "cache"))

from vision import engine, scene, vlm  # noqa: E402
from vision.devcache import DevCache  # noqa: E402
from vision.evaluate import video_path  # noqa: E402
from vision.frames import crop, sample  # noqa: E402

OUT = ROOT / "data" / "processed" / "e2"

# name -> (scene object to crop, question)
QUESTIONS = {
    "machine": ("machine", "This is a close-up of a banknote counting machine on a desk. Are banknotes being "
                           "counted by the machine right now — notes in its top hopper or in its output tray, or a "
                           "hand putting notes into it?"),
    "hands": ("desk", "Is a person holding banknotes or a stack of paper money in their hands in this image?"),
    "door": ("door", "This is a close-up of a glass office door. Is the door open (not shut in its frame)?"),
}


def scene_box(dc: DevCache, frames, key: str):
    if key in ("desk", "laptop", "safe"):
        return getattr(dc.get("scene", lambda: engine.find_scene(frames)), key)

    def find():
        best = None
        for _, im in (frames[0], frames[len(frames) // 2], frames[-1]):
            items = scene.detect(im, [key], threshold=0.25).get(key, [])
            if items and (best is None or items[0][1] > best[1]):
                best = items[0]
        return best[0] if best else None
    return dc.get(("scene_extra", key), find)


def scan(vid: str, name: str, every: float, t0: float, t1: float, sheet: bool = True) -> list:
    path = video_path(vid)
    frames = sample(path, fps=engine.FPS)
    dc = DevCache(path)
    key, q = QUESTIONS[name]
    box = scene_box(dc, frames, key)
    if box is None:
        dc.save()
        print(f"{vid} {name}: no {key}")
        return []
    step = max(1, round(every * engine.FPS))
    crops, rows = [], []
    for t, im in frames[::step]:
        if not t0 <= t <= t1:
            continue
        c = crop(im, box, margin={"desk": 0.2, "door": 0.1}.get(key, 0.25))
        ans = dc.vlm(c, q, lambda c=c: vlm.yes_no([c], q)[0])
        rows.append((t, ans))
        crops.append((t, ans, c))
    dc.save()
    print(f"{vid} {name}: yes at {[t for t, x in rows if x]} of {len(rows)}", flush=True)
    if sheet and crops:
        OUT.mkdir(parents=True, exist_ok=True)
        w = 220
        im = Image.new("RGB", (w * min(8, len(crops)), w * ((len(crops) + 7) // 8)), "white")
        for i, (t, ans, c) in enumerate(crops):
            th = c.copy()
            th.thumbnail((w, w))
            x, y = (i % 8) * w, (i // 8) * w
            im.paste(th, (x, y))
            ImageDraw.Draw(im).text((x + 4, y + 4), f"{t:.1f}s {ans}", fill="yellow")
        im.save(OUT / f"{vid}_{name}.jpg", quality=85)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", help="comma-separated, e.g. X-2,X-13")
    ap.add_argument("questions", help=f"comma-separated of {sorted(QUESTIONS)}")
    ap.add_argument("--every", type=float, default=2.0)
    ap.add_argument("--from", dest="t0", type=float, default=0.0)
    ap.add_argument("--to", dest="t1", type=float, default=1e9)
    a = ap.parse_args()
    from vision.limits import wait_if_hot
    for vid in a.videos.split(","):
        wait_if_hot()
        for name in a.questions.split(","):
            scan(vid.strip(), name.strip(), a.every, a.t0, a.t1)


if __name__ == "__main__":
    main()
