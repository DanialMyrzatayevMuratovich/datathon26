"""Э1 (docs/ML_SPEC.md §10): does the local VLM see the details on crops found by Grounding DINO?

Ground truth comes from the manual review (docs/VIDEO_REVIEW_NOTES.md, contrast pairs in docs/ANALYSIS_RULES.txt).
Phase `crops`: detect the object, crop, save + a review sheet. Phase `ask`: yes/no questions to each VLM, accuracy and
latency -> runs/e1/<model>.json. Usage:
  uv run python -m vision.experiments.e1_vlm_probe crops
  uv run python -m vision.experiments.e1_vlm_probe ask --models qwen3-vl-4b-instruct-mlx-4bit qwen3-vl-8b-instruct-mlx-4bit
"""

import argparse
import json
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from vision.frames import crop, frame_at
from vision.scene import best, detect

ROOT = Path(__file__).resolve().parents[2]
VIDEOS = ROOT / "data" / "raw" / "open" / "videos"
OUT = ROOT / "data" / "processed" / "e1"
RUNS = ROOT / "runs" / "e1"

QUESTIONS = {
    "key": ("safe", "This is a close-up of a grey metal safe. Is there a key inserted in the keyhole of the safe door "
                    "(a small key or key head sticking out of the lock)?"),
    "safe_open": ("safe", "This is a close-up of a grey metal safe. Is the safe door open?"),
    "screen": ("laptop", "This is a close-up of a laptop. Does the laptop screen show an open working application "
                         "(for example a spreadsheet, a table or a program window)? Answer no if the screen shows a lock "
                         "screen, a wallpaper picture, or is dark or closed."),
    "cash": ("desk", "This is a close-up of a desk. Are there banknotes or a stack of cash lying on the desk?"),
}

# (video id, second, attribute, expected) — None = not sure, reported but not scored
ITEMS = [
    ("X-22", 17, "key", True), ("X-22", 3, "key", True), ("X-23", 2, "key", True),
    ("X-23", 16, "key", False), ("X-29", 15, "key", False), ("X-15", 19, "key", False),
    ("X-28", 11, "key", False), ("X-28", 2, "key", False), ("X-24", 11, "key", None),
    ("X-20", 7, "safe_open", True), ("X-20", 9, "safe_open", True), ("X-35", 42.6, "safe_open", True),
    ("X-39", 66.9, "safe_open", True), ("X-22", 17, "safe_open", False), ("X-23", 16, "safe_open", False),
    ("X-38", 32, "safe_open", False), ("X-5", 42.6, "safe_open", False),
    ("X-28", 11, "screen", True), ("X-28", 2, "screen", True), ("X-23", 2, "screen", True),
    ("X-5", 42.6, "screen", True), ("X-23", 16, "screen", False), ("X-29", 15, "screen", False),
    ("X-22", 17, "screen", False),
    ("X-26", 12, "cash", True), ("X-38", 32, "cash", True), ("X-4", 56, "cash", True),
    ("X-29", 15, "cash", False), ("X-23", 16, "cash", False), ("X-28", 11, "cash", False),
]


def video_path(vid: str) -> Path:
    for p in VIDEOS.glob("*.mov"):
        if p.stem.replace("Х", "X") == vid:
            return p
    raise FileNotFoundError(vid)


def item_name(vid, t, attr) -> str:
    return f"{vid}_{t}_{attr}".replace(".", "p")


def make_crops() -> list[dict]:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, cache = [], {}
    for vid, t, attr, expected in ITEMS:
        obj = QUESTIONS[attr][0]
        key = (vid, t)
        if key not in cache:
            im = frame_at(video_path(vid), t)
            cache[key] = (im, detect(im, ["safe", "laptop", "desk"]))
        im, found = cache[key]
        box = best(found, obj, 0.25)
        row = {"video": vid, "t": t, "attr": attr, "expected": expected, "object": obj, "box": box,
               "crop": None}
        if box:
            path = OUT / f"{item_name(vid, t, attr)}.jpg"
            crop(im, box).save(path, quality=92)
            row["crop"] = str(path.relative_to(ROOT))
        rows.append(row)
        print(f"{vid} t={t} {attr}: {obj} box={box}")
    (OUT / "items.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    sheet(rows, OUT / "sheet.jpg")
    return rows


def sheet(rows: list[dict], path: Path, answers: dict | None = None) -> None:
    font = ImageFont.load_default(size=18)
    tiles = []
    for r in rows:
        im = Image.open(ROOT / r["crop"]).convert("RGB") if r["crop"] else Image.new("RGB", (448, 448), (60, 0, 0))
        im = im.resize((300, round(300 * im.height / im.width))) if im.width > im.height else \
            im.resize((round(300 * im.width / im.height), 300))
        tile = Image.new("RGB", (300, 340), (25, 25, 25))
        tile.paste(im, ((300 - im.width) // 2, 40))
        d = ImageDraw.Draw(tile)
        exp = {True: "YES", False: "NO", None: "?"}[r["expected"]]
        label = f"{r['video']} {r['t']}s {r['attr']} exp={exp}"
        if answers:
            label += f" got={answers.get(item_name(r['video'], r['t'], r['attr']), '-')}"
        d.text((4, 8), label, fill=(255, 255, 0), font=font)
        tiles.append(tile)
    cols = 6
    rows_n = (len(tiles) + cols - 1) // cols
    c = Image.new("RGB", (cols * 304, rows_n * 344), (10, 10, 10))
    for i, tl in enumerate(tiles):
        c.paste(tl, ((i % cols) * 304, (i // cols) * 344))
    c.save(path, quality=88)


def ask(models: list[str]) -> None:
    from vision import vlm

    rows = json.loads((OUT / "items.json").read_text(encoding="utf-8"))
    RUNS.mkdir(parents=True, exist_ok=True)
    for m in models:
        t0 = time.perf_counter()
        vlm.load(m)
        load_s = time.perf_counter() - t0
        res, answers = [], {}
        for r in rows:
            if not r["crop"]:
                res.append({**r, "answer": None, "raw": "no crop", "seconds": 0, "correct": None})
                continue
            ans, raw, dt = vlm.yes_no([Image.open(ROOT / r["crop"])], QUESTIONS[r["attr"]][1], model_name=m)
            correct = None if r["expected"] is None or ans is None else ans == r["expected"]
            res.append({**r, "answer": ans, "raw": raw, "seconds": round(dt, 2), "correct": correct})
            answers[item_name(r["video"], r["t"], r["attr"])] = {True: "YES", False: "NO", None: "??"}[ans]
        scored = [x for x in res if x["expected"] is not None]
        by_attr = {}
        for a in QUESTIONS:
            xs = [x for x in scored if x["attr"] == a]
            by_attr[a] = f"{sum(bool(x['correct']) for x in xs)}/{len(xs)}"
        summary = {"model": m, "load_s": round(load_s, 1),
                   "accuracy": round(sum(bool(x["correct"]) for x in scored) / len(scored), 3),
                   "by_attr": by_attr, "mean_s": round(sum(x["seconds"] for x in res) / len(res), 2),
                   "unparsed": sum(x["answer"] is None for x in res)}
        (RUNS / f"{m}.json").write_text(json.dumps({"summary": summary, "items": res}, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
        sheet(rows, OUT / f"sheet_{m}.jpg", answers)
        print(json.dumps(summary, ensure_ascii=False))
        vlm.load.cache_clear()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=["crops", "ask"])
    p.add_argument("--models", nargs="+", default=["qwen3-vl-4b-instruct-mlx-4bit"])
    a = p.parse_args()
    make_crops() if a.phase == "crops" else ask(a.models)


if __name__ == "__main__":
    main()
