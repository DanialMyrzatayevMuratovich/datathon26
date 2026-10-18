"""Э15: money taken without the client's ID (T13, X-2) — Danial's reading: when the client hands over an ID or a
document, it lies on the desk while the money is counted (X-3: the ID first, then the money; X-19: a sheet on the
desk while the machine counts); in X-2 the client hands over only the money. Every second of the client's visit, the
whole desk goes to Qwen: "a document on the desk (paper, form, passport, ID card — not banknotes)?".
One video at a time, answers cached.
    uv run python -m vision.experiments.e15_doc_on_desk X-2
    uv run python -m vision.experiments.e15_doc_on_desk X-3,X-19 --sheets
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

VISIT_S = 30.0
Q_DOC = ("Is there a document lying on this desk — a white sheet of paper, a form, a passport or an ID card? Answer no "
         "for banknotes or a stack of money, a laptop, a phone, a keyboard or a counting machine.")
OUT = Path(os.getenv("E15_OUT", ROOT / "runs" / "e15"))


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
        if not arrivals or sc.desk is None:
            return None
        i0 = arrivals[0]
        rows, ims = [], []
        for i in range(i0, min(n, i0 + round(VISIT_S * engine.FPS)), round(engine.FPS)):
            wait_if_hot()
            im = crop(frames[i][1], sc.desk, margin=0.05, out_side=672)
            rows.append((frames.timestamps[i], dc.vlm(im, Q_DOC, lambda im=im: vlm.yes_no([im], Q_DOC)[0])))
            ims.append(im.resize((224, round(224 * im.height / im.width))))
        dc.save()
    if sheets and ims:
        OUT.mkdir(parents=True, exist_ok=True)
        h = max(i.height for i in ims)
        sheet = Image.new("RGB", (224 * min(10, len(ims)), h * ((len(ims) + 9) // 10)), "white")
        for k, im in enumerate(ims):
            sheet.paste(im, (224 * (k % 10), im.height * (k // 10)))
        sheet.save(OUT / f"{vid}_doc.jpg", quality=85)
    return rows


def main():
    gt = ground_truth()
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    for vid in args[0].split(",") if args else ["X-2"]:
        rows = run(vid, "--sheets" in sys.argv)
        lab = ",".join(e["type_id"] for e in gt[vid]["events"]) or "clean"
        if rows is None:
            print(f"{vid:5} {lab:6} no client visit", flush=True)
            continue
        line = "".join("D" if a else "." if a is False else "?" for _, a in rows)
        hits = [t for t, a in rows if a]
        print(f"{vid:5} {lab:6} document on desk {len(hits)}/{len(rows)} s  {line}  at {hits}", flush=True)


if __name__ == "__main__":
    main()
