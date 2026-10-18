"""Э16: prompt variants for "the client's ID / a document lies on the desk" (T13, X-2). Accept a prompt only if it
says "no document" on X-2 and "document" on X-3 and X-19 (same actors and room; the ID / a sheet is handed over).
Desk crops every second from 4 to 16 s after the client arrives. One video at a time, answers cached.
    uv run python -m vision.experiments.e16_doc_prompts X-2,X-3,X-19
    uv run python -m vision.experiments.e16_doc_prompts X-5,X-35 --strict-only   # window 4-20 s, strict prompt only
"""

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("DT_CACHE", str(ROOT / "runs" / "cache"))

from vision import engine, vlm  # noqa: E402
from vision.devcache import DevCache  # noqa: E402
from vision.evaluate import ground_truth, video_path  # noqa: E402
from vision.frames import crop, sample  # noqa: E402
from vision.limits import wait_if_hot  # noqa: E402

P_LIST = "List the objects lying on the desk surface, one short item per line."
DOC_WORDS = re.compile(r"\b(paper|papers|sheet|document|documents|card|passport|form|booklet|receipt|id)\b", re.I)
P_STRICT = ("Look only at the desk surface. Apart from banknotes, a laptop, a phone, a counting machine and a "
            "currency detector, is there a separate small card or a sheet of paper lying on it? If you are not sure, "
            "answer no.")


LIST = "--strict-only" not in sys.argv


def _nope():
    raise RuntimeError("not cached")


def run(vid):
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
        rows = []
        end = 16 if LIST else 20
        for i in range(i0 + round(4 * engine.FPS), min(n, i0 + round(end * engine.FPS)), round(engine.FPS)):
            wait_if_hot()
            im = crop(frames[i][1], sc.desk, margin=0.05, out_side=672)
            text = (dc.vlm_many([im], P_LIST, lambda im=im: vlm.ask([im], P_LIST, max_tokens=60)[0], max_tokens=60)
                    if LIST else "")
            strict = dc.vlm(im, P_STRICT, lambda im=im: vlm.yes_no([im], P_STRICT)[0])
            words = sorted({w.lower() for w in DOC_WORDS.findall(text)})
            rows.append((frames.timestamps[i], words, strict, " / ".join(text.split("\n"))[:90]))
        dc.save()
    return rows


def main():
    gt = ground_truth()
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    for vid in args[0].split(",") if args else ["X-2", "X-3", "X-19"]:
        rows = run(vid)
        lab = ",".join(e["type_id"] for e in gt[vid]["events"]) or "clean"
        if rows is None:
            print(f"{vid:5} {lab:6} no client visit", flush=True)
            continue
        lst = sum(bool(w) for _, w, _, _ in rows)
        st = sum(bool(s) for _, _, s, _ in rows)
        flags = [bool(s) for _, _, s, _ in rows]
        run_max = max((e - b for b, e in engine._runs(flags, True, 1)), default=0)
        line = "".join("Y" if f else "." for f in flags)
        print(f"{vid:5} {lab:6} list-doc {lst}/{len(rows)}  strict-yes {st}/{len(rows)}  longest run {run_max} s  "
              f"{line}  -> {'document seen' if run_max >= 2 else 'NO DOCUMENT (T13)'}", flush=True)
        for t, w, s, txt in rows if LIST else []:
            print(f"   {t:5.1f}s list={','.join(w) or '-':14} strict={'Y' if s else 'n'}  {txt}", flush=True)


if __name__ == "__main__":
    main()
