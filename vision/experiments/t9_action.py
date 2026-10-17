"""Bounded local T9 action/timing probe, never enabled in production automatically.

Use positive videos AND clean controls. No labels/times/video-specific rules enter
the predictor. Output is action codes/timestamps, never frames or free text.
"""

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

from vision import evaluate, limits, vlm
from vision.devcache import DevCache
from vision.frames import crop, sample

QUESTION = (
    "These are three chronological views of the same cash desk over two seconds. "
    "Which action is clearly visible? A: the cashier is counting individual banknotes by hand, "
    "separating notes one by one. B: feeding banknotes into or using the counting machine. "
    "C: a bundle is only moved, handed over, stacked or put away without counting. "
    "D: none can be identified confidently. Do not infer counting merely because money or hands are visible. "
    "Answer only A, B, C or D."
)


def action_code(text):
    match = re.fullmatch(r"\s*([ABCD])[.!]?\s*", text.upper())
    return match.group(1) if match else "unknown"


def window_indices(start, count, fps):
    return [min(count - 1, start + round(seconds * fps)) for seconds in (0, 1, 2)]


def probe(path, *, every, max_calls, refine):
    from vision import engine
    began = time.perf_counter()
    dc, rows, calls = DevCache(path, fps=engine.FPS), {}, 0
    with sample(path, fps=engine.FPS) as frames:
        limits.wait_if_hot()
        sc = dc.get("scene", lambda: engine.find_scene(frames))
        if sc.desk is None:
            return {"complete": False, "reason": "desk_not_found", "windows": []}

        def check(index):
            nonlocal calls
            if index in rows:
                return True
            if len(rows) >= max_calls:
                return False
            limits.wait_if_hot()
            indices = window_indices(index, len(frames), engine.FPS)
            images = [crop(frames[i][1], sc.desk, margin=.2, out_side=640) for i in indices]
            def ask():
                nonlocal calls
                calls += 1
                return vlm.ask(images, QUESTION, max_tokens=4)[0]
            answer = dc.vlm_many(images, QUESTION, ask, max_tokens=4)
            rows[index] = {"start_sec": frames.timestamps[index], "end_sec": frames.timestamps[indices[-1]],
                           "action": action_code(answer)}
            dc.save()
            return True

        # Repeated end-of-clip stills are not temporal evidence.
        stop = max(0, len(frames) - round(2 * engine.FPS))
        coarse = list(range(0, stop, max(1, round(every * engine.FPS))))
        complete = all(check(i) for i in coarse)
        if refine and complete:
            around = set()
            for i, row in list(rows.items()):
                if row["action"] == "A":
                    around.update(range(max(0, i - round(every * engine.FPS)),
                                        min(stop, i + round(every * engine.FPS) + 1)))
            complete = all(check(i) for i in sorted(around))
    return {"complete": complete and bool(coarse), "actual_vlm_calls": calls,
            "seconds": round(time.perf_counter() - began, 3), "cache_enabled": dc.file is not None,
            "windows": [rows[i] for i in sorted(rows)],
            "note": "A is visible hand counting, not an automatic violation; validate full clean controls"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", required=True, help="Comma-separated positives AND negative controls")
    parser.add_argument("--videos", type=Path, default=evaluate.VIDEOS)
    parser.add_argument("--every", type=float, default=4)
    parser.add_argument("--max-calls", type=int, default=64)
    parser.add_argument("--refine", action="store_true")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("runs/jobs/t9-action.json"))
    args = parser.parse_args()
    if not 1 <= args.every <= 30 or not 1 <= args.max_calls <= 128:
        parser.error("every must be 1..30 seconds; max-calls 1..128 windows/video")
    evaluate.configure(args.cache)
    evaluate.VIDEOS = args.videos
    ids = list(dict.fromkeys(v.strip() for v in args.only.split(",") if v.strip()))
    if not ids:
        parser.error("At least one video ID is required")
    paths = {vid: evaluate.video_path(vid) for vid in ids}
    report = {"purpose": "Development T9 probe; production unchanged", "videos": {},
              "prompt_sha256": hashlib.sha256(QUESTION.encode()).hexdigest(), "model": vlm.DEFAULT_MODEL,
              "config": {"every": args.every, "max_calls": args.max_calls, "refine": args.refine,
                         "cache": args.cache, "temperature": 0, "max_tokens": 4}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for number, (vid, path) in enumerate(paths.items()):
        if number:
            time.sleep(limits.COOLDOWN_S)
        report["videos"][vid] = probe(path, every=args.every, max_calls=args.max_calls, refine=args.refine)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        row = report["videos"][vid]
        print(f"{vid}: complete={row['complete']}, windows={len(row['windows'])}", flush=True)
    print(f"Local metadata report: {args.out}")


if __name__ == "__main__":
    main()
