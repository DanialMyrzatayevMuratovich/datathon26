"""Э6: door left open after entering — change detection, no neural net (idea from the Gemini review, SSIM).

The room is empty at the start, a person enters. The entry region = where the person first appears (expanded): that
is the door. Reference = the frame before anybody appears (door closed). After the person has walked away from the
entry region, the region should look like the reference again if the door was closed; an open door changes it.
    uv run python -m vision.experiments.e6_door_change
"""

import os
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("DT_CACHE", str(ROOT / "runs" / "cache"))

from vision import engine, people  # noqa: E402
from vision.devcache import DevCache  # noqa: E402
from vision.evaluate import ground_truth, video_path  # noqa: E402
from vision.frames import sample  # noqa: E402


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64), b.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a, mu_b = cv2.GaussianBlur(a, (7, 7), 1.5), cv2.GaussianBlur(b, (7, 7), 1.5)
    va = cv2.GaussianBlur(a * a, (7, 7), 1.5) - mu_a ** 2
    vb = cv2.GaussianBlur(b * b, (7, 7), 1.5) - mu_b ** 2
    cov = cv2.GaussianBlur(a * b, (7, 7), 1.5) - mu_a * mu_b
    m = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2))
    return float(m.mean())


def gray(im, box) -> np.ndarray:
    x0, y0, x1, y1 = (int(v) for v in box)
    c = np.asarray(im.convert("L").crop((x0, y0, x1, y1)).resize((96, 160)))
    return c


def entry_door(vid: str):
    path = video_path(vid)
    with sample(path, fps=engine.FPS) as frames:
        dc = DevCache(path, fps=engine.FPS)
        dets = dc.get("dets", lambda: people.detect(frames.images()))
        dc.save()
        W, H = frames[0][1].width, frames[0][1].height
        raw = [[d for d in ds if d.label == "person" and d.score >= 0.5 and d.height >= 0.12 * H] for ds in dets]
        k = lambda s: max(1, round(s * engine.FPS))  # noqa: E731
        if any(raw[i] for i in range(min(len(raw), k(1.0)))):
            return None  # somebody is there from the start: no entry to judge
        first = next((i for i, ps in enumerate(raw) if ps), None)
        if first is None or first == 0:
            return None
        region = engine._grow(list(max(raw[first], key=lambda p: p.score).box), 0.25, 0.05)
        region = [max(0, region[0]), max(0, region[1]), min(W, region[2]), min(H, region[3])]
        ref = gray(frames[first - 1][1], region)
        scores = []
        for i in range(first + k(3.0), min(len(raw), first + k(12.0))):
            if any(engine._ioa(region, p.box) > 0.15 for p in raw[i]):
                continue  # somebody still in the doorway
            scores.append((frames.timestamps[i], round(ssim(ref, gray(frames[i][1], region)), 3)))
        return frames.timestamps[first], [round(100 * v / W) if n % 2 == 0 else round(100 * v / H)
                                          for n, v in enumerate(region)], scores


def main() -> None:
    gt = ground_truth()
    for vid in sorted(gt, key=lambda v: int(v.split("-")[1])):
        r = entry_door(vid)
        if r is None:
            continue
        t0, reg, scores = r
        lab = ",".join(e["type_id"] for e in gt[vid]["events"]) or "-"
        med = sorted(s for _, s in scores)[len(scores) // 2] if scores else None
        print(f"{vid:5} gt={lab:8} entry@{t0:4.1f}s region%={reg} median_ssim={med} {scores[:8]}", flush=True)


if __name__ == "__main__":
    main()
