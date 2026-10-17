"""All cached videos at once, no models and no decoding: the cashier's depth, two-on-her-side runs, leave starts,
client arrivals. A quick regression check of the people-layout rules (uses runs/cache from vision.evaluate --cache).

    uv run python -m vision.experiments.layout_check
"""

import csv
import os

from vision import engine
from vision.devcache import DevCache
from vision.evaluate import DUPLICATES, LABELS, ROOT, video_path
from vision.frames import info

CACHE = ROOT / "runs" / "cache"


class _Frame:
    def __init__(self, width, height):
        self.width, self.height = width, height


def main() -> None:
    os.environ["DT_CACHE"] = str(CACHE)
    gt: dict[str, list] = {}
    for r in csv.DictReader(LABELS.open(encoding="utf-8")):
        gt.setdefault(DUPLICATES.get(r["video"], r["video"]), [])
        if r["label"] == "violation":
            gt[DUPLICATES.get(r["video"], r["video"])].append(f"{r['type_id']}@{float(r['timecode_sec']):g}")
    for vid in sorted(gt, key=lambda v: int(v.split("-")[1])):
        path = video_path(vid)
        dc = DevCache(path, fps=engine.FPS)
        dets, sc = dc.get("dets", lambda: None), dc.get("scene", lambda: None)
        if dets is None or sc is None:
            print(vid, "no current cache; run vision.evaluate --cache first")
            continue
        vi = info(path)
        W, H = vi.width, vi.height
        frames = [(i / engine.FPS, _Frame(W, H)) for i in range(len(dets))]
        tr = engine.tracks(frames, dets, sc)
        n, k = len(tr.t), lambda s: max(1, round(s * engine.FPS))  # noqa: E731
        t7 = [tr.t[s] for s, _ in engine._runs([len(ps) >= 2 for ps in tr.near_cash], True, k(2.0))]
        leaves = [tr.t[engine.leave_start(tr, s, W)] for s, e in engine._runs(tr.present, False, 2)
                  if s and (e - s >= k(engine.LEAVE_MIN_S) or e == n)]
        print(f"{vid:5} gt={','.join(gt[vid]) or '-':14} depth={round(100 * tr.depth / H) if tr.depth else None}% "
              f"T7at={t7} leave={leaves} arrive={[tr.t[i] for i in engine.client_arrivals(tr)]}")


if __name__ == "__main__":
    main()
