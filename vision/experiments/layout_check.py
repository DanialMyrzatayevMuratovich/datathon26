"""All cached videos at once, no models and no decoding: the cashier's depth, two-on-her-side runs, leave starts,
client arrivals. A quick regression check of the people-layout rules (uses runs/cache from vision.evaluate --cache).

    uv run python -m vision.experiments.layout_check
"""

import csv
import pickle
from pathlib import Path

from vision import engine
from vision.devcache import _file_key
from vision.evaluate import DUPLICATES, LABELS, ROOT, video_path

CACHE = ROOT / "runs" / "cache"
W, H = 464, 848  # all training videos


class _Frame:
    width, height = W, H


def main() -> None:
    gt: dict[str, list] = {}
    for r in csv.DictReader(LABELS.open(encoding="utf-8")):
        gt.setdefault(DUPLICATES.get(r["video"], r["video"]), [])
        if r["label"] == "violation":
            gt[DUPLICATES.get(r["video"], r["video"])].append(f"{r['type_id']}@{float(r['timecode_sec']):g}")
    for vid in sorted(gt, key=lambda v: int(v.split("-")[1])):
        f = CACHE / f"{_file_key(video_path(vid))}.pkl"
        if not f.exists():
            print(vid, "no cache")
            continue
        d = pickle.loads(Path(f).read_bytes())
        frames = [(i / engine.FPS, _Frame()) for i in range(len(d["dets"]))]
        tr = engine.tracks(frames, d["dets"], d["scene"])
        n, k = len(tr.t), lambda s: max(1, round(s * engine.FPS))  # noqa: E731
        t7 = [tr.t[s] for s, _ in engine._runs([len(ps) >= 2 for ps in tr.near_cash], True, k(2.0))]
        leaves = [tr.t[engine.leave_start(tr, s, W)] for s, e in engine._runs(tr.present, False, 2)
                  if s and (e - s >= k(engine.LEAVE_MIN_S) or e == n)]
        print(f"{vid:5} gt={','.join(gt[vid]) or '-':14} depth={round(100 * tr.depth / H) if tr.depth else None}% "
              f"T7at={t7} leave={leaves} arrive={[tr.t[i] for i in engine.client_arrivals(tr)]}")


if __name__ == "__main__":
    main()
