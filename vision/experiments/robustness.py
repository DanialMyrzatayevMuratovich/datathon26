"""Robustness to another camera (criterion 5): the same clips shifted/zoomed (crop) and mirrored, run through the
engine without cache — do the findings survive? Transformed copies go to a temp dir (never into git).
    uv run python -m vision.experiments.robustness            # 15 found violations + 5 clean, crop + mirror
"""

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

os.environ.pop("DT_CACHE", None)
os.environ["VISION_ENGINE"] = "real"
os.environ["DT_BATCH"] = "1"

from vision import engine  # noqa: E402
from vision.evaluate import ROOT, ground_truth, video_path  # noqa: E402
from vision.limits import wait_if_hot  # noqa: E402

FOUND = ["X-1", "X-4", "X-11", "X-13", "X-15", "X-17", "X-20", "X-24", "X-26", "X-28", "X-30", "X-34", "X-38",
         "X-40", "X-46"]
CLEAN = ["X-3", "X-23", "X-29", "X-35", "X-39"]
TRANSFORMS = {
    "crop": "crop=trunc(iw*0.85/2)*2:trunc(ih*0.88/2)*2:trunc(iw*0.15/2)*2:trunc(ih*0.08/2)*2",  # camera shifted
    "mirror": "hflip",  # camera on the other side
}


def main() -> None:
    gt = ground_truth()
    out_dir = ROOT / "runs" / "robustness"
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {}
    with tempfile.TemporaryDirectory() as td:
        for name, vf in TRANSFORMS.items():
            for vid in FOUND + CLEAN:
                wait_if_hot()
                src = video_path(vid)
                dst = Path(td) / f"{vid}_{name}.mov"
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-threads", "2", "-i", str(src), "-vf", vf, "-an",
                                "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", str(dst)], check=True)
                t = time.perf_counter()
                a = engine.run(dst)
                types = [(v.type_id, v.timecode_sec) for v in a.violations]
                ok = (a.verdict == "violation") == gt[vid]["violation"]
                res[f"{vid}:{name}"] = {"verdict_ok": ok, "types": types,
                                        "gt": [(e["type_id"], e["t"]) for e in gt[vid]["events"]]}
                print(f"{name:6} {vid:5} {'OK ' if ok else 'BAD'} {types} gt={res[f'{vid}:{name}']['gt']} "
                      f"{time.perf_counter() - t:.0f}s", flush=True)
                dst.unlink()
                (out_dir / "latest.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
                time.sleep(3)
    for name in TRANSFORMS:
        rows = [v for k, v in res.items() if k.endswith(":" + name)]
        print(f"{name}: verdict ok {sum(r['verdict_ok'] for r in rows)}/{len(rows)}")


if __name__ == "__main__":
    main()
