"""Grounding DINO on CPU vs the Apple GPU (MPS): speed and whether the boxes stay the same.

    uv run python -m vision.experiments.gdino_device X-28,X-34,X-13
Each device runs in its own process (the model is cached per process). Prints seconds per call and box differences.
"""

import json
import os
import subprocess
import sys

CHILD = r"""
import json, os, sys, time
from vision import scene
from vision.evaluate import video_path
from vision.frames import sample
out = {}
for vid in sys.argv[1].split(","):
    frames = sample(video_path(vid), fps=2.0)
    picks = [frames[0][1], frames[len(frames) // 2][1], frames[-1][1]]
    scene.detect(picks[0], ["desk"], threshold=0.25)  # warm-up
    t = time.perf_counter()
    res = [scene.detect(im, ["desk", "laptop", "safe", "door"], threshold=0.25) for im in picks]
    res += [scene.detect(im, ["machine"], threshold=0.25) for im in picks]
    dt = (time.perf_counter() - t) / len(res)
    out[vid] = {"sec_per_call": round(dt, 3), "boxes": [{k: v[0][0] if v else None for k, v in r.items()} for r in res]}
print(json.dumps(out))
"""


def main() -> None:
    vids = sys.argv[1] if len(sys.argv) > 1 else "X-28,X-34,X-13"
    res = {}
    for dev in ("cpu", "mps"):
        env = dict(os.environ, DT_GDINO_DEVICE=dev)
        r = subprocess.run([sys.executable, "-c", CHILD, vids], capture_output=True, text=True, env=env)
        res[dev] = json.loads(r.stdout.strip().splitlines()[-1])
    for vid in vids.split(","):
        c, m = res["cpu"][vid], res["mps"][vid]
        diff = 0.0
        for bc, bm in zip(c["boxes"], m["boxes"], strict=True):
            for k in bc:
                if bc[k] and bm.get(k):
                    diff = max(diff, max(abs(x - y) for x, y in zip(bc[k], bm[k], strict=True)))
                elif bool(bc[k]) != bool(bm.get(k)):
                    diff = float("inf")
        print(f"{vid}: cpu {c['sec_per_call']} s/вызов, mps {m['sec_per_call']} s/вызов, "
              f"макс. расхождение рамок {diff:.1f} px")


if __name__ == "__main__":
    main()
