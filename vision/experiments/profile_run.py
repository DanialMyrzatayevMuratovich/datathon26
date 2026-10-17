"""Where the time goes in one uncached engine run: decode, people, scene objects, VLM, rules.
    uv run python -m vision.experiments.profile_run X-14
"""

import os
import sys
import time
from collections import defaultdict

os.environ.pop("DT_CACHE", None)
os.environ["VISION_ENGINE"] = "real"

from vision import engine, frames, people, scene, vlm  # noqa: E402
from vision.evaluate import video_path  # noqa: E402

T: dict[str, float] = defaultdict(float)
N: dict[str, int] = defaultdict(int)


def timed(mod, name, label):
    fn = getattr(mod, name)

    def wrap(*a, **k):
        t = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            T[label] += time.perf_counter() - t
            N[label] += 1
    setattr(mod, name, wrap)


timed(engine, "sample", "decode frames")
timed(people, "detect", "people (RT-DETR)")
timed(scene, "detect", "objects (Grounding DINO)")
timed(vlm, "ask", "VLM questions")
timed(vlm, "load", "VLM load")
t0 = time.perf_counter()
res = engine.run(video_path(sys.argv[1] if len(sys.argv) > 1 else "X-14"))
total = time.perf_counter() - t0
print(f"total {total:.1f} s, duration {res.duration_sec} s, frames ~{int(res.duration_sec * engine.FPS)}")
for k, v in sorted(T.items(), key=lambda x: -x[1]):
    print(f"  {k:26} {v:6.1f} s  ({N[k]} calls)")
print("  rules + other            ", round(total - sum(v for k, v in T.items() if k != 'VLM questions' or True), 1))
print(res.engine)
_ = frames
