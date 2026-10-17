"""Keep the fanless MacBook Air below ~80 % CPU: thread caps for torch/BLAS/ffmpeg + pause batch runs when hot.

scripts/macmon.sh writes runs/.hot when CPU > 80 % for ~45 s or macOS thermal state >= 2; batch jobs call wait_if_hot().
The live service does not pause (the jury's 5-minute limit) — it relies on the thread caps.
"""

import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
THREADS = int(os.getenv("DT_THREADS", "2"))  # of 10 cores on M4 (~20 %); GPU work (MLX/MPS) is separate
COOLDOWN_S = float(os.getenv("DT_COOLDOWN_S", "20"))  # pause between videos in batch runs — let the fanless Air cool
FFMPEG_THREADS = os.getenv("DT_FFMPEG_THREADS", "2")
HOT = ROOT / "runs" / ".hot"

for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(var, str(THREADS))
_applied = False


def apply_torch() -> None:
    global _applied
    if not _applied:
        import torch

        torch.set_num_threads(THREADS)
        _applied = True


def wait_if_hot(poll_s: float = 10.0, log=print, *, stale_s: float = 90.0, max_wait_s: float = 900.0) -> None:
    """Wait for a live monitor to clear the flag; fail rather than ignore stale heat evidence."""
    if not HOT.exists():
        return
    log("⏸ Мак перегружен/греется (runs/.hot) — жду, пока остынет…")
    started = time.monotonic()
    while True:
        try:
            age = time.time() - HOT.stat().st_mtime
        except FileNotFoundError:
            break
        if age > stale_s:
            raise RuntimeError("Thermal monitor heartbeat is stale; restart scripts/macmon.sh and retry. "
                               "The heat flag was not removed and inference was not resumed.")
        if time.monotonic() - started >= max_wait_s:
            raise TimeoutError("Mac remains busy/hot; batch stopped after bounded wait. Check runs/macmon.log.")
        time.sleep(poll_s)
    log("▶ продолжаю")
