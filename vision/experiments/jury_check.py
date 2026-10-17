"""Check like the jury: every video through the running API (real engine, no cache), one at a time, then the same
metrics as vision.evaluate from the API answers.

    VISION_ENGINE=real SERVICE_WORKERS=1 bash scripts/run_service.sh     # in another terminal
    uv run python -m vision.experiments.jury_check                        # all 40 -> runs/jury/<run_id>/
    uv run python -m vision.experiments.jury_check --only X-28,X-15

Uses `service.cli check --strict --expect-engine real` per video (the service reads its own key; nothing is printed).
"""

import argparse
import json
import secrets
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from vision.evaluate import ROOT, ground_truth, metrics, video_path
from vision.limits import wait_if_hot

OUT = ROOT / "runs" / "jury"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--pause", type=float, default=3.0)
    ap.add_argument("--parallel", type=int, default=1, help="videos in flight at once (= SERVICE_WORKERS)")
    ap.add_argument("--resume", default="", help="runs/jury/<run_id>: keep its answers, send only the rest")
    a = ap.parse_args()
    gt = ground_truth()
    vids = [v.strip() for v in a.only.split(",") if v.strip()] or sorted(gt, key=lambda v: int(v.split("-")[1]))
    out = Path(a.resume) if a.resume else OUT / f"{datetime.now():%Y%m%dT%H%M}-{secrets.token_hex(2)}"
    out.mkdir(parents=True, exist_ok=True)
    pred = json.loads((out / "predictions.json").read_text()) if (out / "predictions.json").exists() else {}
    vids = [v for v in vids if v not in pred]
    lock = threading.Lock()

    def one(vid):
        wait_if_hot()
        rep = out / f"{vid}.json"
        r = subprocess.run([sys.executable, "-m", "service.cli", "check", str(video_path(vid)), "--count", "1",
                            "--strict", "--expect-engine", "real", "--url", a.url, "--report", str(rep)],
                           cwd=ROOT, capture_output=True, text=True)
        row = json.loads(rep.read_text())["results"][0] if rep.exists() else {"status": "failed", "events": []}
        ok = row.get("status") == "done"
        with lock:
            pred[vid] = {"verdict": row.get("verdict") if ok else "failed",
                         "violations": [{"type_id": e["type_id"], "timecode_sec": e["timecode_sec"]}
                                        for e in row.get("events", [])],
                         "processing_sec": row.get("elapsed_sec", 0)}
            events = [(e["type_id"], e["timecode_sec"]) for e in row.get("events", [])]
            print(f"{vid:5} {row.get('status')} {row.get('verdict')} {events} {row.get('elapsed_sec')}s "
                  f"rc={r.returncode}", flush=True)
            (out / "predictions.json").write_text(json.dumps(pred, ensure_ascii=False, indent=1), encoding="utf-8")
        time.sleep(a.pause)

    with ThreadPoolExecutor(max_workers=max(1, a.parallel)) as ex:
        list(ex.map(one, vids))
    m = metrics(gt, pred)
    m["failed"] = [v for v, p in pred.items() if p["verdict"] == "failed"]
    (out / "metrics.json").write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in m.items() if k != "per_video"}, ensure_ascii=False))
    print(f"→ {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
