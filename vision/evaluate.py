"""Evaluation and the double-run check (docs/ML_SPEC.md §5, §9).

  uv run python -m vision.evaluate                       # all 40 unique videos -> runs/eval/<run_id>/
  uv run python -m vision.evaluate --only X-28,X-26      # subset for quick iterations
  uv run python -m vision.evaluate --compare <run1> <run2>   # second run must match the first exactly
  uv run python -m vision.evaluate --cache --only X-28    # rule tuning: detections/VLM answers from runs/cache

Duplicates (X-47 = X-4, X-48 = X-5) are one video; X-4 then has two labelled events.
"""

import argparse
import csv
import json
import os
import secrets
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VIDEOS = ROOT / "data" / "raw" / "open" / "videos"
LABELS = ROOT / "data" / "annotations.csv"
RUNS = ROOT / "runs" / "eval"
DUPLICATES = {"X-47": "X-4", "X-48": "X-5"}
DT_OK = 2.0


def video_path(vid: str) -> Path:
    for p in VIDEOS.glob("*.mov"):
        if p.stem.replace("Х", "X") == vid:
            return p
    raise FileNotFoundError(vid)


def ground_truth() -> dict[str, dict]:
    gt: dict[str, dict] = {}
    with LABELS.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            vid = DUPLICATES.get(r["video"], r["video"])
            g = gt.setdefault(vid, {"violation": False, "events": []})
            if r["label"] == "violation":
                g["violation"] = True
                g["events"].append({"t": float(r["timecode_sec"]), "type_id": r["type_id"],
                                    "confidence": r["confidence"]})
    return gt


def metrics(gt: dict, pred: dict) -> dict:
    tp = fp = fn = tn = 0
    ev_total = ev_type = ev_time = 0
    per_video = {}
    for vid, g in gt.items():
        p = pred.get(vid)
        if p is None:
            continue
        pv = p["verdict"] == "violation"
        if g["violation"] and pv:
            tp += 1
        elif g["violation"]:
            fn += 1
        elif pv:
            fp += 1
        else:
            tn += 1
        hits = []
        for e in g["events"]:
            ev_total += 1
            same = [v for v in p["violations"] if v["type_id"] == e["type_id"]]
            near = [v for v in p["violations"] if abs(v["timecode_sec"] - e["t"]) <= DT_OK]
            ev_type += bool(same)
            ev_time += bool(near)
            hits.append({"gt": e, "type_hit": bool(same), "time_hit": bool(near)})
        per_video[vid] = {"gt": g["violation"], "pred": pv, "events": hits,
                          "pred_types": [(v["type_id"], v["timecode_sec"]) for v in p["violations"]],
                          "seconds": p["processing_sec"]}
    n = tp + fp + fn + tn
    return {
        "videos": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "recall": round(tp / max(1, tp + fn), 3), "precision": round(tp / max(1, tp + fp), 3),
        "accuracy": round((tp + tn) / max(1, n), 3),
        "type_hit": f"{ev_type}/{ev_total}", "time_hit_2s": f"{ev_time}/{ev_total}",
        "max_seconds": max((v["seconds"] for v in per_video.values()), default=0),
        "per_video": per_video,
    }


def run(only: list[str] | None) -> Path:
    os.environ.setdefault("VISION_ENGINE", "real")
    os.environ["DT_BATCH"] = "1"  # the engine pauses mid-video when runs/.hot appears
    from vision import analyze

    gt = ground_truth()
    vids = only or sorted(gt, key=lambda v: int(v.split("-")[1]))
    run_id = f"{datetime.now():%Y%m%dT%H%M}-{secrets.token_hex(2)}"
    out = RUNS / run_id
    out.mkdir(parents=True, exist_ok=True)
    pred = {}
    from vision.limits import COOLDOWN_S, wait_if_hot

    for n_done, vid in enumerate(vids):
        if n_done:
            time.sleep(COOLDOWN_S)  # one video at a time, with a cool-down pause
        wait_if_hot()
        t0 = time.perf_counter()
        a = analyze(video_path(vid)).model_dump()
        for v in a["violations"]:  # keep the file small: frames are re-creatable
            v["evidence_frames"] = [{"timestamp_sec": f["timestamp_sec"]} for f in v["evidence_frames"]]
        pred[vid] = a
        g = gt.get(vid, {})
        print(f"{vid:5} gt={'V' if g.get('violation') else '-'} pred={'V' if a['verdict'] == 'violation' else '-'} "
              f"{[(v['type_id'], v['timecode_sec']) for v in a['violations']]} gt_ev="
              f"{[(e['type_id'], e['t']) for e in g.get('events', [])]} {time.perf_counter() - t0:.0f}s", flush=True)
        (out / "predictions.json").write_text(json.dumps(pred, ensure_ascii=False, indent=1), encoding="utf-8")
    m = metrics(gt, pred)
    (out / "metrics.json").write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in m.items() if k != "per_video"}, ensure_ascii=False))
    print(f"→ {out.relative_to(ROOT)}")
    return out


def compare(a: str, b: str) -> int:
    pa = json.loads((RUNS / a / "predictions.json").read_text(encoding="utf-8"))
    pb = json.loads((RUNS / b / "predictions.json").read_text(encoding="utf-8"))
    diffs = []
    for vid in sorted(set(pa) | set(pb)):
        x, y = pa.get(vid), pb.get(vid)
        if x is None or y is None:
            diffs.append(f"{vid}: есть только в одном прогоне")
            continue
        key = lambda r: [(v["type_id"], v["timecode_sec"], v["description"]) for v in r["violations"]]  # noqa: E731
        if x["verdict"] != y["verdict"] or key(x) != key(y):
            diffs.append(f"{vid}: {x['verdict']} {key(x)}  ≠  {y['verdict']} {key(y)}")
    print("✔ прогоны совпадают полностью" if not diffs else "✖ расхождения:\n" + "\n".join(diffs))
    return 1 if diffs else 0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--only", default="")
    p.add_argument("--cache", action="store_true", help="reuse detections/VLM answers (tuning only, not for metrics)")
    p.add_argument("--compare", nargs=2, metavar=("RUN1", "RUN2"))
    a = p.parse_args()
    if a.compare:
        raise SystemExit(compare(*a.compare))
    if a.cache:
        os.environ["DT_CACHE"] = str(ROOT / "runs" / "cache")
    run([v.strip() for v in a.only.split(",") if v.strip()] or None)


if __name__ == "__main__":
    main()
