"""Evaluation and the double-run check (docs/ML_SPEC.md §5, §9).

  uv run python -m vision.evaluate                       # all 40 unique videos -> runs/eval/<run_id>/
  uv run python -m vision.evaluate --only X-28,X-26      # subset for quick iterations
  uv run python -m vision.evaluate --compare <run1> <run2>   # second run must match the first exactly
  uv run python -m vision.evaluate --cache --only X-28    # rule tuning: detections/VLM answers from runs/cache

Duplicates (X-47 = X-4, X-48 = X-5) are one video; X-4 then has two labelled events.
"""

import argparse
import hashlib
import json
import os
import platform
import secrets
import subprocess
import time
from datetime import datetime
from pathlib import Path

from service.quality import audit, labels, video_id
from vision.devcache import _backend_versions, model_content_identity
from vision.labels import DUPLICATES  # noqa: F401 - historical experiment imports

ROOT = Path(__file__).resolve().parent.parent
VIDEOS = ROOT / "data" / "raw" / "open" / "videos"
LABELS = ROOT / "data" / "annotations.csv"
RUNS = ROOT / "runs" / "eval"
DT_OK = 2.0


def video_path(vid: str) -> Path:
    for p in VIDEOS.glob("*.mov"):
        if p.stem.replace("Х", "X") == vid:
            return p
    raise FileNotFoundError(vid)


def ground_truth() -> dict[str, dict]:
    return labels(LABELS)


def metrics(gt: dict, pred: dict) -> dict:
    checked = audit(gt, pred)
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
        "accuracy": checked["binary"]["accuracy"],
        "accuracy_observed_only": checked["binary"]["accuracy_observed_only"],
        "coverage": checked["coverage"], "joint_events": checked["joint_events"],
        "event_hit_note": "type_hit/time_hit_2s are separate legacy diagnostics; use joint_events for event quality",
        "type_hit": f"{ev_type}/{ev_total}", "time_hit_2s": f"{ev_time}/{ev_total}",
        "max_seconds": max((v["seconds"] for v in per_video.values()), default=0),
        "per_video": per_video,
    }


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_identity() -> dict:
    paths = [*sorted((ROOT / "vision").rglob("*.py")), ROOT / "service/quality.py",
             ROOT / "uv.lock", ROOT / "scripts/download_models.sh", ROOT / "contract/schemas.py",
             ROOT / "data/laws/rules_catalog.json"]
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in paths if p.is_file()}


def runtime_identity() -> dict:
    import torch
    return {"python": platform.python_version(), "system": platform.system(), "machine": platform.machine(),
            "packages": _backend_versions(), "mps": torch.backends.mps.is_available(),
            "ffmpeg": subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0],
            "vlm_model": os.getenv("VLM_MODEL", "qwen3-vl-4b-instruct-mlx-4bit"),
            "threads": os.getenv("DT_THREADS", "2")}


def configure(cache: bool) -> None:
    # Explicit validation cannot silently inherit the demo engine or a tuning cache.
    os.environ["VISION_ENGINE"] = "real"
    if cache:
        os.environ["DT_CACHE"] = str(ROOT / "runs" / "cache")
    else:
        os.environ.pop("DT_CACHE", None)


def run(only: list[str] | None, *, cache: bool = False) -> Path:
    configure(cache)
    os.environ["DT_BATCH"] = "1"  # the engine pauses mid-video when runs/.hot appears
    from vision import analyze

    gt = ground_truth()
    vids = [video_id(v) for v in only] if only else sorted(gt, key=lambda v: int(v.split("-")[1]))
    if len(set(vids)) != len(vids) or set(vids) - set(gt):
        raise ValueError("Subset must contain unique, known video IDs")
    paths = {vid: video_path(vid) for vid in vids}  # fail before loading any model
    run_id = f"{datetime.now():%Y%m%dT%H%M}-{secrets.token_hex(2)}"
    out = RUNS / run_id
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1, "status": "running", "engine": "real", "cache_enabled": cache,
        "subset": bool(only), "expected": sorted(gt), "requested": vids,
        "labels_sha256": sha256(LABELS), "source_sha256": source_identity(),
        "video_sha256": {vid: sha256(path) for vid, path in paths.items()},
        "runtime": runtime_identity(),
        "models": [model_content_identity(name) for name in (
            os.getenv("VLM_MODEL", "qwen3-vl-4b-instruct-mlx-4bit"), "rtdetr-v2-r50", "grounding-dino-tiny")],
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "timing_note": "Cached timings are not inference/SLA measurements" if cache else "Uncached local inference",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out / "labels.json").write_text(json.dumps(gt, ensure_ascii=False, indent=2), encoding="utf-8")
    pred = {}
    from vision.limits import COOLDOWN_S, wait_if_hot

    for n_done, vid in enumerate(vids):
        if n_done:
            time.sleep(COOLDOWN_S)  # one video at a time, with a cool-down pause
        wait_if_hot()
        t0 = time.perf_counter()
        a = analyze(paths[vid]).model_dump()
        if not a.get("engine", "").startswith("rules+"):
            raise ValueError("Evaluation requires the real rules/VLM engine")
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
    manifest.update(status="complete", completed=sorted(pred), predictions_sha256=sha256(out / "predictions.json"))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in m.items() if k != "per_video"}, ensure_ascii=False))
    print(f"→ {out.relative_to(ROOT)}")
    return out


def compare(a: str, b: str) -> int:
    try:
        ma, mb = [json.loads((RUNS / name / "manifest.json").read_text(encoding="utf-8")) for name in (a, b)]
        for name, manifest in ((a, ma), (b, mb)):
            if (manifest["status"] != "complete" or manifest["engine"] != "real" or manifest["cache_enabled"]
                    or manifest["subset"] or manifest["completed"] != manifest["expected"]
                    or sha256(RUNS / name / "predictions.json") != manifest["predictions_sha256"]):
                raise ValueError("Need complete, uncached real runs with intact predictions")
        for field in ("labels_sha256", "source_sha256", "video_sha256", "models", "runtime", "expected"):
            if ma[field] != mb[field]:
                raise ValueError(f"Run provenance differs: {field}")
    except (OSError, KeyError, ValueError) as exc:
        print(f"✖ нельзя подтвердить воспроизводимость: {exc}")
        return 2
    pa = json.loads((RUNS / a / "predictions.json").read_text(encoding="utf-8"))
    pb = json.loads((RUNS / b / "predictions.json").read_text(encoding="utf-8"))
    if set(pa) != set(ma["expected"]) or set(pb) != set(mb["expected"]):
        print("✖ неполное покрытие предсказаний")
        return 2
    diffs = []
    for vid in sorted(set(pa) | set(pb)):
        x, y = pa.get(vid), pb.get(vid)
        if x is None or y is None:
            diffs.append(f"{vid}: есть только в одном прогоне")
            continue
        key = lambda r: [(v["type_id"], v["timecode_sec"], v["description"]) for v in r["violations"]]  # noqa: E731
        if x["verdict"] != y["verdict"] or key(x) != key(y):
            diffs.append(f"{vid}: verdict, type, timestamp or description differs (content kept local)")
    print("✔ прогоны совпадают полностью" if not diffs else "✖ расхождения:\n" + "\n".join(diffs))
    return 1 if diffs else 0


def main() -> None:
    global VIDEOS, LABELS
    p = argparse.ArgumentParser()
    p.add_argument("--only", default="")
    p.add_argument("--cache", action="store_true", help="reuse detections/VLM answers (tuning only, not for metrics)")
    p.add_argument("--compare", nargs=2, metavar=("RUN1", "RUN2"))
    p.add_argument("--videos", type=Path, default=VIDEOS)
    p.add_argument("--labels", type=Path, default=LABELS)
    a = p.parse_args()
    if a.compare:
        raise SystemExit(compare(*a.compare))
    VIDEOS, LABELS = a.videos, a.labels
    run([v.strip() for v in a.only.split(",") if v.strip()] or None, cache=a.cache)


if __name__ == "__main__":
    main()
