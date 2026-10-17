"""Audit saved predictions without loading models, videos, descriptions or evidence.

Supplementary diagnostics, not the official jury metric or a replacement for vision.evaluate.
One prediction can explain at most one labelled event of the same type within the tolerance.
"""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from vision.evaluate import DUPLICATES


def video_id(value: str) -> str:
    value = value.strip().replace("Х", "X")
    return DUPLICATES.get(value, value)


def timestamp(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Event time must be a number")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("Event time must be finite and nonnegative")
    return value


def labels(path: Path) -> dict:
    result = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            vid = video_id(row["video"])
            if row["label"] not in {"violation", "no_violation"}:
                raise ValueError("Unknown annotation label")
            entry = result.setdefault(vid, {"violation": False, "events": []})
            if row["label"] == "violation":
                entry["violation"] = True
                entry["events"].append({
                    "type_id": row["type_id"], "t": timestamp(float(row["timecode_sec"])),
                    "confidence": row.get("confidence", "unknown"),
                })
    if not result:
        raise ValueError("Annotations are empty")
    return result


def match_events(truth: list[dict], predicted: list[dict], tolerance: float) -> list[tuple[int, int]]:
    """Maximum-cardinality bipartite matching; never reuse a prediction or truth event."""
    edges = []
    for event in truth:
        candidates = [i for i, pred in enumerate(predicted)
                      if pred["type_id"] == event["type_id"]
                      and abs(pred["timecode_sec"] - event["t"]) <= tolerance]
        edges.append(sorted(candidates, key=lambda i: (abs(predicted[i]["timecode_sec"] - event["t"]), i)))
    owner: dict[int, int] = {}

    def assign(g: int, seen: set[int]) -> bool:
        for p in edges[g]:
            if p in seen:
                continue
            seen.add(p)
            if p not in owner or assign(owner[p], seen):
                owner[p] = g
                return True
        return False

    for g in range(len(truth)):
        assign(g, set())
    return sorted((g, p) for p, g in owner.items())


def ratio(n: int, d: int):
    return round(n / d, 6) if d else None


def audit(truth: dict, predictions: dict) -> dict:
    if not truth or not isinstance(predictions, dict):
        raise ValueError("Need nonempty annotations and an object of predictions")
    normalized = {}
    for key, row in predictions.items():
        vid = video_id(key)
        if vid in normalized:
            raise ValueError("Duplicate video predictions after alias normalization")
        verdict, events = row.get("verdict"), row.get("violations")
        if verdict not in {"violation", "no_violation"} or not isinstance(events, list):
            raise ValueError("Invalid verdict or event list")
        if (verdict == "violation") != bool(events):
            raise ValueError("Verdict disagrees with event list")
        cleaned = []
        for event in events:
            if not isinstance(event.get("type_id"), str) or not event["type_id"]:
                raise ValueError("Every event needs a type_id")
            cleaned.append({"type_id": event["type_id"], "timecode_sec": timestamp(event["timecode_sec"])})
        normalized[vid] = {"violation": verdict == "violation", "events": cleaned}
    missing = sorted(set(truth) - set(normalized))
    extra = sorted(set(normalized) - set(truth))
    counts = Counter({"tp": 0, "fp": 0, "fn": 0, "tn": 0})
    totals = {1: Counter(), 2: Counter()}
    per_type: dict[str, Counter] = {}
    per_video = {}
    for vid, expected in truth.items():
        actual = normalized.get(vid)
        events = actual["events"] if actual else []
        if actual is not None:
            kind = ("tp" if actual["violation"] else "fn") if expected["violation"] else (
                "fp" if actual["violation"] else "tn")
            counts[kind] += 1
        else:
            kind = "missing"
        for event in expected["events"]:
            per_type.setdefault(event["type_id"], Counter())["truth"] += 1
        for event in events:
            per_type.setdefault(event["type_id"], Counter())["predicted"] += 1
        row = {"verdict_outcome": kind, "truth": expected["events"], "predicted": events, "matches": {}}
        for tol in (1, 2):
            pairs = match_events(expected["events"], events, tol)
            row["matches"][str(tol)] = [{"truth_index": g, "prediction_index": p} for g, p in pairs]
            totals[tol].update({"tp": len(pairs), "fp": len(events) - len(pairs),
                                "fn": len(expected["events"]) - len(pairs)})
            for g, _p in pairs:
                per_type[expected["events"][g]["type_id"]][f"matched_{tol}s"] += 1
        per_video[vid] = row
    observed = sum(counts.values())
    correct = counts["tp"] + counts["tn"]
    complete = not missing and not extra
    joint = {}
    for tol, values in totals.items():
        joint[f"type_and_time_{tol}s"] = {
            **dict(values), "precision": ratio(values["tp"], values["tp"] + values["fp"]),
            "recall": ratio(values["tp"], values["tp"] + values["fn"]),
            "f1": ratio(2 * values["tp"], 2 * values["tp"] + values["fp"] + values["fn"]),
        }
    return {
        "purpose": "Supplementary diagnostics; not jury score or held-out validation",
        "coverage": {"complete": complete, "expected": len(truth), "observed": observed,
                     "missing": missing, "extra": extra},
        "binary": {**dict(counts), "accuracy": ratio(correct, len(truth)) if complete else None,
                   "accuracy_observed_only": ratio(correct, observed),
                   "recall": ratio(counts["tp"], counts["tp"] + counts["fn"]),
                   "precision": ratio(counts["tp"], counts["tp"] + counts["fp"])},
        "joint_events": joint,
        "per_type": {key: dict(value) for key, value in sorted(per_type.items())},
        "label_confidence": dict(Counter(e.get("confidence", "unknown")
                                         for g in truth.values() for e in g["events"])),
        "verdict_errors": [vid for vid, row in per_video.items() if row["verdict_outcome"] in {"fp", "fn", "missing"}],
        "per_video": per_video,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--labels", type=Path, default=Path("data/annotations.csv"))
    parser.add_argument("--report", type=Path, required=True,
                        help="JSON output: metadata only, no evidence or descriptions")
    args = parser.parse_args()
    try:
        report = audit(labels(args.labels), json.loads(args.predictions.read_text(encoding="utf-8-sig")))
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        parser.error(f"Invalid audit input ({type(exc).__name__})")
    report["input_sha256"] = {name: hashlib.sha256(path.read_bytes()).hexdigest()
                              for name, path in (("predictions", args.predictions), ("labels", args.labels))}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"per_video", "input_sha256"}},
                     ensure_ascii=False, indent=2))
    if not report["coverage"]["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
