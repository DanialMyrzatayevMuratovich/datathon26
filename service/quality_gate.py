"""Compare saved candidates without mistaking a cached subset for a quality gain.

No videos/images/descriptions are emitted. Passing this gate only qualifies a
candidate for independent evaluation; it never certifies generalization or SLA.
"""

import argparse
import hashlib
import json
from pathlib import Path

from service.quality import audit, labels


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_run(directory, truth, label_digest):
    path = directory / "predictions.json"
    predictions = json.loads(path.read_text(encoding="utf-8"))
    checked = audit(truth, predictions)
    reasons = []
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        manifest = {}
    if not checked["coverage"]["complete"]:
        reasons.append("incomplete_prediction_coverage")
    if not manifest:
        reasons.append("missing_manifest")
    else:
        if manifest.get("status") != "complete" or manifest.get("engine") != "real":
            reasons.append("not_completed_real_run")
        if manifest.get("cache_enabled") is not False or manifest.get("subset") is not False:
            reasons.append("cached_or_subset_run")
        expected = set(truth)
        if any(set(manifest.get(key, [])) != expected for key in ("expected", "requested", "completed")):
            reasons.append("incomplete_manifest_coverage")
        if manifest.get("predictions_sha256") != digest(path):
            reasons.append("prediction_hash_mismatch")
        if manifest.get("labels_sha256") != label_digest:
            reasons.append("label_hash_mismatch")
        if set(manifest.get("video_sha256", {})) != expected:
            reasons.append("incomplete_video_hashes")
        models = manifest.get("models")
        if not isinstance(models, list) or not models or any(not m.get("files") for m in models):
            reasons.append("missing_model_content_provenance")
        if not manifest.get("source_sha256") or not manifest.get("runtime"):
            reasons.append("missing_source_or_runtime_provenance")
    return checked, manifest, reasons


def compare(baseline, candidate, truth, label_digest):
    before, bm, br = load_run(baseline, truth, label_digest)
    after, cm, cr = load_run(candidate, truth, label_digest)
    blockers = [f"baseline:{reason}" for reason in br] + [f"candidate:{reason}" for reason in cr]
    if bm and cm and bm.get("video_sha256") != cm.get("video_sha256"):
        blockers.append("video_hashes_differ")
    transitions = {}
    for vid in truth:
        a, b = before["per_video"][vid]["verdict_outcome"], after["per_video"][vid]["verdict_outcome"]
        # A missing row is missing evidence, not a model prediction.
        if a != b and "missing" not in (a, b):
            transitions.setdefault(f"{a}_to_{b}", []).append(vid)
    regressions = list(transitions.get("tp_to_fn", [])) + list(transitions.get("tn_to_fp", []))
    event_regressions = []
    for vid in truth:
        a, b = before["per_video"][vid], after["per_video"][vid]
        if "missing" in (a["verdict_outcome"], b["verdict_outcome"]):
            continue
        for tolerance in ("1", "2"):
            matched_a = {m["truth_index"] for m in a["matches"][tolerance]}
            matched_b = {m["truth_index"] for m in b["matches"][tolerance]}
            lost = sorted(matched_a - matched_b)
            extra = (len(b["predicted"]) - len(matched_b)) - (len(a["predicted"]) - len(matched_a))
            if lost or extra > 0:
                event_regressions.append({"video": vid, "tolerance_sec": int(tolerance),
                                          "lost_truth_indices": lost, "extra_unmatched_predictions": max(0, extra)})
    event_regression = bool(event_regressions)
    if regressions or event_regression:
        decision = "regression_requires_review"
    elif blockers:
        decision = "insufficient_evidence"
    else:
        decision = "eligible_for_independent_evaluation"
    summaries = {}
    for name, report in (("baseline", before), ("candidate", after)):
        summaries[name] = {key: report[key] for key in ("coverage", "binary", "joint_events")}
    return {
        "purpose": "Development comparison only; never automatic promotion or a jury-score guarantee",
        "decision": decision, "automatic_promotion": False,
        "blockers": blockers, "transitions_on_observed_videos": transitions,
        "regressed_videos": regressions, "joint_event_regression": event_regression,
        "event_regressions_on_observed_videos": event_regressions,
        "independent_evaluation": "not_assessed",
        "latency_and_memory": "not_assessed",
        "labels_sha256": label_digest,
        "runs": {"baseline": baseline.name, "candidate": candidate.name}, **summaries,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=Path("data/annotations.csv"))
    parser.add_argument("--out", type=Path, default=Path("runs/jobs/quality-gate.json"))
    args = parser.parse_args()
    report = compare(args.baseline, args.candidate, labels(args.labels), digest(args.labels))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Decision: {report['decision']}; automatic promotion: false")
    print("Blockers: " + ", ".join(report["blockers"]))
    print(f"Metadata report: {args.out}")


if __name__ == "__main__":
    main()
