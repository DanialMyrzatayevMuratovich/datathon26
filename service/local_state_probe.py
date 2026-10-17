"""Local visual-state screen; not a violation detector or an accuracy evaluation.

Run under an external process deadline with an already verified loopback server.
The CLI handles one clip; no labels, event annotations or video-name rules enter
the screen. Reports contain state codes and input hashes, never images/free text.
"""

import argparse
import hashlib
import math
import os
import time
from pathlib import Path

from service.local_vlm_probe import MODEL, PREPARED_MODELS, ROOT, LocalVLM, keep_awake
from service.worker import write_json

QUESTIONS = {
    "door": (
        "These are three chronological views of the same room door over two seconds. "
        "Describe only its visible state at the final frame, using the preceding frames for context. "
        "A: the door is visibly open. B: the door is visibly closed. "
        "C: the door is visibly moving between open and closed. "
        "D: the door or its state cannot be identified confidently. "
        "Do not infer its state from a person being present or absent. Answer only A, B, C or D."
    ),
    "key": (
        "These are three chronological views of the same safe over two seconds. "
        "Describe only the visible safe-lock state at the final frame. "
        "A: a key is clearly visible inserted in the lock. B: the lock is clearly visible and empty. "
        "C: a hand obscures the lock. D: the lock or presence of a key cannot be identified confidently. "
        "Do not infer a key from a handle, hinge, reflection, or an open safe door. Answer only A, B, C or D."
    ),
}
FPS = 2.0
WINDOWS = 5


def state_code(text):
    """Malformed outputs remain unknown; never coerce them into a negative state."""
    code = text.strip().upper() if isinstance(text, str) else ""
    return code if code in ("A", "B", "C", "D") else "unknown"


def uniform_windows(count, fps=FPS):
    """At most five unique chronological triples, each spanning two seconds."""
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    middle, end = round(fps), round(2 * fps)
    if middle < 1 or end <= middle or count <= end:
        return []
    last = count - end - 1
    starts = sorted({round(i * last / (WINDOWS - 1)) for i in range(WINDOWS)})
    return [(start, start + middle, start + end) for start in starts]


def prepare_frames(frames, *, mode, find_scene, wait_if_hot):
    """Return metadata and PIL triples without VLM inference or predicted answers.

    Run scene detection while the VLM server is stopped, then retain these small
    crops locally after releasing the detector process. A prepared report is not
    a completed screen: every window still requires an actual model response.
    """
    from vision.frames import clipped_box, crop

    if mode not in QUESTIONS:
        raise ValueError("Unknown visual-state mode")
    began = time.perf_counter()
    report = {"complete": False, "prepared": False, "mode": mode, "windows": [], "actual_vlm_calls": 0,
              "roi": None, "scene_seconds": 0.0, "request_seconds": 0.0}
    triples = []

    def finish(reason):
        report["reason"] = reason
        report["seconds"] = round(time.perf_counter() - began, 3)
        return report, triples if report["prepared"] else []

    windows = uniform_windows(len(frames))
    if len(windows) != WINDOWS:
        return finish("insufficient_distinct_windows")
    # Check timestamps before calling detectors: repeated stills are not temporal evidence.
    for indices in windows:
        times = [frames[i][0] for i in indices]
        if not all(math.isfinite(t) for t in times) or not times[0] < times[1] < times[2]:
            return finish("invalid_sample_timestamps")
        if not math.isclose(times[2] - times[0], 2.0, abs_tol=0.002):
            return finish("invalid_window_span")
    wait_if_hot()
    scene_started = time.perf_counter()
    scene = find_scene(frames)
    report["scene_seconds"] = round(time.perf_counter() - scene_started, 3)
    box = getattr(scene, "door" if mode == "door" else "safe", None)
    if box is None:
        return finish("roi_not_found")
    first_image = frames[0][1]
    box = clipped_box(box, first_image.width, first_image.height)
    if box is None:
        return finish("invalid_roi")
    margin = 0.4 if mode == "door" else 0.25
    x0, y0, x1, y1 = box
    width, height = x1 - x0, y1 - y0
    expanded = clipped_box((x0 - margin * width, y0 - margin * height,
                            x1 + margin * width, y1 + margin * height),
                           first_image.width, first_image.height)
    if expanded is None:
        return finish("invalid_roi")
    roi = [int(value) for value in expanded]
    if roi[2] <= roi[0] or roi[3] <= roi[1]:
        return finish("invalid_roi")
    report["roi"] = {"box_pixels": roi, "source_box_pixels": list(box), "margin": margin,
                     "frame_size": list(first_image.size), "output_long_side": 640}
    for indices in windows:
        wait_if_hot()
        selected = [frames[i] for i in indices]
        if any(image.size != first_image.size for _, image in selected):
            return finish("frame_dimensions_changed")
        triples.append([crop(image, roi, margin=0, out_side=640) for _, image in selected])
        report["windows"].append({"sample_timestamps_sec": [timestamp for timestamp, _ in selected]})
    report["prepared"] = True
    return finish("prepared")


def probe_frames(frames, *, mode, client, find_scene, wait_if_hot):
    """Prepare crops, then perform actual inference; no fabricated answers in preparation."""
    began = time.perf_counter()
    report, triples = prepare_frames(frames, mode=mode, find_scene=find_scene, wait_if_hot=wait_if_hot)
    report["preparation_seconds"] = report["seconds"]
    if not report["prepared"]:
        return report
    for row, images in zip(report["windows"], triples, strict=True):
        try:
            wait_if_hot()
            answer, seconds = client.ask(images, QUESTIONS[mode], max_tokens=4)
            report["actual_vlm_calls"] += 1
            report["request_seconds"] += seconds
            trace = client.inputs[-1]
            row.update(state=state_code(answer), seconds=seconds,
                       jpeg_sha256=list(trace["jpeg_sha256"]), question_sha256=trace["question_sha256"])
        except Exception as exc:
            # Preserve completed windows, but never export private prompts/server diagnostics.
            report.update(complete=False, reason="request_failed", error_type=type(exc).__name__,
                          seconds=round(time.perf_counter() - began, 3))
            return report
    report["complete"] = (report["actual_vlm_calls"] == WINDOWS
                          and all(row["state"] != "unknown" for row in report["windows"]))
    report["reason"] = "complete" if report["complete"] else "invalid_model_output"
    report["seconds"] = round(time.perf_counter() - began, 3)
    return report


def probe(path, *, mode, client):
    from vision import engine, limits
    from vision.frames import sample

    began = time.perf_counter()
    limits.wait_if_hot()
    decode_started = time.perf_counter()
    with sample(path, fps=FPS) as frames:
        decode_seconds = time.perf_counter() - decode_started
        report = probe_frames(frames, mode=mode, client=client, find_scene=engine.find_scene,
                              wait_if_hot=limits.wait_if_hot)
    report["analysis_seconds"] = report["seconds"]
    report["decode_seconds"] = round(decode_seconds, 3)
    report["seconds"] = round(time.perf_counter() - began, 3)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=QUESTIONS, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--model", choices=PREPARED_MODELS, default=MODEL)
    parser.add_argument("--budget-sec", type=float, default=300)
    args = parser.parse_args()
    output, video = args.out.resolve(), args.video.resolve()
    if not output.is_relative_to((ROOT / "runs/jobs").resolve()):
        parser.error("Write local reports under runs/jobs")
    if not video.is_file():
        parser.error("Local video file is missing")
    if not 0 < args.budget_sec <= 1800:
        parser.error("Request budget must be in (0, 1800]; use an outer process deadline too")
    os.environ.pop("DT_CACHE", None)
    os.environ.pop("DT_BATCH", None)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "complete": False, "mode": args.mode, "model": args.model,
              "windows": [], "actual_vlm_calls": 0, "prediction_cache_enabled": False,
              "purpose": "Visual states only; not T3/T6 violations, accuracy or API SLA",
              "requested_backend": "llama.cpp; server identity must be verified by the launcher"}
    write_json(output, report)
    began = time.perf_counter()
    try:
        client = LocalVLM(args.port, model=args.model, budget_sec=args.budget_sec, action_codes=True,
                          trace_path=output.with_suffix(".inputs.jsonl"))
        if os.getenv("PROFILE_DEADLINE"):
            client.deadline = min(client.deadline, float(os.environ["PROFILE_DEADLINE"]))
        with video.open("rb") as stream:
            report["video_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
        with keep_awake():
            report.update(probe(video, mode=args.mode, client=client))
    except Exception as exc:
        report.update(complete=False, reason="execution_failed", error_type=type(exc).__name__)
    report["status"] = "failed" if "error_type" in report else "complete" if report["complete"] else "incomplete"
    report["wall_seconds"] = round(time.perf_counter() - began, 3)
    write_json(output, report)
    print(f"Visual-state screen: {report['status']}; calls={report['actual_vlm_calls']}; report={output}")
    raise SystemExit(0 if report["complete"] else 1)


if __name__ == "__main__":
    main()
