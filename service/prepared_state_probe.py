"""Two-phase private state screen: release scene-detector RAM before starting the VLM.

Prepared PNGs contain confidential image crops: keep the directory under ignored
runs/jobs, never upload or commit it. This is not a violation/accuracy evaluation.
"""

import argparse
import hashlib
import io
import json
import os
import re
import sys
import time
from pathlib import Path

from PIL import Image

from service.local_state_probe import FPS, QUESTIONS, WINDOWS, prepare_frames, state_code
from service.local_vlm_probe import MODEL, PREPARED_MODELS, ROOT, LocalVLM, keep_awake
from service.worker import write_json


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def jpeg_digest(image):
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=92)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def prepare(directory, videos, ids, mode):
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "preparing", "prepared": False, "complete": False, "mode": mode,
                "question_sha256": hashlib.sha256(QUESTIONS[mode].encode()).hexdigest(), "videos": []}
    output = directory / "manifest.json"
    write_json(output, manifest)
    try:
        from vision import engine, limits
        from vision.frames import sample

        for identifier in ids:
            paths = [p for p in videos.iterdir() if p.is_file() and p.stem.replace(chr(0x425), "X") == identifier
                     and p.suffix.lower() in {".mov", ".mp4"}]
            if len(paths) != 1:
                raise ValueError("Expected one local video for each ID")
            with sample(paths[0], fps=FPS) as frames:
                metadata, triples = prepare_frames(frames, mode=mode, find_scene=engine.find_scene,
                                                   wait_if_hot=limits.wait_if_hot)
            row = {"id": identifier, "video_sha256": digest(paths[0]), "metadata": metadata, "windows": []}
            manifest["videos"].append(row)
            if not metadata["prepared"]:
                raise ValueError("Scene preparation incomplete")
            for window, images in zip(metadata["windows"], triples, strict=True):
                saved = {"sample_timestamps_sec": window["sample_timestamps_sec"], "images": []}
                row["windows"].append(saved)
                for image in images:
                    name = f"{len(manifest['videos']):03d}-{len(row['windows']):02d}-{len(saved['images'])}.png"
                    path = directory / name
                    image.save(path, format="PNG")
                    saved["images"].append({"file": name, "png_sha256": digest(path),
                                            "jpeg_sha256": jpeg_digest(image)})
            write_json(output, manifest)
        manifest.update(status="prepared", prepared=True)
    except Exception as exc:
        manifest.update(status="failed", error_type=type(exc).__name__)
    write_json(output, manifest)
    (directory / "manifest.sha256").write_text(digest(output) + "\n", encoding="ascii")
    return manifest


def load_inputs(directory):
    """Validate the entire replay before any request; loading imports no detector stack."""
    path = directory / "manifest.json"
    if digest(path) != (directory / "manifest.sha256").read_text("ascii").strip():
        raise ValueError("Manifest hash mismatch")
    manifest = json.loads(path.read_text("utf-8"))
    mode = manifest["mode"]
    if mode not in QUESTIONS or not manifest.get("prepared") or manifest.get("complete"):
        raise ValueError("Invalid prepared manifest")
    if manifest["question_sha256"] != hashlib.sha256(QUESTIONS[mode].encode()).hexdigest():
        raise ValueError("Question hash mismatch")
    if not manifest["videos"]:
        raise ValueError("No prepared inputs")
    prepared = []
    for video in manifest["videos"]:
        if len(video["windows"]) != WINDOWS:
            raise ValueError("Incomplete window set")
        for window in video["windows"]:
            if len(window["images"]) != 3:
                raise ValueError("Incomplete image triple")
            images = []
            for item in window["images"]:
                path = (directory / item["file"]).resolve()
                if path.parent != directory.resolve() or path.suffix != ".png":
                    raise ValueError("Input path escapes prepared directory")
                if digest(path) != item["png_sha256"]:
                    raise ValueError("PNG hash mismatch")
                with Image.open(path) as image:
                    rgb = image.convert("RGB")
                if jpeg_digest(rgb) != item["jpeg_sha256"]:
                    raise ValueError("Replay JPEG hash mismatch")
                images.append(rgb)
            prepared.append((video["id"], window, images))
    return manifest, prepared


def infer(directory, output, client):
    report = {"status": "running", "complete": False, "windows": [], "actual_vlm_calls": 0,
              "prediction_cache_enabled": False, "model": client.model}
    write_json(output, report)
    began = time.perf_counter()
    try:
        if (ROOT / "runs/.hot").exists():
            raise RuntimeError("Thermal flag present")
        manifest, prepared = load_inputs(directory)
        report.update(mode=manifest["mode"], manifest_sha256=digest(directory / "manifest.json"),
                      question_sha256=manifest["question_sha256"])
        for identifier, window, images in prepared:
            if (ROOT / "runs/.hot").exists():
                raise RuntimeError("Thermal flag present")
            answer, seconds = client.ask(images, QUESTIONS[manifest["mode"]], max_tokens=4)
            report["actual_vlm_calls"] += 1
            trace = client.inputs[-1]
            expected = [item["jpeg_sha256"] for item in window["images"]]
            if trace["jpeg_sha256"] != expected or trace["question_sha256"] != manifest["question_sha256"]:
                raise ValueError("Actual request hash mismatch")
            report["windows"].append({"id": identifier, "sample_timestamps_sec": window["sample_timestamps_sec"],
                                      "state": state_code(answer), "seconds": seconds, "jpeg_sha256": expected})
            write_json(output, report)
        report["complete"] = all(row["state"] != "unknown" for row in report["windows"])
        report["status"] = "complete" if report["complete"] else "incomplete"
    except Exception as exc:
        report.update(status="failed", complete=False, error_type=type(exc).__name__)
    report["seconds"] = round(time.perf_counter() - began, 3)
    write_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "infer"))
    parser.add_argument("--mode", choices=QUESTIONS)
    parser.add_argument("--videos", type=Path)
    parser.add_argument("--only")
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", choices=PREPARED_MODELS, default=MODEL)
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--child-report", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = args.out.resolve()
    if not output.is_relative_to((ROOT / "runs/jobs").resolve()):
        parser.error("Keep private artifacts under runs/jobs")
    if args.stage == "prepare":
        if not args.mode or not args.videos or not args.videos.is_dir() or not args.only:
            parser.error("prepare needs --mode, --videos, --only")
        ids = list(dict.fromkeys(args.only.split(",")))
        if any(not re.fullmatch(r"[A-Za-z0-9_-]+", item) for item in ids) or output.exists():
            parser.error("Need valid IDs and a new output directory")
    elif not args.inputs or not args.inputs.resolve().is_relative_to((ROOT / "runs/jobs").resolve()):
        parser.error("infer needs an input directory under runs/jobs")
    if args.child_report:
        os.environ.pop("DT_CACHE", None)
        os.environ.pop("DT_BATCH", None)
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        if args.stage == "prepare":
            result = prepare(output, args.videos.resolve(), ids, args.mode)
            success = result["prepared"]
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            client = LocalVLM(args.port, model=args.model, budget_sec=600, action_codes=True,
                              trace_path=output.with_suffix(".inputs.jsonl"))
            if os.getenv("PROFILE_DEADLINE"):
                client.deadline = min(client.deadline, float(os.environ["PROFILE_DEADLINE"]))
            result = infer(args.inputs.resolve(), output, client)
            success = result["complete"]
        write_json(args.child_report, {"status": "complete" if success else "failed",
                                       "stage": args.stage, "predictions_complete": args.stage == "infer" and success})
        raise SystemExit(0 if success else 1)
    from service.inference_profile import supervise

    command = [sys.executable, "-m", "service.prepared_state_probe", *sys.argv[1:]]
    with keep_awake():
        runtime = supervise(lambda result: [*command, "--child-report", str(result)],
                            output.with_suffix(".runtime.json"), 600, os.environ)
    target = output / "manifest.json" if args.stage == "prepare" else output
    if runtime["status"] != "complete":
        result = json.loads(target.read_text("utf-8")) if target.exists() else {"windows": []}
        result.update(status=runtime["status"], complete=False)
        if args.stage == "prepare":
            result["prepared"] = False
        target.parent.mkdir(parents=True, exist_ok=True)
        write_json(target, result)
        if args.stage == "prepare":
            (output / "manifest.sha256").write_text(digest(target) + "\n", encoding="ascii")
    print(f"{args.stage}: {runtime['status']}; private metadata: {target}")
    raise SystemExit(0 if runtime["status"] == "complete" else 1)


if __name__ == "__main__":
    main()
