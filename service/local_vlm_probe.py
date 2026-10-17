"""Experimental, loopback-only GGUF VLM adapter for the existing T9 probe.

Does not alter production's MLX backend. Start a verified, offline llama-server
on 127.0.0.1 first. Reports live under ignored runs/jobs; images stay in memory.
"""

import argparse
import base64
import hashlib
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
MODEL = "gemma-4-e2b-it-qat-q4_0"
EXPECTED_WEIGHTS = {
    "gemma-4-E2B_q4_0-it.gguf": "fa401b55b07ee70a54c6dae3903c783a6e65064312529ea57175cb5f8dec6634",
    "gemma-4-E2B-it-mmproj.gguf": "021059cce659fe7f9170d5599761d7bbaf644b798dab9503aca30dc43e6beb14",
}
PREPARED_MODELS = {
    MODEL: EXPECTED_WEIGHTS,
    "gemma-4-e4b-it-qat-q4_0": {
        "gemma-4-E4B_q4_0-it.gguf": "676c35070db6dbe52f93e9c864ee0fba4eddea94b9c875d9cb10daff453fbaee",
        "gemma-4-E4B-it-mmproj.gguf": "7498a37cb619e55f2fcf87eb931f56e99389ed6d432e4c5c66110694c0d65578",
    },
}


@contextmanager
def keep_awake():
    """Hold an idle-sleep request only for this Windows experiment, not a power-plan edit."""
    if os.name != "nt":
        yield
        return
    import ctypes

    execution_state = ctypes.windll.kernel32.SetThreadExecutionState
    execution_state.argtypes = [ctypes.c_uint]
    execution_state.restype = ctypes.c_uint
    previous = execution_state(0x80000003)  # continuous + system + display; no away mode
    if not previous:
        raise OSError("Could not acquire the temporary Windows awake request")
    try:
        yield
    finally:
        execution_state(previous | 0x80000000)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        raise urllib.error.HTTPError(request.full_url, code, "Local VLM redirects are forbidden", headers, None)


class LocalVLM:
    def __init__(self, port=8091, timeout=60, model=MODEL, budget_sec=1200, action_codes=False, trace_path=None):
        if not 1024 <= port <= 65535 or not 0 < timeout <= 120 or not 0 < budget_sec <= 1800:
            raise ValueError("Invalid local VLM port or time budget")
        self.url = f"http://127.0.0.1:{port}/v1/chat/completions"
        self.timeout, self.model = timeout, model
        self.deadline = time.monotonic() + budget_sec
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        self.calls = 0
        self.seconds = 0.0
        self.completion_tokens = 0
        self.action_codes = action_codes
        self.inputs = []
        self.trace_path = trace_path
        if self.trace_path:
            self.trace_path.write_text("", encoding="utf-8")

    def ask(self, images, question, *, max_tokens=64, model_name=None):
        wrong_model = model_name is not None and model_name != self.model
        if wrong_model or not 1 <= len(images) <= 3 or not 1 <= max_tokens <= 128:
            raise ValueError("Unexpected model, image count or output budget")
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Local experiment budget exhausted")
        content = []
        image_hashes = []
        for image in images:
            stream = io.BytesIO()
            image.convert("RGB").save(stream, format="JPEG", quality=92)
            encoded = base64.b64encode(stream.getvalue()).decode("ascii")
            image_hashes.append(hashlib.sha256(stream.getvalue()).hexdigest())
            content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + encoded}})
        content.append({"type": "text", "text": question})
        payload = {"model": self.model, "messages": [{"role": "user", "content": content}],
                   "temperature": 0, "seed": 0, "max_tokens": max_tokens, "stream": False,
                   "cache_prompt": False, "chat_template_kwargs": {"enable_thinking": False}}
        if self.action_codes:
            # D remains an explicit uncertainty choice; this only constrains output syntax.
            payload["grammar"] = 'root ::= [ABCD]'
        request = urllib.request.Request(self.url, data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"}, method="POST")
        started = time.monotonic()
        try:
            with self.opener.open(request, timeout=min(self.timeout, remaining)) as response:
                body = response.read(1024 * 1024 + 1)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            # Do not forward server diagnostics that could include the private prompt.
            raise RuntimeError(f"Local VLM request failed ({type(exc).__name__})") from None
        elapsed = time.monotonic() - started
        if time.monotonic() > self.deadline:
            raise TimeoutError("Local experiment budget exhausted")
        if len(body) > 1024 * 1024:
            raise ValueError("Local VLM response too large")
        try:
            answer = json.loads(body)
            message = answer["choices"][0]["message"]["content"]
            tokens = answer.get("usage", {}).get("completion_tokens", 0)
            if not isinstance(message, str) or not isinstance(tokens, int):
                raise ValueError
        except (ValueError, KeyError, IndexError, TypeError):
            raise ValueError("Invalid local VLM response") from None
        self.calls += 1
        self.seconds += elapsed
        self.completion_tokens += tokens
        self.inputs.append({"jpeg_sha256": image_hashes,
                            "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
                            "max_tokens": max_tokens, "seconds": elapsed})
        if self.trace_path:
            with self.trace_path.open("a", encoding="utf-8") as trace:
                trace.write(json.dumps(self.inputs[-1]) + "\n")
        return message.strip(), elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", required=True, help="Experimental positive and negative video IDs")
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--model", choices=PREPARED_MODELS, default=MODEL)
    parser.add_argument("--every", type=float, default=4)
    parser.add_argument("--max-calls", type=int, default=40)
    parser.add_argument("--budget-sec", type=float, default=1200)
    parser.add_argument("--child-report", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = args.out.resolve()
    if not output.is_relative_to((ROOT / "runs/jobs").resolve()):
        parser.error("Write local reports under runs/jobs")
    if not args.videos.is_dir():
        parser.error("Local videos directory is missing")
    if (ROOT / "runs/.hot").exists():
        parser.error("Thermal flag present; wait for the host to cool")
    if not 0 < args.budget_sec <= 1800:
        parser.error("Experiment budget must be in (0, 1800]")
    if not args.child_report:
        from service.inference_profile import supervise
        from service.worker import write_json

        weights = Path(os.getenv("MODELS_DIR", ROOT.parent / "models")) / args.model
        for name, expected in PREPARED_MODELS[args.model].items():
            path = weights / name
            if not path.is_file():
                parser.error(f"Missing prepared weight file: {name}")
            with path.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                    parser.error(f"Prepared weight file hash mismatch: {name}")
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, {"status": "running", "videos": {}})
        command = [sys.executable, "-m", "service.local_vlm_probe", *sys.argv[1:]]
        with keep_awake():
            report = supervise(lambda result: [*command, "--child-report", str(result)],
                               output.with_suffix(".runtime.json"), args.budget_sec, os.environ)
        print(f"Experimental run: {report['status']}; local metadata: {output}")
        raise SystemExit(0 if report.get("status") == "complete" and report.get("all_videos_complete") else 1)
    child_report = args.child_report.resolve()
    if not child_report.is_relative_to((ROOT / "runs/jobs").resolve()):
        parser.error("Child report must stay under runs/jobs")
    os.environ.pop("DT_CACHE", None)
    os.environ.pop("DT_BATCH", None)
    os.environ.update(VLM_MODEL=args.model, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    from vision import vlm
    from vision.experiments import t9_action

    client = LocalVLM(args.port, model=args.model, budget_sec=args.budget_sec, action_codes=True,
                      trace_path=output.with_suffix(".inputs.jsonl"))
    if os.getenv("PROFILE_DEADLINE"):
        client.deadline = min(client.deadline, float(os.environ["PROFILE_DEADLINE"]))
    if vlm.DEFAULT_MODEL != args.model:
        parser.error("Run this experiment in a fresh process")
    # Only the experimental process redirects ask(); the default MLX implementation is untouched.
    argv = ["t9_action", "--only", args.only, "--videos", str(args.videos.resolve()),
            "--every", str(args.every), "--max-calls", str(args.max_calls), "--out", str(output)]
    started = time.monotonic()
    with patch.object(vlm, "ask", client.ask), patch.object(sys, "argv", argv):
        t9_action.main()
    result = json.loads(output.read_text("utf-8"))
    metadata = {"status": "complete", "requested_backend": "llama.cpp-b11715", "model": args.model,
                "expected_prepared_weights_sha256": PREPARED_MODELS[args.model],
                "server_identity_note": "Prepared files verified by parent; actual server launch verified separately",
                "prediction_cache_enabled": False,
                "all_videos_complete": bool(result["videos"]) and all(v["complete"] for v in result["videos"].values()),
                "wall_seconds": time.monotonic() - started, "requests": client.calls,
                "request_seconds": client.seconds, "completion_tokens": client.completion_tokens,
                "action_grammar": "A/B/C/D (D includes uncertainty)", "inputs": client.inputs,
                "note": "Experimental Gemma backend; not MLX/Qwen, not API SLA or full accuracy"}
    child_report.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
