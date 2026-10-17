"""Read-only deployment checks for the actual engine present in this checkout."""

import importlib.util
import json
import os
import platform
import shutil
import sys
from pathlib import Path

from service.config import ROOT, Settings


def real_engine_errors() -> list[str]:
    errors = []
    if sys.platform != "darwin" or platform.machine().lower() not in {"arm64", "aarch64"}:
        errors.append("Текущий vision/vlm.py требует macOS Apple Silicon; Linux/Windows поддерживают только mock.")
    if importlib.util.find_spec("mlx_vlm") is None:
        errors.append("Не установлен mlx-vlm для текущего real-движка.")
    models = Path(os.getenv("MODELS_DIR", ROOT.parent / "models")).expanduser().resolve()
    names = (os.getenv("VLM_MODEL", "qwen3-vl-4b-instruct-mlx-4bit"), "rtdetr-v2-r50", "grounding-dino-tiny")
    for name in names:
        directory = models / name
        weights = list(directory.glob("*.safetensors")) + list(directory.glob("pytorch_model*.bin"))
        if not (directory / "config.json").is_file() or not weights or any(p.stat().st_size == 0 for p in weights):
            errors.append(f"Нет полного локального набора весов: {name} (MODELS_DIR).")
        index = directory / "model.safetensors.index.json"
        if index.is_file():
            try:
                parts = set(json.loads(index.read_text("utf-8"))["weight_map"].values())
                if any(not (directory / part).is_file() for part in parts):
                    errors.append(f"Не хватает частей весов: {name}.")
            except (OSError, ValueError, KeyError, TypeError):
                errors.append(f"Повреждён индекс весов: {name}.")
    return errors


def inspect(settings: Settings, require_ui: bool = False) -> dict:
    errors = []
    try:
        settings.validate()
    except ValueError as exc:
        errors.append(str(exc))
    for executable in ("ffmpeg", "ffprobe"):
        if not shutil.which(executable):
            errors.append(f"Нет {executable} в PATH.")
    if importlib.util.find_spec("python_multipart") is None:
        errors.append("Нет python-multipart; выполните uv sync --frozen.")
    if require_ui and not (settings.ui_dir / "index.html").is_file():
        errors.append("Нет ui/index.html.")
    if settings.engine == "real":
        errors.extend(real_engine_errors())
    return {
        "ok": not errors, "engine": settings.engine, "platform": sys.platform,
        "architecture": platform.machine(), "api_key_configured": bool(settings.api_key),
        "ui_present": (settings.ui_dir / "index.html").is_file(), "errors": errors,
        "note": "Проверка окружения; скорость и качество модели подтверждаются отдельным реальным прогоном.",
    }
