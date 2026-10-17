"""Local VLM expert: Qwen3-VL via MLX on Apple Silicon (docs/ML_SPEC.md §7).

Greedy decoding (temperature 0) so two runs give the same answers. Images go in as JPEG files; the model never sees
anything but the frames/crops we pass. Weights are read from MODELS_DIR (default: ../models next to the repo).
"""

import os
import re
import tempfile
import time
from functools import cache
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.getenv("MODELS_DIR", ROOT.parent / "models"))
DEFAULT_MODEL = os.getenv("VLM_MODEL", "qwen3-vl-4b-instruct-mlx-4bit")
YES = re.compile(r"^\W*(yes|да)\b", re.I)
NO = re.compile(r"^\W*(no|нет)\b", re.I)


@cache
def load(name: str = DEFAULT_MODEL):
    from mlx_vlm import load as mlx_load
    from mlx_vlm.utils import load_config

    path = MODELS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"нет весов {path} — scripts/download_models.sh")
    model, processor = mlx_load(str(path))
    return model, processor, load_config(str(path))


def ask(images: list[Image.Image], question: str, *, max_tokens: int = 64,
        model_name: str = DEFAULT_MODEL) -> tuple[str, float]:
    """(answer text, seconds). Greedy, deterministic."""
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    model, processor, config = load(model_name)
    with tempfile.TemporaryDirectory() as td:
        paths = []
        for i, im in enumerate(images):
            p = Path(td) / f"{i}.jpg"
            im.convert("RGB").save(p, quality=92)
            paths.append(str(p))
        prompt = apply_chat_template(processor, config, question, num_images=len(paths))
        t = time.perf_counter()
        out = generate(model, processor, prompt, image=paths, max_tokens=max_tokens, temperature=0.0,
                       verbose=False)
        dt = time.perf_counter() - t
    text = getattr(out, "text", out)
    return str(text).strip(), dt


def yes_no(images: list[Image.Image], question: str, **kw) -> tuple[bool | None, str, float]:
    """Ask a yes/no question; None when the answer is neither."""
    text, dt = ask(images, question + " Answer with one word: yes or no.", max_tokens=8, **kw)
    if YES.match(text):
        return True, text, dt
    if NO.match(text):
        return False, text, dt
    return None, text, dt
