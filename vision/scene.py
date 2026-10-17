"""Scene objects by text (Grounding DINO tiny, Apache-2.0): safe, laptop, door, desk, green bag, cash.

Positions come from detection on the actual frame, not from fixed pixels — the camera moves between videos and the
final test video is shot from another angle.
"""

import os
from functools import cache
from pathlib import Path

import torch
from PIL import Image

from vision import limits

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.getenv("MODELS_DIR", ROOT.parent / "models"))
GDINO = MODELS_DIR / "grounding-dino-tiny"

# key -> (phrase for the prompt, words that identify the returned label)
OBJECTS = {
    "safe": ("a grey metal safe", ("safe", "grey metal", "metal")),
    "laptop": ("a laptop", ("laptop",)),
    "door": ("a glass door", ("door",)),
    "desk": ("a wooden desk", ("desk", "wooden")),
    "bag": ("a green bag", ("bag",)),
    "cash": ("a stack of banknotes", ("banknote", "stack", "money")),
    "machine": ("a banknote counting machine", ("machine", "counting")),
    "table": ("a table", ("table",)),
    "monitor": ("a computer monitor", ("monitor", "computer")),
    "cabinet": ("a metal cabinet", ("cabinet", "metal")),
}


@cache
def _model():
    limits.apply_torch()
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    if not GDINO.exists():
        raise FileNotFoundError(f"нет весов {GDINO} — scripts/download_models.sh")
    return AutoProcessor.from_pretrained(GDINO), AutoModelForZeroShotObjectDetection.from_pretrained(GDINO).eval()


def _key_of(label: str, keys: list[str]) -> str | None:
    label = label.lower()
    for k in keys:
        if any(word in label for word in OBJECTS[k][1]):
            return k
    return None


def detect(im: Image.Image, keys: list[str], threshold: float = 0.3) -> dict[str, list[tuple[list[float], float]]]:
    """key -> [(box xyxy, score)] sorted by score."""
    proc, model = _model()
    text = " ".join(OBJECTS[k][0] + "." for k in keys)
    inputs = proc(images=im, text=text, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)
    res = proc.post_process_grounded_object_detection(
        out, inputs.input_ids, threshold=threshold, text_threshold=0.25, target_sizes=[im.size[::-1]])[0]
    found: dict[str, list] = {k: [] for k in keys}
    # text_labels can be longer than boxes in transformers 5 for single-phrase prompts: pair by position
    for box, score, label in zip(res["boxes"].tolist(), res["scores"].tolist(), res["text_labels"], strict=False):
        k = _key_of(label, keys)
        if k:
            found[k].append(([round(v, 1) for v in box], round(float(score), 3)))
    return {k: sorted(v, key=lambda x: -x[1]) for k, v in found.items()}


def best(found: dict, key: str, min_score: float = 0.3) -> list[float] | None:
    items = [b for b, s in found.get(key, []) if s >= min_score]
    return items[0] if items else None
