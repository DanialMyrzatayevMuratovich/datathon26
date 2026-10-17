"""People and COCO objects on sampled frames (RT-DETRv2-R50, Apache-2.0) + who is inside the cash room.

The room has glass walls: people outside are visible through the glass. They are smaller and stand higher in the frame
(their floor is behind the wall), so an in-room person must be tall enough and stand low enough. Thresholds are relative
to the frame height, not pixels, so they survive another camera.
"""

import os
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import torch
from PIL import Image

from vision import limits
from vision.frames import clipped_box

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.getenv("MODELS_DIR", ROOT.parent / "models"))
RTDETR = MODELS_DIR / "rtdetr-v2-r50"
KEEP = {"person", "laptop", "cup", "bottle", "handbag", "backpack", "cell phone", "suitcase", "book"}


@dataclass(frozen=True)
class Det:
    label: str
    score: float
    box: tuple[float, float, float, float]  # x0 y0 x1 y1 in pixels

    @property
    def cx(self) -> float:
        return (self.box[0] + self.box[2]) / 2

    @property
    def bottom(self) -> float:
        return self.box[3]

    @property
    def height(self) -> float:
        return self.box[3] - self.box[1]


def device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


@cache
def _model():
    limits.apply_torch()
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    if not RTDETR.exists():
        raise FileNotFoundError(f"нет весов {RTDETR} — scripts/download_models.sh")
    proc = AutoImageProcessor.from_pretrained(RTDETR)
    model = AutoModelForObjectDetection.from_pretrained(RTDETR).eval().to(device())
    return proc, model


def detect(frames: list[Image.Image], threshold: float = 0.45, batch: int = 8) -> list[list[Det]]:
    proc, model = _model()
    out: list[list[Det]] = []
    for i in range(0, len(frames), batch):
        chunk = frames[i:i + batch]
        inputs = proc(images=chunk, return_tensors="pt").to(device())
        with torch.no_grad():
            res = model(**inputs)
        sizes = torch.tensor([(im.height, im.width) for im in chunk])
        for im, r in zip(chunk, proc.post_process_object_detection(res, target_sizes=sizes, threshold=threshold),
                         strict=True):
            dets = []
            for s, lab, b in zip(r["scores"].tolist(), r["labels"].tolist(), r["boxes"].tolist(), strict=True):
                name = model.config.id2label[int(lab)]
                box = clipped_box(tuple(round(v, 1) for v in b), im.width, im.height)
                if name in KEEP and box is not None:
                    dets.append(Det(name, round(float(s), 3), box))
            out.append(dets)
    return out


def in_room(d: Det, frame_h: int, min_height: float = 0.14, min_bottom: float = 0.36) -> bool:
    """Inside the cash room: not a small figure seen through the glass wall."""
    return d.label == "person" and d.height >= min_height * frame_h and d.bottom >= min_bottom * frame_h
