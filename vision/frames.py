"""Frames from a video at exact timestamps. Deterministic: the same video and times always give the same pixels."""

import io
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass(frozen=True)
class VideoInfo:
    duration: float
    fps: float
    width: int
    height: int


def info(path: str | Path) -> VideoInfo:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate:format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True)
    j = json.loads(r.stdout)
    s = j["streams"][0]
    num, den = (int(x) for x in s["r_frame_rate"].split("/"))
    return VideoInfo(duration=float(j["format"]["duration"]), fps=num / den if den else 30.0,
                     width=int(s["width"]), height=int(s["height"]))


def frame_at(path: str | Path, t: float) -> Image.Image:
    """Exact frame at second t (ffmpeg seeks accurately when -ss goes before -i on modern ffmpeg)."""
    r = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(path), "-frames:v", "1",
                        "-f", "image2", "-c:v", "png", "pipe:1"], capture_output=True, check=True)
    return Image.open(io.BytesIO(r.stdout)).convert("RGB")


def sample(path: str | Path, fps: float = 2.0, max_side: int | None = None) -> list[tuple[float, Image.Image]]:
    """Every 1/fps seconds from t=0, decoded in one ffmpeg pass (fast and deterministic)."""
    vi = info(path)
    vf = f"fps={fps}"
    if max_side:
        vf += f",scale='if(gt(iw,ih),min({max_side},iw),-2)':'if(gt(iw,ih),-2,min({max_side},ih))'"
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", vf, "-f", "image2pipe", "-c:v", "png",
                        "pipe:1"], capture_output=True, check=True)
    data, out, start = r.stdout, [], 0
    sig = b"\x89PNG\r\n\x1a\n"
    while True:  # split the PNG stream at signatures
        nxt = data.find(sig, start + 1)
        chunk = data[start:nxt if nxt != -1 else len(data)]
        if chunk:
            out.append(Image.open(io.BytesIO(chunk)).convert("RGB"))
        if nxt == -1:
            break
        start = nxt
    times = [round(i / fps, 3) for i in range(len(out))]
    return [(t, im) for t, im in zip(times, out, strict=True) if t <= vi.duration]


def crop(im: Image.Image, box: tuple[float, float, float, float], margin: float = 0.25,
         out_side: int = 448) -> Image.Image:
    """Box with a margin, clipped to the image, upscaled so the longer side is out_side (detail for the VLM)."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    x0, y0 = max(0, x0 - margin * w), max(0, y0 - margin * h)
    x1, y1 = min(im.width, x1 + margin * w), min(im.height, y1 + margin * h)
    c = im.crop((int(x0), int(y0), int(x1), int(y1)))
    k = out_side / max(c.width, c.height)
    return c.resize((max(1, round(c.width * k)), max(1, round(c.height * k))), Image.LANCZOS)
