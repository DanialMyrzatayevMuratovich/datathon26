"""Frames from a video at exact timestamps. Deterministic: the same video and times always give the same pixels."""

import io
import json
import math
import struct
import subprocess
import tempfile
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from vision.limits import FFMPEG_THREADS


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
    r = subprocess.run(["ffmpeg", "-v", "error", "-threads", FFMPEG_THREADS, "-ss", f"{t:.3f}",
                        "-i", str(path), "-frames:v", "1", "-f", "image2", "-c:v", "png",
                        "-threads", FFMPEG_THREADS, "pipe:1"], capture_output=True, check=True)
    return Image.open(io.BytesIO(r.stdout)).convert("RGB")


class SampledFrames(Sequence):
    """Lossless PNGs in an anonymous temporary file; only four decoded frames stay in RAM.

    Close explicitly (or use a context manager). The OS closes the scratch file even
    if a service worker is killed at its deadline; no bank frames remain on disk.
    Returned images are read-only by convention and retain the original resolution.
    """

    def __init__(self):
        self._file = tempfile.TemporaryFile()
        self._entries = []
        self._decoded = OrderedDict()
        self.timestamps = []

    def __len__(self):
        return len(self._entries)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if index not in self._decoded:
            offset, size = self._entries[index]
            self._file.seek(offset)
            with Image.open(io.BytesIO(self._file.read(size))) as image:
                self._decoded[index] = image.convert("RGB")
            if len(self._decoded) > 4:
                self._decoded.popitem(last=False)
        self._decoded.move_to_end(index)
        return self.timestamps[index], self._decoded[index]

    def close(self):
        self._decoded.clear()
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        if hasattr(self, "_file"):
            self._file.close()

    def images(self):
        return _Images(self)


class _Images(Sequence):
    """Lazy detector input: slicing loads just its current batch, never the clip."""

    def __init__(self, frames):
        self.frames = frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self.frames[i][1] for i in range(*index.indices(len(self)))]
        return self.frames[index][1]


def _read_exact(stream, size):
    parts = bytearray()
    while len(parts) < size:
        part = stream.read(size - len(parts))
        if not part:
            raise ValueError("Truncated ffmpeg PNG stream")
        parts.extend(part)
    return parts


def sample(path: str | Path, fps: float = 2.0, max_side: int | None = None) -> SampledFrames:
    """Every 1/fps seconds; same ffmpeg pixels, bounded decoded RAM, original detail.

    PNG chunks are parsed by length, not searched for a signature that might also
    occur inside compressed image data. Scratch storage contains encoded PNG only.
    """
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive and finite")
    if max_side is not None and max_side <= 0:
        raise ValueError("max_side must be positive")
    vi = info(path)
    vf = f"fps={fps}"
    if max_side:
        vf += f",scale='if(gt(iw,ih),min({max_side},iw),-2)':'if(gt(iw,ih),-2,min({max_side},ih))'"
    command = ["ffmpeg", "-v", "error", "-threads", FFMPEG_THREADS, "-i", str(path), "-vf", vf,
               "-f", "image2pipe", "-c:v", "png", "-threads", FFMPEG_THREADS, "pipe:1"]
    out = SampledFrames()
    try:
        with tempfile.TemporaryFile() as errors, subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=errors,
        ) as proc:
            try:
                number = 0
                while signature := proc.stdout.read(8):
                    if signature != b"\x89PNG\r\n\x1a\n":
                        raise ValueError("Invalid ffmpeg PNG stream")
                    start = out._file.tell()
                    out._file.write(signature)
                    while True:
                        header = _read_exact(proc.stdout, 8)
                        size = struct.unpack(">I", header[:4])[0]
                        if size > 64 * 1024 * 1024:
                            raise ValueError("Unexpectedly large PNG chunk")
                        out._file.write(header)
                        out._file.write(_read_exact(proc.stdout, size + 4))  # data and CRC
                        if header[4:] == b"IEND":
                            break
                    timestamp = round(number / fps, 3)
                    if timestamp <= vi.duration:
                        out._entries.append((start, out._file.tell() - start))
                        out.timestamps.append(timestamp)
                    number += 1
                if proc.wait():
                    raise RuntimeError("ffmpeg could not decode video")
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
        if not out:
            raise ValueError("Video yielded no sampled frames")
        out._file.flush()
        return out
    except BaseException:
        out.close()
        raise


def clipped_box(box, width: int, height: int):
    """Reject malformed detections rather than inventing a crop or failing the whole clip."""
    if len(box) != 4 or not all(math.isfinite(v) for v in box):
        return None
    x0, y0, x1, y1 = box
    x0, x1 = max(0, min(width, x0)), max(0, min(width, x1))
    y0, y1 = max(0, min(height, y0)), max(0, min(height, y1))
    if int(x1) <= int(x0) or int(y1) <= int(y0):
        return None
    return x0, y0, x1, y1


def crop(im: Image.Image, box: tuple[float, float, float, float], margin: float = 0.25,
         out_side: int = 448) -> Image.Image:
    """Box with a margin, clipped to the image, upscaled so the longer side is out_side (detail for the VLM)."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    x0, y0 = max(0, x0 - margin * w), max(0, y0 - margin * h)
    x1, y1 = min(im.width, x1 + margin * w), min(im.height, y1 + margin * h)
    clipped = clipped_box((x0, y0, x1, y1), im.width, im.height)
    if clipped is None:
        raise ValueError("Crop has no valid pixels")
    c = im.crop(tuple(int(v) for v in clipped))
    k = out_side / max(c.width, c.height)
    return c.resize((max(1, round(c.width * k)), max(1, round(c.height * k))), Image.LANCZOS)
