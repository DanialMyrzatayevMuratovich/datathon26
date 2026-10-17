"""Analyst pass: the local VLM reviews key moments the way the clips were annotated by hand — a strip of frames in
time order (seconds in the corner) + a zoom on the small object, and a checklist with yes / no / unclear answers.

Moments come from the people layout (vision.engine.tracks): a client at the desk, the cashier entering an empty room,
the cashier leaving. Only a clear answer becomes a finding; "unclear" never does. Scenarios — docs/SCENARIO_GUIDE.md.
"""

import hashlib
import os
import re

from PIL import Image, ImageDraw

from vision import vlm

MODEL = os.getenv("DT_ANALYST_MODEL", "qwen3-vl-8b-instruct-mlx-4bit")
TILE_W = 300

Q_VISIT = ("These frames are in time order (seconds in the corner) from a camera above a bank cash desk while a client "
           "is served. Answer each question with yes, no or unclear.\n"
           "1) Does the client show or hand over an identity card or passport to the cashier at any point?\n"
           "2) Is there another visitor in the room besides the cashier and this client (a second client or an "
           "outsider standing or walking in)?\n"
           "3) Is the cash counted only partly — some banknotes taken from the counting machine before it finished, "
           "or part of the stack handed back without counting?\n"
           "Reply exactly in this form: 1=<answer>; 2=<answer>; 3=<answer>")
Q_ENTRY = ("These frames are in time order (seconds in the corner): a cashier enters the cash room through its door. "
           "After she has passed through, is the door closed behind her or left open? "
           "Reply with one word: closed, open or unclear.")
Q_KEY = ("Image 1 is a zoom on the safe right after the cashier has left the room; image 2 shows the room. Is a key "
         "left inserted in the lock of the safe door (a small key head sticking out of the keyhole)? "
         "Reply with one word: yes, no or unclear.")


def strip(frames, idx: list[int], cols: int = 4) -> Image.Image:
    """Frames idx side by side (time-stamped), like a contact sheet."""
    tiles = []
    for i in idx:
        t, im = frames[i]
        k = TILE_W / im.width
        tile = im.convert("RGB").resize((TILE_W, round(im.height * k)), Image.LANCZOS)
        ImageDraw.Draw(tile).text((6, 6), f"{t:.1f}s", fill=(255, 255, 0))
        tiles.append(tile)
    rows = (len(tiles) + cols - 1) // cols
    th = max(t.height for t in tiles)
    sheet = Image.new("RGB", (TILE_W * min(cols, len(tiles)), th * rows), "black")
    for n, tile in enumerate(tiles):
        sheet.paste(tile, ((n % cols) * TILE_W, (n // cols) * th))
    return sheet


def _key(images, q):
    h = hashlib.sha1(q.encode() + MODEL.encode())
    for im in images:
        h.update(im.tobytes())
    return ("analyst", h.hexdigest()[:20])


def ask(images, q, dc=None, max_tokens=40) -> str:
    def run():
        return vlm.ask(images, q, max_tokens=max_tokens, model_name=MODEL)[0]
    return (dc.get(_key(images, q), run) if dc else run()).strip().lower()


def parse_visit(text: str) -> dict:
    out = {}
    for n in ("1", "2", "3"):
        m = re.search(rf"{n}\s*[=:)]\s*(yes|no|unclear)", text)
        out[n] = m.group(1) if m else "unclear"
    return out
