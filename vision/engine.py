"""Real engine v1: detection + rules + VLM checks (docs/ML_SPEC.md §6, rules — docs/ANALYSIS_RULES.txt §9).

1. frames at 2 fps;  2. scene objects (desk, laptop, safe, door) by Grounding DINO on 3 frames;
3. people by RT-DETRv2 on every frame -> who is at the workplace / near the laptop;  4. events (leave, arrival,
two at the workplace, long-open safe ...) -> 5. Qwen3-VL yes/no on crops of the event frames -> violations.
Deterministic: fixed frame times, greedy VLM, fixed thresholds below.
"""

import base64
import io
import time
from dataclasses import dataclass, field

from PIL import Image

from contract.schemas import EvidenceFrame, VideoAnalysis, ViolationDetail
from vision import people, scene, vlm
from vision.frames import crop, info, sample
from vision.pipeline import describe_rule, rules_for

FPS = 2.0
LEAVE_MIN_S = 3.0  # nobody at the workplace for this long = the cashier left
ARRIVE_MIN_S = 2.0
TWO_AT_DESK_MIN_S = 2.0
SAFE_OPEN_MIN_S = 6.0
SAFE_CHECK_EVERY_S = 2.0
EAT_CHECK_EVERY_S = 3.0
BAG_SCORE = 0.55

Q_SCREEN = ("This is a close-up of a laptop. Does the laptop screen show an open working application (for example a "
            "spreadsheet, a table or a program window)? Answer no if the screen shows a lock screen, a wallpaper "
            "picture, or is dark or closed.")
Q_SAFE = "This is a close-up of a grey metal safe. Is the safe door open?"
Q_CASH = "Are there banknotes or a stack of paper money in this image?"
Q_EAT = "Is this person eating food or drinking from a cup or bottle right now?"
Q_ITEMS = "Is there a jacket, a coat or a handbag lying on this desk?"


@dataclass
class Scene:
    desk: list[float] | None = None
    laptop: list[float] | None = None
    safe: list[float] | None = None
    door: list[float] | None = None


@dataclass
class Event:
    type_id: str
    t: float
    evidence_t: list[float]
    seen: str  # what is seen, Russian, for the description
    confidence: float
    extra: dict = field(default_factory=dict)


def _grow(b, k_w, k_h=None):
    if b is None:
        return None
    k_h = k_w if k_h is None else k_h
    w, h = b[2] - b[0], b[3] - b[1]
    return [b[0] - k_w * w, b[1] - k_h * h, b[2] + k_w * w, b[3] + k_h * h]


def _union(*boxes):
    bs = [b for b in boxes if b]
    if not bs:
        return None
    return [min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs)]


def _ioa(person, zone) -> float:
    """Share of the person box inside the zone."""
    if zone is None:
        return 0.0
    x0, y0 = max(person[0], zone[0]), max(person[1], zone[1])
    x1, y1 = min(person[2], zone[2]), min(person[3], zone[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area = max(1e-6, (person[2] - person[0]) * (person[3] - person[1]))
    return inter / area


def _runs(flags: list[bool], value: bool, min_len: int) -> list[tuple[int, int]]:
    """[(start, end_exclusive)] of runs of `value` at least min_len long."""
    out, i = [], 0
    while i < len(flags):
        if flags[i] == value:
            j = i
            while j < len(flags) and flags[j] == value:
                j += 1
            if j - i >= min_len:
                out.append((i, j))
            i = j
        else:
            i += 1
    return out


def find_scene(frames: list[tuple[float, Image.Image]]) -> Scene:
    picks = [frames[0], frames[len(frames) // 2], frames[-1]]
    best: dict[str, tuple[list[float], float]] = {}
    for _, im in picks:
        found = scene.detect(im, ["desk", "laptop", "safe", "door"], threshold=0.25)
        for k, items in found.items():
            if items and (k not in best or items[0][1] > best[k][1]):
                best[k] = items[0]
    return Scene(**{k: v[0] for k, v in best.items()})


def _jpeg_b64(im: Image.Image) -> str:
    buf = io.BytesIO()
    im.convert("RGB").save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _tiles(im: Image.Image, box, nx=3, ny=2, overlap=0.35) -> list[Image.Image]:
    c = crop(im, box, margin=0.05, out_side=900)
    W, H = c.size
    tw, th = W / (nx - (nx - 1) * overlap), H / (ny - (ny - 1) * overlap)
    out = []
    for j in range(ny):
        for i in range(nx):
            x0, y0 = i * tw * (1 - overlap), j * th * (1 - overlap)
            t = c.crop((int(x0), int(y0), int(min(W, x0 + tw)), int(min(H, y0 + th))))
            k = 448 / max(t.size)
            out.append(t.resize((round(t.width * k), round(t.height * k)), Image.LANCZOS))
    return out


class Checker:
    """VLM checks with a per-frame cache (the same question on the same frame is asked once)."""

    def __init__(self, frames, sc: Scene):
        self.frames, self.sc, self.cache, self.calls = frames, sc, {}, 0

    def _ask(self, key, images, q) -> bool:
        if key not in self.cache:
            ans = False
            for im in images:
                a, _, _ = vlm.yes_no([im], q)
                self.calls += 1
                if a:
                    ans = True
                    break
            self.cache[key] = ans
        return self.cache[key]

    def screen_app(self, i):
        return self.sc.laptop is not None and self._ask(("screen", i), [crop(self.frames[i][1], self.sc.laptop)],
                                                         Q_SCREEN)

    def safe_open(self, i):
        return self.sc.safe is not None and self._ask(("safe", i), [crop(self.frames[i][1], self.sc.safe)], Q_SAFE)

    def cash_on_desk(self, i):
        return self.sc.desk is not None and self._ask(("cash", i), _tiles(self.frames[i][1], self.sc.desk), Q_CASH)

    def items_on_desk(self, i):
        return self.sc.desk is not None and self._ask(("items", i), _tiles(self.frames[i][1], self.sc.desk, 2, 2),
                                                       Q_ITEMS)

    def eating(self, i, person_box):
        return self._ask(("eat", i), [crop(self.frames[i][1], person_box, margin=0.15)], Q_EAT)

    def green_bag(self, i) -> bool:
        if self.sc.desk is None:
            return False
        key = ("bag", i)
        if key not in self.cache:
            found = scene.detect(self.frames[i][1], ["bag"], threshold=0.3)
            self.cache[key] = any(s >= BAG_SCORE and _ioa(b, _grow(self.sc.desk, 0.1)) > 0.5
                                  for b, s in found.get("bag", []))
        return self.cache[key]


def _iou(a, b) -> float:
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def room_persons(dets, sc: Scene, H: int) -> list[list[people.Det]]:
    """People inside the cash room: confident, deduplicated, and standing/sitting at or below the desk line.
    Figures seen through the glass wall are further away, so their box ends above the desk top."""
    line = (sc.desk[1] if sc.desk else 0.42 * H) - 0.03 * H
    out = []
    for ds in dets:
        ps = sorted([d for d in ds if d.label == "person" and d.score >= 0.5], key=lambda d: -d.score)
        keep: list[people.Det] = []
        for d in ps:
            if d.bottom >= line and d.height >= 0.08 * H and all(_iou(d.box, k.box) < 0.6 for k in keep):
                keep.append(d)
        out.append(keep)
    return out


def stand_up_times(main_tops: list[float | None], t: list[float], H: int) -> list[float]:
    """Moments when the person at the desk stands up: the box top jumps up by > 7 % of the frame."""
    out = []
    for i in range(4, len(main_tops)):
        prev = [v for v in main_tops[i - 4:i - 1] if v is not None]
        if main_tops[i] is not None and prev and min(prev) - main_tops[i] > 0.07 * H:
            if not out or t[i] - out[-1] > 5:
                out.append(t[i - 1])
    return out


def detect_events(frames, dets, sc: Scene, chk: Checker) -> list[Event]:
    H = frames[0][1].height
    n = len(frames)
    t = [f[0] for f in frames]
    work = _grow(_union(sc.desk, sc.laptop, sc.safe), 0.15)
    cash_zone = _grow(_union(sc.laptop, sc.safe), 0.6, 0.4)
    safe_zone = _grow(sc.safe, 0.25)
    persons = room_persons(dets, sc, H)
    at_work = [[p for p in ps if _ioa(p.box, work) >= 0.25] for ps in persons]
    near_cash = [[p for p in ps if _ioa(p.box, cash_zone) >= 0.35] for ps in at_work]
    at_safe = [any(_ioa(p.box, safe_zone) >= 0.2 for p in ps) for ps in at_work]
    present = [len(ps) > 0 for ps in at_work]
    counts = [len(ps) for ps in at_work]
    main_top = [min((p.box[1] for p in ps), default=None) if ps else None for ps in near_cash]
    stands = stand_up_times(main_top, t, H)
    k = lambda s: max(1, round(s * FPS))  # noqa: E731
    events: list[Event] = []

    def has(type_id):
        return any(e.type_id == type_id for e in events)

    # 1) the cashier leaves her place (laptop/safe side): somebody there -> nobody for LEAVE_MIN_S,
    #    or nobody until the end of the video (clips often end right after she walks out)
    seat = [len(ps) > 0 for ps in near_cash]
    leave_runs = [(s, e) for s, e in _runs(seat, False, 2) if e - s >= k(LEAVE_MIN_S) or e == n]
    for s, _e in leave_runs:
        if s == 0:
            continue  # empty from the start: nobody left
        before = [x for x in stands if t[s] - 20 <= x <= t[s]]
        t_leave = before[-1] if before else t[max(0, s - 2)]  # she stands up, then walks out
        j = min(n - 1, s + k(2.0))  # state of the workplace 2 s after it is empty
        if chk.screen_app(j) and not has("T5"):
            events.append(Event("T5", t_leave, [t[j]], "кассир покидает рабочее место, экран ноутбука остаётся "
                                "с открытым рабочим окном (не заблокирован)", 0.8))
        if chk.cash_on_desk(j) and not has("T1"):
            events.append(Event("T1", t_leave, [t[j]], "кассир покидает рабочее место, на столе остаются наличные",
                                0.75))
        elif chk.green_bag(j) and not has("T1"):
            events.append(Event("T1", t_leave, [t[j]], "кассир покидает рабочее место, на столе остаётся "
                                "инкассаторская сумка", 0.8))
        if chk.safe_open(j) and not has("T4"):
            events.append(Event("T4", t_leave, [t[j]], "кассир покидает рабочее место, дверца сейфа открыта", 0.8))

    # 2) a visitor arrives while someone is already there -> cash already lying open on the desk?
    for i in range(1, n - k(ARRIVE_MIN_S)):
        if counts[i - 1] >= 1 and all(c > counts[i - 1] for c in counts[i:i + k(ARRIVE_MIN_S)]):
            if chk.cash_on_desk(i):
                events.append(Event("T2", t[i], [t[i]], "в зону стола входит посетитель, а на столе открыто лежат "
                                    "наличные", 0.7))
            break

    # 3) two or more people at the cashier's place (laptop/safe side), or three people at the desk
    for s, _e in _runs([len(ps) >= 2 for ps in near_cash], True, k(TWO_AT_DESK_MIN_S)):
        events.append(Event("T7", t[s], [t[s]], "рядом с кассиром на её стороне стола находится ещё один человек",
                            0.65))
        break
    if not has("T7"):
        for s, _e in _runs([c >= 3 for c in counts], True, k(TWO_AT_DESK_MIN_S)):
            events.append(Event("T7", t[s], [t[s]], "в кассовой зоне одновременно три человека — посторонний во "
                                "время обслуживания", 0.6))
            break

    # 4) safe left open: open while nobody is at the safe, two checks in a row (someone works at the desk)
    step = k(SAFE_CHECK_EVERY_S)
    streak = []
    for i in range(0, n, step):
        if present[i] and not at_safe[i] and chk.safe_open(i):
            streak.append(i)
            if len(streak) >= 2 and not has("T4"):
                events.append(Event("T4", t[streak[0]], [t[streak[0]], t[i]], "дверца сейфа остаётся открытой, "
                                    "пока кассир работает за столом", 0.75))
        else:
            streak = []

    # 5) eating/drinking at the workplace
    hits = []
    for i in range(0, n, k(EAT_CHECK_EVERY_S)):
        for p in near_cash[i][:1]:
            if chk.eating(i, list(p.box)):
                hits.append(i)
    if len(hits) >= 2:
        events.append(Event("T12", t[hits[0]], [t[hits[0]]], "кассир ест или пьёт за рабочим столом", 0.7))

    # 6) personal items on the desk right after the cashier arrives at an empty workplace
    for s, _e in _runs(present, True, k(ARRIVE_MIN_S)):
        if s > 0:
            j = min(n - 1, s + k(3.0))
            if chk.items_on_desk(j):
                events.append(Event("T11", t[s], [t[j]], "на рабочий стол положены верхняя одежда или личная сумка",
                                    0.6))
            break

    # 7) a visitor hands over a bag and leaves within ~12 s (no document/seal check); the bag was not there before
    for s, e in _runs([c >= 2 for c in counts], True, 1):
        if s == 0 or (t[e - 1] - t[s]) > 12.0 or chk.green_bag(max(0, s - k(2.0))):
            continue
        if any(chk.green_bag(i) for i in range(s, min(n, e + k(2.0)), 2)):
            events.append(Event("T10", t[s], [t[min(n - 1, s + k(2.0))]], "посетитель передаёт инкассаторскую "
                                "сумку и сразу уходит, проверки документа и пломбы не видно", 0.6))
            break
    return events


def run(path, progress=None) -> VideoAnalysis:
    started = time.perf_counter()
    vi = info(path)
    frames = sample(path, fps=FPS)
    if progress:
        progress(0.1, "decode")
    sc = find_scene(frames)
    if progress:
        progress(0.2, "scene")
    dets = people.detect([im for _, im in frames])
    if progress:
        progress(0.5, "detect")
    chk = Checker(frames, sc)
    events = detect_events(frames, dets, sc, chk)
    if progress:
        progress(0.9, "verify")
    by_t = {round(t_, 3): im for t_, im in frames}
    violations = []
    for ev in sorted(events, key=lambda e: e.t):
        type_name, rules, why = rules_for(ev.type_id)
        times = sorted({ev.t, *ev.evidence_t})[:3]
        frames_b64 = [EvidenceFrame(timestamp_sec=tt, image_base64=_jpeg_b64(by_t[round(tt, 3)])) for tt in times
                      if round(tt, 3) in by_t]
        violations.append(ViolationDetail(
            type=type_name, type_id=ev.type_id, timecode_sec=round(ev.t, 1), start_sec=round(ev.t, 1),
            end_sec=round(max(times), 1), confidence=ev.confidence,
            description=f"{ev.seen[0].upper()}{ev.seen[1:]} (≈{ev.t:.0f} с). {describe_rule(rules)}",
            evidence_frames=frames_b64, rules=rules, why=why))
    if progress:
        progress(1.0, "report")
    return VideoAnalysis(verdict="violation" if violations else "no_violation", violations=violations,
                         duration_sec=round(vi.duration, 2), processing_sec=round(time.perf_counter() - started, 2),
                         engine=f"rules+{vlm.DEFAULT_MODEL}+vlm_calls={chk.calls}")
