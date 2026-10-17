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
from vision.devcache import DevCache
from vision.frames import crop, info, sample
from vision.pipeline import describe_rule, rules_for

FPS = 2.0
LEAVE_MIN_S = 3.0  # nobody at the workplace for this long = the cashier left
ARRIVE_MIN_S = 2.0
TWO_AT_DESK_MIN_S = 2.0
SAFE_OPEN_MIN_S = 6.0
SAFE_CHECK_EVERY_S = 2.0
SAFE_IDLE_EVERY_S = 8.0
T9_CASH_CHECKS = 4
T9_MACHINE_EVERY_S = 4.0
EAT_CHECK_EVERY_S = 10.0
BAG_SCORE = 0.55
BAGS = ("handbag", "backpack", "suitcase")
RETURN_MAX_S = 1.0  # a return to the post this short does not end leaving
LEAVE_LOOKBACK_S = 12.0
DEPTH_BAND = 0.3  # of a person's box height: feet this close in the image = the same distance from the camera
#                   (relative to people, not the frame: a closer or farther camera scales both)

Q_SCREEN = ("This is a close-up of a laptop. Does the laptop screen show an open working application (for example a "
            "spreadsheet, a table or a program window)? Answer no if the screen shows a lock screen, a wallpaper "
            "picture, or is dark or closed.")
Q_SAFE = "This is a close-up of a grey metal safe. Is the safe door open?"
Q_CASH = "Are there banknotes or a stack of paper money in this image?"
Q_EAT = "Is this person eating food or drinking from a cup or bottle right now?"
Q_ITEMS = ("Is there a personal item — a jacket, a coat or a handbag — lying on this desk? Answer no for cash, "
           "documents, a laptop, a phone or a green cash-collection bag.")
Q_MACHINE = ("This is a close-up of a banknote counting machine on a desk. Are banknotes being counted by the machine "
             "right now — notes in its top hopper or in its output tray, or a hand putting notes into it?")
Q_HANDLING = ("Is someone handling banknotes in this image right now — counting cash by hand or feeding money into "
              "a banknote counting machine (a machine with a slot and a display)?")


@dataclass
class Scene:
    desk: list[float] | None = None
    laptop: list[float] | None = None
    safe: list[float] | None = None
    door: list[float] | None = None
    machine: list[float] | None = None  # banknote counting machine (separate detection, see find_object)


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


def find_object(frames, key: str) -> list[float] | None:
    """One more scene object by its own text prompt (a joint prompt would change the other boxes)."""
    best = None
    for _, im in (frames[0], frames[len(frames) // 2], frames[-1]):
        items = scene.detect(im, [key], threshold=0.25).get(key, [])
        if items and (best is None or items[0][1] > best[1]):
            best = items[0]
    return best[0] if best else None


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

    def __init__(self, frames, sc: Scene, dc: DevCache | None = None):
        self.frames, self.sc, self.cache, self.calls = frames, sc, {}, 0
        self.by_kind: dict[str, int] = {}
        self.dc = dc

    def _ask(self, key, images, q) -> bool:
        if key not in self.cache:
            ans = False
            for im in images:
                self.by_kind[key[0]] = self.by_kind.get(key[0], 0) + 1
                a = self._yes_no(im, q)
                if a:
                    ans = True
                    break
            self.cache[key] = ans
        return self.cache[key]

    def _yes_no(self, im, q):
        def ask():
            self.calls += 1
            return vlm.yes_no([im], q)[0]
        return self.dc.vlm(im, q, ask) if self.dc else ask()

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

    def handling_cash(self, i, person_box):
        """Cash work at the counting machine: look at the desk around the machine, not at the person — from behind
        the cashier her hands are hidden by her body, but the machine and the notes are visible from above."""
        zone = _union(self.sc.desk, person_box) if self.sc.desk else person_box
        return self._ask(("cash_hands", i), _tiles(self.frames[i][1], zone, 2, 2), Q_HANDLING)

    def machine_busy(self, i):
        return self.sc.machine is not None and self._ask(("machine", i), [crop(self.frames[i][1], self.sc.machine)],
                                                         Q_MACHINE)

    def eating(self, i, person_box):
        return self._ask(("eat", i), [crop(self.frames[i][1], person_box, margin=0.15)], Q_EAT)

    def _bags(self, i) -> list:
        key = ("bag", i)
        if key not in self.cache:
            def det():
                return scene.detect(self.frames[i][1], ["bag"], threshold=0.3)
            self.cache[key] = (self.dc.get(key, det) if self.dc else det()).get("bag", [])
        return self.cache[key]

    def green_bag(self, i, near: float = 0.5) -> bool:
        """A collection bag on the desk (near=0.5: mostly inside the desk box) or at it (lower share: held over it)."""
        return self.sc.desk is not None and any(s >= BAG_SCORE and _ioa(b, _grow(self.sc.desk, 0.1)) > near
                                                for b, s in self._bags(i))


def _iou(a, b) -> float:
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def room_persons(dets, sc: Scene, H: int) -> list[list[people.Det]]:
    """People inside the cash room: confident, deduplicated, and standing/sitting at or below the desk line.
    Figures seen through the glass wall are further away, so their box ends above the desk top."""
    line = (sc.desk[1] - 0.03 * H) if sc.desk else 0.0  # no desk found (new camera): do not drop anybody
    out = []
    for ds in dets:
        ps = sorted([d for d in ds if d.label == "person" and d.score >= 0.5], key=lambda d: -d.score)
        keep: list[people.Det] = []
        for d in ps:
            if d.bottom >= line and d.height >= 0.08 * H and all(_iou(d.box, k.box) < 0.6 for k in keep):
                keep.append(d)
        out.append(keep)
    return out


@dataclass
class Tracks:
    """Per-frame layout of people relative to the scene objects."""
    t: list[float]
    persons: list  # people in the room
    at_work: list  # people in the workplace zone (desk + laptop + safe)
    near_cash: list  # ... on the cashier's side of the desk
    behind: list  # ... on the client's side
    present: list[bool]
    counts: list[int]
    client_here: list[bool]
    at_seat: list[bool]
    at_safe: list[bool]
    cashier_below: bool
    desk_top: float
    cashier: list  # the cashier's box per frame (the person on her side closest to her depth) or None
    depth: float | None  # the cashier's box bottom (pixels)


def cashier_depth(at_work, band: float, anchor=None) -> float | None:
    """Box bottom (pixels) of the cashier = her depth in the image.

    People at the workplace form depth groups (similar box bottoms): the cashier's and the client's across the desk.
    The cashier's group is the one closest to the laptop and the safe (`anchor` = their centre), which always stand on
    her side whatever the camera angle. Without them: the group seen alone at the workplace most often (before/after a
    client the cashier is there alone)."""
    people_ = [p for ps in at_work for p in ps]
    if not people_:
        return None
    w = band / 2
    left, groups = sorted(p.bottom for p in people_), []
    while left and len(groups) < 3:
        best = max(left, key=lambda b: sum(abs(x - b) <= w for x in left))
        near = [x for x in left if abs(x - best) <= w]
        if groups and len(near) < 0.15 * len(people_):
            break
        groups.append(near[len(near) // 2])
        left = [x for x in left if abs(x - best) > 2 * w]
    if len(groups) == 1:
        return groups[0]
    if anchor is not None:
        def dist(g):
            ps = [p for p in people_ if abs(p.bottom - g) <= w]
            cx = sorted(p.cx for p in ps)[len(ps) // 2]
            cy = sorted((p.box[1] + p.box[3]) / 2 for p in ps)[len(ps) // 2]
            return ((cx - anchor[0]) ** 2 + (cy - anchor[1]) ** 2) ** 0.5
        return min(groups, key=dist)
    solo = [ps[0].bottom for ps in at_work if len(ps) == 1]
    return max(groups, key=lambda g: sum(abs(x - g) <= w for x in solo))


def cashier_track(at_work, depth: float | None, band: float) -> list:
    """The cashier's box per frame: found by her depth, then followed to the nearest box frame to frame (she may move
    her chair or stand up). A jump to another depth (the client when she has left) is not followed."""
    out, last = [], None
    for ps in at_work:
        b = None
        if ps and depth is not None:
            if last is not None:
                b = min(ps, key=lambda p: abs(p.cx - last.cx) + abs(p.bottom - last.bottom))
                if abs(b.bottom - last.bottom) > 1.5 * band:
                    b = None
            if b is None:
                b = min(ps, key=lambda p: abs(p.bottom - depth))
                if abs(b.bottom - depth) > band:
                    b = None
        out.append(b)
        last = b if b is not None else last
    return out


def tracks(frames, dets, sc: Scene) -> Tracks:
    H = frames[0][1].height
    W = frames[0][1].width
    work = _grow(_union(sc.desk, sc.laptop, sc.safe), 0.15) or [0.0, 0.0, float(W), float(H)]  # none found: all
    safe_zone = _grow(sc.safe, 0.25)
    desk_top = sc.desk[1] if sc.desk else 0.45 * H
    persons = room_persons(dets, sc, H)
    at_work = [[p for p in ps if _ioa(p.box, work) >= 0.25] for ps in persons]
    # The cashier's side, without assuming a camera angle: people at the same depth as the cashier (box bottom within
    # DEPTH_BAND of a person's height) are on her side of the desk; a client across the desk stands at another depth
    # whichever way the camera looks (behind the cashier, in front of her or from the side). See cashier_depth.
    post = _union(sc.laptop, sc.safe)
    hs_ = sorted(p.height for ps in at_work for p in ps)
    band = max(0.03 * H, DEPTH_BAND * hs_[len(hs_) // 2]) if hs_ else 0.06 * H
    depth = cashier_depth(at_work, band, ((post[0] + post[2]) / 2, (post[1] + post[3]) / 2) if post else None)
    cashier_below = depth is not None and depth > desk_top + 0.08 * H  # informational: camera behind the cashier

    cashier = cashier_track(at_work, depth, band)

    def cashier_side(p, c) -> bool:
        return c is not None and abs(p.bottom - c.bottom) <= band

    near_cash = [[p for p in ps if cashier_side(p, c)] for ps, c in zip(at_work, cashier, strict=True)]
    behind = [[p for p in ps if not cashier_side(p, c)] for ps, c in zip(at_work, cashier, strict=True)]
    seat_zone = _grow(_union(sc.laptop, sc.safe), 0.6, 0.4)
    hs = sorted(p.height for ps in near_cash for p in ps)
    h_sit = hs[len(hs) // 4] if hs else 0.0  # the cashier's box when seated; taller = standing (at the safe)
    in_front = lambda p: sc.safe is not None and sc.safe[0] <= p.cx <= sc.safe[2]  # noqa: E731
    return Tracks(
        t=[f[0] for f in frames], persons=persons, at_work=at_work, near_cash=near_cash, behind=behind,
        present=[len(ps) > 0 for ps in at_work], counts=[len(ps) for ps in at_work],
        client_here=[len(ps) > 0 for ps in behind],
        at_seat=[any(_ioa(p.box, seat_zone) >= 0.25 for p in ps) for ps in near_cash],
        # at the safe = operating it: in its zone and standing, or right in front of it (sitting next to it is not)
        at_safe=[any(_ioa(p.box, safe_zone) >= 0.2 and (p.height > 1.25 * h_sit or in_front(p)) for p in ps)
                 for ps in at_work],
        cashier_below=cashier_below, desk_top=desk_top,
        cashier=cashier, depth=depth)


def leave_start(tr: Tracks, s: int, W: int) -> int:
    """First frame of leaving before the workplace got empty at frame s: the cashier steps away from her working
    position and does not come back (returns of <= RETURN_MAX_S are ignored). Her working position = where she sits
    (the frames with her lowest boxes); leaving can start up to LEAVE_LOOKBACK_S before s."""
    boxes = [b for b in tr.cashier if b is not None]
    if not boxes:
        return s
    h_sit = sorted(b.height for b in boxes)[len(boxes) // 4]
    sit = [b for b in boxes if b.height <= 1.25 * h_sit]
    x_sit = sorted(b.cx for b in sit)[len(sit) // 2]
    tol = max(0.05 * W, 0.35 * sorted(b.box[2] - b.box[0] for b in sit)[len(sit) // 2])
    at_post = [b is not None and abs(b.cx - x_sit) <= tol for b in tr.cashier]
    k_ret = max(1, round(RETURN_MAX_S * FPS))
    lo = max(0, s - round(LEAVE_LOOKBACK_S * FPS))
    i = s - 1
    while i >= lo:
        if not at_post[i]:
            i -= 1
            continue
        j = i
        while j >= lo and at_post[j]:
            j -= 1
        if i - j <= k_ret and j >= lo and not at_post[j]:
            i = j  # a short return to the post on the way out
            continue
        break
    return min(s, i + 1)


def client_arrivals(tr: Tracks) -> list[int]:
    """Frames where a new client comes to the desk: a client-side person standing (box >= 1.4x the seated client
    box) when nobody sat there for 3 s, who sits down within 4 s; or a client appearing after >= 2 s with nobody on
    the client side."""
    hs = sorted(p.height for ps in tr.behind for p in ps)
    if not hs:
        return []
    h_sit = hs[len(hs) // 4]
    stand = [any(p.height >= 1.4 * h_sit for p in ps) for ps in tr.behind]
    sit = [any(p.height <= 1.25 * h_sit for p in ps) for ps in tr.behind]
    k = lambda sec: max(1, round(sec * FPS))  # noqa: E731
    out = []
    for s, e in _runs(stand, True, 1):
        new = not any(sit[max(0, s - k(3.0)):s])  # nobody was sitting there just before: a new client
        if new and any(sit[e:e + k(4.0)]) and not (out and s - out[-1] < k(3.0)):
            out.append(s)
    for s, _e in _runs(tr.client_here, True, k(2.0)):
        if s >= k(2.0) and not any(abs(s - o) < k(3.0) for o in out):
            out.append(s)
    return sorted(out)


def detect_events(frames, dets, sc: Scene, chk: Checker) -> list[Event]:
    n = len(frames)
    tr = tracks(frames, dets, sc)
    t, near_cash, present, counts = tr.t, tr.near_cash, tr.present, tr.counts
    client_here, at_safe = tr.client_here, tr.at_safe
    k = lambda s: max(1, round(s * FPS))  # noqa: E731
    events: list[Event] = []

    def has(type_id):
        return any(e.type_id == type_id for e in events)

    # 1) the cashier leaves the workplace: somebody at the desk -> nobody for LEAVE_MIN_S,
    #    or nobody until the end of the video (clips often end right after she walks out)
    leave_runs = [(s, e) for s, e in _runs(present, False, 2) if e - s >= k(LEAVE_MIN_S) or e == n]
    for s, _e in leave_runs:
        if s == 0:
            continue  # empty from the start: nobody left
        t_leave = t[leave_start(tr, s, frames[0][1].width)]
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
        if not has("T11") and chk.items_on_desk(j) and chk.items_on_desk(min(n - 1, j + k(1.0))):
            events.append(Event("T11", t_leave, [t[j]], "кассир покидает рабочее место, на столе остаются личные "
                                "вещи (сумка, верхняя одежда)", 0.6))

    # 2) a client comes to the desk while cash already lies open on it (checked for every client, not the first)
    for i in client_arrivals(tr):
        j = min(n - 1, i + k(1.0))
        if present[i] and chk.cash_on_desk(i) and chk.cash_on_desk(j) and not has("T2"):
            events.append(Event("T2", t[i], [t[i], t[j]], "к столу подходит клиент, а на столе открыто лежат "
                                "наличные", 0.7))

    # 3) an outsider on the cashier's side: two or more people between the camera and the desk for a while.
    #    (People waiting by the door or seen through the glass are not on the cashier's side and do not count.)
    for s, _e in _runs([len(ps) >= 2 for ps in near_cash], True, k(TWO_AT_DESK_MIN_S)):
        events.append(Event("T7", t[s], [t[s]], "рядом с кассиром на её стороне стола находится ещё один человек",
                            0.65))
        break

    # 4) safe left open: open while the cashier is away from the safe and no client is being served
    #    (an open safe during a client operation is normal practice in the clean clips)
    #    Checked every SAFE_CHECK_EVERY_S in the first 6 s and right after she steps away from the safe (when a
    #    forgotten door shows up), and every SAFE_IDLE_EVERY_S otherwise — the long clips stay within the time limit.
    step = k(SAFE_CHECK_EVERY_S)
    when = set(range(0, min(n, k(6.0)), step)) | set(range(0, n, k(SAFE_IDLE_EVERY_S)))
    for i in range(1, n):
        if at_safe[i - 1] and not at_safe[i]:
            when |= set(range(i, min(n, i + k(6.0)), step))
    #    Only while she sits: standing by the open safe she is still working with it (X-25, X-27 close it seconds
    #    later); sitting down to count with the door open is the violation (X-20).
    hs = sorted(b.height for b in tr.cashier if b is not None)
    h_sit = hs[len(hs) // 10] if hs else 0.0  # her lowest boxes = sitting (she may stand most of the clip)
    seated = [b is not None and b.height <= 1.25 * h_sit for b in tr.cashier]
    streak = []
    for i in sorted(when):
        if present[i] and seated[i] and not at_safe[i] and not client_here[i] and chk.safe_open(i):
            streak.append(i)
            if len(streak) >= 2 and not has("T4"):
                events.append(Event("T4", t[streak[0]], [t[streak[0]], t[i]], "дверца сейфа остаётся открытой, "
                                    "пока кассир работает за столом", 0.75))
        else:
            streak = []

    # 4b) counting without the client: a client was served (>= 5 s), left, and the cashier keeps handling cash
    was_client = _runs(client_here, True, k(5.0))
    for _s, e in was_client:
        if e >= n - 1 or not all(present[e:min(n, e + k(3.0))]) or has("T8"):
            continue
        j, j2 = min(n - 1, e + k(1.5)), min(n - 1, e + k(4.0))
        ps, ps2 = near_cash[j], near_cash[j2]
        if ps and ps2 and chk.handling_cash(j, list(ps[0].box)) and chk.handling_cash(j2, list(ps2[0].box)):
            events.append(Event("T8", t[e], [t[j]], "клиент уходит, а кассир продолжает пересчёт его наличных "
                                "уже без него", 0.7))

    # 5) eating/drinking at the workplace: asked when the detector sees a cup/bottle at the cashier (every 2 s at most)
    #    and once per EAT_CHECK_EVERY_S otherwise — a few VLM calls instead of one every few seconds
    cand = set(range(0, n, k(EAT_CHECK_EVERY_S)))
    last = -k(2.0)
    for i, ds in enumerate(dets):
        cb = tr.cashier[i]
        if cb and i - last >= k(2.0) and any(d.label in ("cup", "bottle") and d.score >= 0.4
                                               and _ioa(d.box, _grow(list(cb.box), 0.3)) > 0.5 for d in ds):
            cand.add(i)
            last = i
    hits = [i for i in sorted(cand) if tr.cashier[i] and chk.eating(i, list(tr.cashier[i].box))]
    if len(hits) >= 2:
        events.append(Event("T12", t[hits[0]], [t[hits[0]], t[hits[1]]], "кассир ест или пьёт за рабочим столом",
                            0.7))

    # 6) personal items: the detector sees a bag lying on the cashier's half of the desk (not carried by anyone) for
    #    >= 3 s, and the VLM confirms a jacket/handbag on the desk. The cashier's half: the half of the desk box on the
    #    side of her depth line — so client's things across the desk do not count, whatever the camera angle.
    if sc.desk and tr.depth is not None:
        desk_zone, mid = _grow(sc.desk, 0.05), (sc.desk[1] + sc.desk[3]) / 2
        mine = lambda b: ((b[1] + b[3]) / 2 >= mid) == (tr.depth >= mid)  # noqa: E731
        bag = [any(d.label in BAGS and d.score >= 0.5 and _ioa(d.box, desk_zone) >= 0.6 and mine(d.box)
                   and all(_ioa(d.box, p.box) < 0.5 for p in tr.persons[i]) for d in dets[i]) for i in range(n)]
        for s, _e in _runs(bag, True, k(3.0)):
            j = min(n - 1, s + k(1.0))
            if chk.items_on_desk(j):
                events.append(Event("T11", t[s], [t[s], t[j]], "на рабочем столе кассира лежат личные вещи "
                                    "(сумка, верхняя одежда)", 0.6))
                break

    # 7) a visitor hands over a bag and leaves within ~12 s (no document/seal check); the bag was not there before
    for s, e in _runs([c >= 2 for c in counts], True, 1):
        if s == 0 or (t[e - 1] - t[s]) > 12.0 or chk.green_bag(max(0, s - k(2.0))):
            continue
        if any(chk.green_bag(i) for i in range(s, min(n, e + k(2.0)), 2)):
            events.append(Event("T10", t[s], [t[min(n - 1, s + k(2.0))]], "посетитель передаёт инкассаторскую "
                                "сумку и сразу уходит, проверки документа и пломбы не видно", 0.6))
            break
    # 8) a cash operation without the counting machine: cash is seen on the desk while a client is served or a
    #    collection bag is open, and the machine is never seen at work in the whole video. With a client — only as a
    #    refinement of a video that already shows a violation: on its own "machine not seen at work" also fires on
    #    clean client clips (E2: X-3, X-5, X-19, X-37), where the VLM misses the machine.
    if sc.machine is not None and not has("T9"):
        ctx = [i for s, e in _runs(client_here, True, k(5.0)) for i in range(s, e)] if events else []
        if not ctx:  # a collection bag opened at the desk (X-13): its cash must be counted on the machine too
            ctx = [i for i in range(0, n, max(1, n // 6)) if present[i] and chk.green_bag(i, near=0.25)][:6]
        picks = sorted(set(ctx[:: max(1, len(ctx) // T9_CASH_CHECKS)]))[:T9_CASH_CHECKS] if ctx else []
        cash = [i for i in picks if chk.cash_on_desk(i)]
        if cash and not any(chk.machine_busy(i) for i in range(0, n, k(T9_MACHINE_EVERY_S)) if present[i]):
            i = cash[0]
            events.append(Event("T9", t[i], [t[i]], "операция с наличными проходит вручную: купюры на столе, "
                                "счётная машина за всё видео не используется", 0.6))
    return events


def run(path, progress=None) -> VideoAnalysis:
    started = time.perf_counter()
    vi = info(path)
    frames = sample(path, fps=FPS)
    dc = DevCache(path)
    if progress:
        progress(0.1, "decode")
    sc = dc.get("scene", lambda: find_scene(frames))
    # another camera / room: a missing object is searched once more by a plainer description
    for key, alt in (("desk", "table"), ("laptop", "monitor"), ("safe", "cabinet")):
        if getattr(sc, key) is None:
            setattr(sc, key, dc.get(("scene_extra", alt), lambda alt=alt: find_object(frames, alt)))
    sc.machine = dc.get(("scene_extra", "machine"), lambda: find_object(frames, "machine"))
    if progress:
        progress(0.2, "scene")
    dets = dc.get("dets", lambda: people.detect([im for _, im in frames]))
    if progress:
        progress(0.5, "detect")
    chk = Checker(frames, sc, dc)
    events = detect_events(frames, dets, sc, chk)
    dc.save()
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
                         engine=f"rules+{vlm.DEFAULT_MODEL}+vlm_calls={chk.calls}"
                         f"+asked={','.join(f'{a}:{b}' for a, b in sorted(chk.by_kind.items()))}")
