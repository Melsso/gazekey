from dataclasses import dataclass

from .geometry import ScreenGeometry

Point = tuple[float, float]

BACKSPACE = "\b"
KEY_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ .,\b"
COLS, ROWS = 6, 5
_TEXT_STRIP_FRAC = 0.16


def char_label(ch: str) -> str:
    return {" ": "\u2423", BACKSPACE: "\u232b"}.get(ch, ch)


@dataclass(frozen=True, slots=True)
class Rect:
    x: float
    y: float
    w: float
    h: float

    def contains(self, p: Point) -> bool:
        return self.x <= p[0] < self.x + self.w and self.y <= p[1] < self.y + self.h


@dataclass(frozen=True, slots=True)
class Key:
    id: str
    label: str
    rect: Rect


class KeyboardModel:
    def __init__(self, width: float, height: float) -> None:
        self.width = width
        self.height = height
        self.text = ""

    @property
    def keys(self) -> list[Key]:
        top = self.height * _TEXT_STRIP_FRAC
        cell_w, cell_h = self.width / COLS, (self.height - top) / ROWS
        return [
            Key(
                f"c:{ch}",
                char_label(ch),
                Rect((i % COLS) * cell_w, top + (i // COLS) * cell_h, cell_w, cell_h),
            )
            for i, ch in enumerate(KEY_CHARS)
        ]

    def key_at(self, point: Point | None) -> str | None:
        if point is None:
            return None
        for key in self.keys:
            if key.rect.contains(point):
                return key.id
        return None

    def press(self, key_id: str) -> None:
        if not key_id.startswith("c:") or key_id[2:] not in KEY_CHARS or len(key_id) != 3:
            raise ValueError(f"unknown key {key_id!r}")
        ch = key_id[2:]
        self.text = self.text[:-1] if ch == BACKSPACE else self.text + ch


def describe_keys(screen: ScreenGeometry) -> str:
    mm_x, mm_y = screen.mm_per_px
    top = screen.height_px * _TEXT_STRIP_FRAC
    w = screen.width_px / COLS
    h = (screen.height_px - top) / ROWS
    return (
        f"{len(KEY_CHARS)} keys, each {w * mm_x / 10:.1f} x {h * mm_y / 10:.1f} cm = "
        f"{screen.px_to_deg(w, 'x'):.1f}\u00b0 x {screen.px_to_deg(h, 'y'):.1f}\u00b0"
    )


@dataclass(frozen=True, slots=True)
class DwellState:
    key_id: str | None
    progress: float
    selected: str | None


class DwellSelector:
    def __init__(
        self, dwell_s: float = 0.8, grace_s: float = 0.15, cooldown_s: float = 0.5
    ) -> None:
        if dwell_s <= 0 or grace_s < 0 or cooldown_s < 0:
            raise ValueError("dwell_s must be positive; grace_s and cooldown_s non-negative")
        self._dwell_ns = round(dwell_s * 1e9)
        self._grace_ns = round(grace_s * 1e9)
        self._cool_ns = round(cooldown_s * 1e9)
        self._target: str | None = None
        self._start = 0
        self._last_on = 0
        self._cool_until = 0

    def update(self, t_ns: int, key_id: str | None) -> DwellState:
        if t_ns < self._cool_until:
            return DwellState(None, 0.0, None)
        if (
            self._target is not None
            and key_id != self._target
            and t_ns - self._last_on > self._grace_ns
        ):
            self._target = None
        if self._target is None:
            if key_id is None:
                return DwellState(None, 0.0, None)
            self._target, self._start, self._last_on = key_id, t_ns, t_ns
        elif key_id == self._target:
            self._last_on = t_ns
        progress = min((t_ns - self._start) / self._dwell_ns, 1.0)
        if progress >= 1.0:
            selected = self._target
            self._target = None
            self._cool_until = t_ns + self._cool_ns
            return DwellState(selected, 1.0, selected)
        return DwellState(self._target, progress, None)
