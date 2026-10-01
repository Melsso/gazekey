import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .geometry import ScreenGeometry
from .landmarks import LandmarkFrame
from .session import Conditions, DotRole, Session, SessionRecorder

CALIBRATION_FRACTIONS = (0.1, 0.5, 0.9)
Point = tuple[float, float]
Phase = Literal["waiting", "lead_in", "dot", "done", "aborted"]


@dataclass(frozen=True, slots=True)
class DotTiming:
    skip_s: float = 0.5
    window_s: float = 1.0
    tail_s: float = 0.2

    @property
    def total_s(self) -> float:
        return self.skip_s + self.window_s + self.tail_s


DEFAULT_TIMING = DotTiming()


@dataclass(frozen=True, slots=True)
class DotPlan:
    position_px: Point
    role: DotRole


def calibration_points(width: float, height: float) -> list[Point]:
    points: list[Point] = []
    for row, fy in enumerate(CALIBRATION_FRACTIONS):
        columns = CALIBRATION_FRACTIONS if row % 2 == 0 else CALIBRATION_FRACTIONS[::-1]
        points.extend((fx * width, fy * height) for fx in columns)
    return points


def validation_points(
    width: float,
    height: float,
    avoid: list[Point],
    *,
    n: int = 16,
    seed: int = 0,
    margin: float = 0.1,
    jitter: float = 0.35,
    min_separation_frac: float = 0.05,
    max_tries: int = 200,
) -> list[Point]:
    side = math.isqrt(n)
    if side * side != n:
        raise ValueError("n must be a perfect square")
    rng = np.random.default_rng(seed)
    cell = (1.0 - 2.0 * margin) / side
    min_sep = min_separation_frac * height
    avoid_arr = np.asarray(avoid, dtype=np.float64).reshape(-1, 2)

    points: list[Point] = []
    for row in range(side):
        for col in range(side):
            for _ in range(max_tries):
                fx = margin + (col + 0.5 + rng.uniform(-jitter, jitter)) * cell
                fy = margin + (row + 0.5 + rng.uniform(-jitter, jitter)) * cell
                p = (fx * width, fy * height)
                if (
                    avoid_arr.size == 0
                    or np.min(np.hypot(avoid_arr[:, 0] - p[0], avoid_arr[:, 1] - p[1])) >= min_sep
                ):
                    points.append(p)
                    break
            else:
                raise ValueError("could not place a validation point away from calibration points")
    return [points[i] for i in rng.permutation(len(points))]


def build_schedule(
    width: float, height: float, *, seed: int = 0, n_validation: int = 16
) -> list[DotPlan]:
    cal = calibration_points(width, height)
    val = validation_points(width, height, cal, n=n_validation, seed=seed)
    return [DotPlan(p, "cal") for p in cal] + [DotPlan(p, "val") for p in val]


def analysis_window(session: Session, dot_index: int, timing: DotTiming) -> slice:
    start = int(session.dot_t_on_ns[dot_index]) + round(timing.skip_s * 1e9)
    return session.frame_slice(start, start + round(timing.window_s * 1e9))


class CalibrationController:
    def __init__(self, timing: DotTiming = DEFAULT_TIMING, lead_in_s: float = 1.5) -> None:
        self.timing = timing
        self._lead_in_ns = round(lead_in_s * 1e9)
        self._total_ns = round(timing.total_s * 1e9)
        self.phase: Phase = "waiting"
        self._schedule: list[DotPlan] = []
        self._recorder: SessionRecorder | None = None
        self._last_frame: LandmarkFrame | None = None
        self._t_start = 0
        self._dot_on = 0
        self._index = -1

    @property
    def camera_ready(self) -> bool:
        return self._last_frame is not None

    @property
    def face_visible(self) -> bool:
        return self._last_frame is not None and self._last_frame.valid

    @property
    def current_dot(self) -> DotPlan | None:
        return self._schedule[self._index] if self.phase == "dot" else None

    @property
    def n_dots(self) -> int:
        return len(self._schedule)

    @property
    def dot_number(self) -> int:
        return self._index + 1

    def progress(self, t_ns: int) -> float:
        if self.phase != "dot":
            return 0.0
        return min(max((t_ns - self._dot_on) / self._total_ns, 0.0), 1.0)

    def on_frame(self, frame: LandmarkFrame) -> None:
        self._last_frame = frame
        if self._recorder is not None and self.phase in ("lead_in", "dot"):
            self._recorder.add_frame(frame)

    def start(self, t_ns: int, schedule: list[DotPlan]) -> None:
        if self.phase != "waiting":
            raise RuntimeError("already started")
        if self._last_frame is None:
            raise RuntimeError("no camera frames received yet")
        if not schedule:
            raise ValueError("empty schedule")
        self._schedule = schedule
        self._recorder = SessionRecorder(self._last_frame.image_size)
        self._t_start = t_ns
        self.phase = "lead_in"

    def update(self, t_ns: int) -> None:
        if self.phase == "lead_in" and t_ns >= self._t_start + self._lead_in_ns:
            self._begin_dot(0, t_ns)
        elif self.phase == "dot" and t_ns >= self._dot_on + self._total_ns:
            self._end_dot(t_ns)
            if self._index + 1 < len(self._schedule):
                self._begin_dot(self._index + 1, t_ns)
            else:
                self.phase = "done"

    def abort(self) -> None:
        if self.phase not in ("done", "aborted"):
            self.phase = "aborted"

    def build_session(self, conditions: Conditions, screen: ScreenGeometry) -> Session:
        if self.phase != "done" or self._recorder is None:
            raise RuntimeError("calibration is not complete")
        return self._recorder.build(conditions=conditions, screen=screen)

    def _begin_dot(self, index: int, t_ns: int) -> None:
        assert self._recorder is not None
        plan = self._schedule[index]
        self._recorder.begin_dot(plan.position_px, plan.role, t_ns)
        self._index, self._dot_on, self.phase = index, t_ns, "dot"

    def _end_dot(self, t_ns: int) -> None:
        assert self._recorder is not None
        self._recorder.end_dot(t_ns)
