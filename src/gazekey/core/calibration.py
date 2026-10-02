import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from . import F64
from .features import DEFAULT_OPEN_THRESHOLD, features_from_normalized
from .geometry import ScreenGeometry
from .landmarks import LandmarkFrame
from .mapper import RidgeMapper, feature_columns, make_mapper, usable_mask
from .session import Conditions, DotRole, Session, SessionRecorder

Point = tuple[float, float]
Phase = Literal["waiting", "lead_in", "dot", "done", "aborted"]
CALIBRATION_FRACTIONS = (0.1, 0.5, 0.9)
EXTRA_FRACTIONS = ((0.3, 0.3), (0.7, 0.3), (0.7, 0.7), (0.3, 0.7))
FIVE_POINT = (0, 2, 4, 6, 8)
CALIBRATION_SIZES = (5, 9, 13)
MIN_CALIBRATION_DOTS = 3


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


def extra_calibration_points(width: float, height: float) -> list[Point]:
    """The 4 inner points (at 30/70 %) that turn the 9-point grid into the 13-point variant."""
    return [(fx * width, fy * height) for fx, fy in EXTRA_FRACTIONS]


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
    width: float,
    height: float,
    *,
    seed: int = 0,
    n_validation: int = 16,
    n_calibration: int = 9,
) -> list[DotPlan]:
    """9 grid dots ("cal"), optionally 4 inner ones ("ext", when n_calibration is 13), then
    `n_validation` validation dots that avoid every calibration point."""
    if n_calibration not in (9, 13):
        raise ValueError("n_calibration must be 9 or 13")
    cal = calibration_points(width, height)
    ext = extra_calibration_points(width, height) if n_calibration == 13 else []
    val = (
        validation_points(width, height, cal + ext, n=n_validation, seed=seed)
        if n_validation
        else []
    )
    return (
        [DotPlan(p, "cal") for p in cal]
        + [DotPlan(p, "ext") for p in ext]
        + [DotPlan(p, "val") for p in val]
    )


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


class FitError(RuntimeError):
    pass


def calibration_dot_indices(session: Session, points: int = 9) -> NDArray[np.intp]:
    cal = session.dot_indices("cal")
    if points == 9:
        return cal
    if points == 5:
        if cal.size < 9:
            raise ValueError("the 5-point variant needs the full 9-dot grid")
        return cal[list(FIVE_POINT)]
    if points == 13:
        ext = session.dot_indices("ext")
        if ext.size == 0:
            raise ValueError("session has no extra calibration dots; record with `calibrate -n 13`")
        return np.concatenate([cal, ext])
    raise ValueError(f"calibration points must be one of {CALIBRATION_SIZES}")


@dataclass(frozen=True, slots=True, eq=False)
class CalibrationFit:
    mapper: RidgeMapper
    columns: list[int]
    n_used: int
    n_total: int
    x: F64
    y: F64


def fit_calibration(
    session: Session,
    *,
    mapper: str = "ridge",
    feature_set: str = "iris",
    calibration: int = 9,
    timing: DotTiming = DEFAULT_TIMING,
    open_threshold: float = DEFAULT_OPEN_THRESHOLD,
    min_valid_frames: int = 5,
    feats: F64 | None = None,
    usable: NDArray[np.bool_] | None = None,
) -> CalibrationFit:
    columns = feature_columns(feature_set)
    try:
        dots = calibration_dot_indices(session, calibration)
    except ValueError as exc:
        raise FitError(str(exc)) from exc
    if feats is None:
        feats = features_from_normalized(session.landmarks, session.image_size, session.transforms)
    if usable is None:
        usable = usable_mask(feats, columns, open_threshold)

    x_rows: list[F64] = []
    y_rows: list[F64] = []
    for i in dots:
        sl = analysis_window(session, int(i), timing)
        rows = feats[sl][usable[sl]][:, columns]
        if len(rows) >= min_valid_frames:
            x_rows.append(np.median(rows, axis=0))
            y_rows.append(session.dot_positions_px[i])
    if len(x_rows) < MIN_CALIBRATION_DOTS:
        raise FitError(
            f"only {len(x_rows)} usable calibration dots (need >= {MIN_CALIBRATION_DOTS})"
        )
    x, y = np.array(x_rows), np.array(y_rows)
    return CalibrationFit(make_mapper(mapper).fit(x, y), columns, len(x_rows), int(dots.size), x, y)
