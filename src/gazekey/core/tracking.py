from dataclasses import dataclass, field

import numpy as np

from .calibration import DEFAULT_TIMING, CalibrationFit, DotTiming, fit_calibration
from .features import DEFAULT_OPEN_THRESHOLD, features_from_normalized
from .filters import GazeFilter
from .landmarks import LandmarkFrame
from .mapper import RidgeMapper, usable_mask
from .metrics import summarise_ms
from .session import Session

Point = tuple[float, float]


@dataclass(frozen=True, slots=True)
class GazeEstimate:
    t_ns: int
    raw_px: Point | None
    point_px: Point | None
    held: bool


class GazeTracker:
    def __init__(
        self,
        mapper: RidgeMapper,
        columns: list[int],
        gaze_filter: GazeFilter,
        open_threshold: float = DEFAULT_OPEN_THRESHOLD,
    ) -> None:
        self._mapper = mapper
        self._columns = columns
        self._filter = gaze_filter
        self._open_threshold = open_threshold

    def update(self, frame: LandmarkFrame) -> GazeEstimate:
        raw: Point | None = None
        if frame.points is not None:
            transform = None if frame.transform is None else frame.transform[None]
            feats = features_from_normalized(frame.points[None], frame.image_size, transform)
            if usable_mask(feats, self._columns, self._open_threshold)[0]:
                x, y = self._mapper.predict(feats[:, self._columns])[0]
                raw = (float(x), float(y))
        smoothed, held = self._filter.update(frame.t_ns, raw)
        point = None if smoothed is None else (float(smoothed[0]), float(smoothed[1]))
        return GazeEstimate(frame.t_ns, raw, point, held)


def build_tracker(
    session: Session,
    *,
    mapper: str = "ridge",
    feature_set: str = "iris",
    calibration: int = 9,
    min_cutoff: float = 0.8,
    beta: float = 0.005,
    timing: DotTiming = DEFAULT_TIMING,
    open_threshold: float = DEFAULT_OPEN_THRESHOLD,
) -> tuple[GazeTracker, CalibrationFit]:
    fit = fit_calibration(
        session,
        mapper=mapper,
        feature_set=feature_set,
        calibration=calibration,
        timing=timing,
        open_threshold=open_threshold,
    )
    gaze_filter = GazeFilter(min_cutoff=min_cutoff, beta=beta)
    return GazeTracker(fit.mapper, fit.columns, gaze_filter, open_threshold), fit


@dataclass(slots=True)
class TrackingStats:
    n_frames: int = 0
    n_face: int = 0
    n_valid: int = 0
    n_held: int = 0
    latency_ms: list[float] = field(default_factory=list)

    def record(self, frame: LandmarkFrame, estimate: GazeEstimate, latency_ms: float) -> None:
        self.n_frames += 1
        self.n_face += int(frame.valid)
        self.n_valid += int(estimate.raw_px is not None)
        self.n_held += int(estimate.held)
        self.latency_ms.append(latency_ms)

    def summary(self) -> str:
        if self.n_frames == 0:
            return "no frames processed"
        n = self.n_frames
        return "\n".join(
            [
                f"frames: {n}   face found: {100 * self.n_face / n:.1f}%   "
                f"gaze-valid: {100 * self.n_valid / n:.1f}%   held through dropouts: "
                f"{100 * self.n_held / n:.1f}%",
                f"latency capture->processed: {summarise_ms(self.latency_ms)}",
            ]
        )


def calibration_error_deg(fit: CalibrationFit, session: Session) -> float | None:
    if session.screen is None:
        return None
    return float(np.mean(session.screen.angular_distance_deg(fit.mapper.predict(fit.x), fit.y)))
