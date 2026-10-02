from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from . import F64
from .calibration import (
    DEFAULT_TIMING,
    DotTiming,
    FitError,
    analysis_window,
    calibration_dot_indices,
    fit_calibration,
)
from .features import DEFAULT_OPEN_THRESHOLD, features_from_normalized
from .geometry import ScreenGeometry, size_mm_for_angle
from .mapper import RidgeMapper, feature_columns, usable_mask
from .metrics import (
    chebyshev_deg,
    hit_rates,
    radial_sd_deg,
    rms,
    sample_to_sample_deg,
    size_for_hit_rate,
)
from .session import Conditions, Session


class EvaluationError(RuntimeError):
    pass


def _finite_mean(x: F64) -> float:
    finite = x[np.isfinite(x)]
    return float(finite.mean()) if finite.size else float("nan")


@dataclass(frozen=True, slots=True, eq=False)
class Summary:
    n_sessions: int
    n_val_dots: int
    mean_offset_deg: float
    median_offset_deg: float
    precision_rms_deg: float
    precision_sd_deg: float
    no_face_pct: float
    not_valid_pct: float
    cheb_deg: F64
    mean_distance_mm: float

    def hit_rates(self, sizes_deg: Sequence[float]) -> F64:
        return hit_rates(self.cheb_deg, sizes_deg)

    def size_for_hit_rate(self, rate: float = 0.9) -> float:
        return size_for_hit_rate(self.cheb_deg, rate)

    def size_cm(self, size_deg: float) -> float:
        return size_mm_for_angle(size_deg, self.mean_distance_mm) / 10.0


@dataclass(frozen=True, slots=True, eq=False)
class SessionResult:
    conditions: Conditions
    screen: ScreenGeometry
    mapper_name: str
    feature_set: str
    calibration_points: int
    fitted: RidgeMapper
    n_cal_used: int
    n_cal_total: int
    n_val_used: int
    n_val_total: int
    cal_offsets_deg: F64
    val_offsets_deg: F64
    val_cheb_deg: F64
    s2s_deg: F64
    val_sd_deg: F64
    n_frames: int
    n_no_face: int
    n_not_valid: int

    @property
    def summary(self) -> Summary:
        return summarise([self])


def summarise(results: Sequence[SessionResult]) -> Summary:
    if not results:
        raise ValueError("no results")
    offsets = np.concatenate([r.val_offsets_deg for r in results])
    frames = sum(r.n_frames for r in results)
    return Summary(
        n_sessions=len(results),
        n_val_dots=int(offsets.size),
        mean_offset_deg=float(offsets.mean()),
        median_offset_deg=float(np.median(offsets)),
        precision_rms_deg=rms(np.concatenate([r.s2s_deg for r in results])),
        precision_sd_deg=_finite_mean(np.concatenate([r.val_sd_deg for r in results])),
        no_face_pct=100.0 * sum(r.n_no_face for r in results) / max(frames, 1),
        not_valid_pct=100.0 * sum(r.n_not_valid for r in results) / max(frames, 1),
        cheb_deg=np.concatenate([r.val_cheb_deg for r in results]),
        mean_distance_mm=float(np.mean([r.screen.viewing_distance_mm for r in results])),
    )


def evaluate_session(
    session: Session,
    *,
    mapper: str = "ridge",
    feature_set: str = "iris",
    calibration: int = 9,
    timing: DotTiming = DEFAULT_TIMING,
    open_threshold: float = DEFAULT_OPEN_THRESHOLD,
    min_valid_frames: int = 5,
) -> SessionResult:
    screen = session.screen
    if screen is None:
        raise EvaluationError("session has no screen geometry (record it with `gazekey calibrate`)")
    val_dots = session.dot_indices("val")
    if session.dot_indices("cal").size == 0 or val_dots.size == 0:
        raise EvaluationError("session needs both calibration and validation dots")

    cols = feature_columns(feature_set)
    feats = features_from_normalized(session.landmarks, session.image_size, session.transforms)
    usable = usable_mask(feats, cols, open_threshold)
    face = session.face_found

    try:
        fit = fit_calibration(
            session,
            mapper=mapper,
            feature_set=feature_set,
            calibration=calibration,
            timing=timing,
            open_threshold=open_threshold,
            min_valid_frames=min_valid_frames,
            feats=feats,
            usable=usable,
        )
        cal_dots = calibration_dot_indices(session, calibration)
    except (FitError, ValueError) as exc:
        raise EvaluationError(str(exc)) from exc
    cal_offsets = screen.angular_distance_deg(fit.mapper.predict(fit.x), fit.y)

    n_frames = n_no_face = n_not_valid = 0
    for i in (*cal_dots, *val_dots):
        sl = analysis_window(session, int(i), timing)
        n_frames += len(range(*sl.indices(session.n_frames)))
        n_no_face += int(np.count_nonzero(~face[sl]))
        n_not_valid += int(np.count_nonzero(~usable[sl]))

    offsets: list[float] = []
    cheb: list[F64] = []
    s2s: list[F64] = []
    sds: list[float] = []
    for i in val_dots:
        sl = analysis_window(session, int(i), timing)
        idx = np.flatnonzero(usable[sl])
        if idx.size < min_valid_frames:
            continue
        pred = fit.mapper.predict(feats[sl][idx][:, cols])
        target = session.dot_positions_px[i]
        offsets.append(float(screen.angular_distance_deg(np.median(pred, axis=0), target)))
        cheb.append(chebyshev_deg(screen, pred, target))
        s2s.append(sample_to_sample_deg(screen, pred, idx))
        sds.append(radial_sd_deg(screen, pred))
    if not offsets:
        raise EvaluationError("no usable validation dots")

    return SessionResult(
        conditions=session.conditions,
        screen=screen,
        mapper_name=mapper,
        feature_set=feature_set,
        calibration_points=calibration,
        fitted=fit.mapper,
        n_cal_used=fit.n_used,
        n_cal_total=fit.n_total,
        n_val_used=len(offsets),
        n_val_total=int(val_dots.size),
        cal_offsets_deg=np.asarray(cal_offsets, dtype=np.float64),
        val_offsets_deg=np.array(offsets),
        val_cheb_deg=np.concatenate(cheb),
        s2s_deg=np.concatenate(s2s) if s2s else np.empty(0),
        val_sd_deg=np.array(sds),
        n_frames=n_frames,
        n_no_face=n_no_face,
        n_not_valid=n_not_valid,
    )
