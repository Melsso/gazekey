from typing import Any

import numpy as np
import pytest

from gazekey.core.calibration import CalibrationFit, FitError
from gazekey.core.landmarks import LandmarkFrame
from gazekey.core.session import SessionRecorder
from gazekey.core.tracking import (
    GazeTracker,
    TrackingStats,
    build_tracker,
    calibration_error_deg,
)
from tests.synthetic import TRUE_SCREEN, gaze_features, make_frame, synthetic_session

STEP = round(1e9 / 30)
TARGET = (900.0, 300.0)


def frames_at(
    pos: tuple[float, float],
    n: int,
    *,
    start: int = 0,
    noise: float = 0.0,
    seed: int = 0,
    **kw: float,
) -> list[LandmarkFrame]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        feats = {k: v + float(rng.normal(0.0, noise)) for k, v in gaze_features(*pos).items()}
        out.append(make_frame((start + i) * STEP, **{**feats, **kw}))
    return out


def make_tracker(**kwargs: Any) -> tuple[GazeTracker, CalibrationFit]:
    return build_tracker(synthetic_session(), **kwargs)


def test_tracker_converges_on_a_steady_gaze() -> None:
    tracker, _ = make_tracker()
    last = None
    for frame in frames_at(TARGET, 90):
        last = tracker.update(frame)
    assert last is not None and last.point_px is not None and last.raw_px is not None
    assert last.raw_px == pytest.approx(TARGET, abs=0.5)
    assert last.point_px == pytest.approx(TARGET, abs=1.0)
    assert not last.held


def test_tracker_matches_offline_mapper_output() -> None:
    session = synthetic_session()
    tracker, fit = build_tracker(session)
    from gazekey.core.features import features_from_normalized

    frame = make_frame(0, **gaze_features(*TARGET))
    assert frame.points is not None
    feats = features_from_normalized(frame.points[None], frame.image_size, None)
    expected = fit.mapper.predict(feats[:, fit.columns])[0]
    raw = tracker.update(frame).raw_px
    assert raw == pytest.approx(tuple(expected), abs=1e-6)


def test_no_face_is_held_then_lost() -> None:
    tracker, _ = make_tracker()
    for frame in frames_at(TARGET, 10):
        tracker.update(frame)
    t = 10 * STEP
    held = tracker.update(make_frame(t + STEP, face=False))
    assert held.raw_px is None and held.held and held.point_px is not None
    lost = tracker.update(make_frame(t + round(0.6e9), face=False))
    assert lost.point_px is None and not lost.held


def test_blink_is_not_used_for_gaze() -> None:
    tracker, _ = make_tracker()
    est = tracker.update(make_frame(0, open_ratio=0.02, **gaze_features(*TARGET)))
    assert est.raw_px is None and est.point_px is None


def test_smoothing_reduces_jitter() -> None:
    tracker, _ = make_tracker()
    raws, smooth = [], []
    for frame in frames_at(TARGET, 240, noise=0.02, seed=3):
        est = tracker.update(frame)
        assert est.raw_px is not None and est.point_px is not None
        raws.append(est.raw_px)
        smooth.append(est.point_px)
    ratio = np.std(smooth[60:], axis=0) / np.std(raws[60:], axis=0)
    assert (ratio < 0.6).all()


def test_build_tracker_reports_fit_quality_and_failures() -> None:
    session = synthetic_session()
    _, fit = build_tracker(session)
    assert (fit.n_used, fit.n_total) == (9, 9)
    err = calibration_error_deg(fit, session)
    assert err is not None and err < 0.05
    empty = SessionRecorder((1280, 720)).build(screen=TRUE_SCREEN)
    with pytest.raises(FitError):
        build_tracker(empty)


def test_tracking_stats_counts_and_summary() -> None:
    tracker, _ = make_tracker()
    stats = TrackingStats()
    assert stats.summary() == "no frames processed"
    frames = [*frames_at(TARGET, 8), make_frame(8 * STEP, face=False)]
    for i, frame in enumerate(frames):
        stats.record(frame, tracker.update(frame), latency_ms=10.0 + i)
    assert (stats.n_frames, stats.n_face, stats.n_valid, stats.n_held) == (9, 8, 8, 1)
    text = stats.summary()
    assert (
        "frames: 9" in text
        and "face found: 88.9%" in text
        and "held through dropouts: 11.1%" in text
    )
    assert "median 14.0 ms" in text
