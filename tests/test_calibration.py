import numpy as np
import pytest

from gazekey.core.calibration import (
    CalibrationController,
    DotTiming,
    FitError,
    analysis_window,
    build_schedule,
    calibration_dot_indices,
    calibration_points,
    extra_calibration_points,
    fit_calibration,
    validation_points,
)
from gazekey.core.session import Conditions
from tests.synthetic import TRUE_SCREEN, make_frame, synthetic_session

W, H = 1440.0, 900.0


def test_calibration_grid_is_3x3_serpentine() -> None:
    pts = calibration_points(W, H)
    assert len(pts) == 9 and len(set(pts)) == 9
    xs = [p[0] / W for p in pts]
    ys = [p[1] / H for p in pts]
    assert sorted(set(xs)) == pytest.approx([0.1, 0.5, 0.9])
    assert sorted(set(ys)) == pytest.approx([0.1, 0.5, 0.9])
    assert xs[:3] == pytest.approx([0.1, 0.5, 0.9])
    assert xs[3:6] == pytest.approx([0.9, 0.5, 0.1])


def test_validation_points_never_coincide_with_calibration_points() -> None:
    cal = calibration_points(W, H)
    for seed in range(20):
        val = validation_points(W, H, cal, seed=seed)
        assert len(val) == 16
        d = np.hypot(
            np.array(val)[:, None, 0] - np.array(cal)[None, :, 0],
            np.array(val)[:, None, 1] - np.array(cal)[None, :, 1],
        )
        assert d.min() >= 0.05 * H


def test_validation_points_cover_grid_cells_inside_margins() -> None:
    val = validation_points(W, H, calibration_points(W, H), seed=4)
    fx = np.array([p[0] / W for p in val])
    fy = np.array([p[1] / H for p in val])
    assert fx.min() >= 0.1 and fx.max() <= 0.9 and fy.min() >= 0.1 and fy.max() <= 0.9
    cells = {(int((x - 0.1) / 0.2), int((y - 0.1) / 0.2)) for x, y in zip(fx, fy, strict=True)}
    assert len(cells) == 16


def test_validation_points_are_seeded_and_not_grid_aligned() -> None:
    cal = calibration_points(W, H)
    a = validation_points(W, H, cal, seed=1)
    assert a == validation_points(W, H, cal, seed=1)
    assert a != validation_points(W, H, cal, seed=2)
    assert len({round(p[0] / W, 3) for p in a}) > 4


def test_validation_points_argument_checks() -> None:
    with pytest.raises(ValueError, match="perfect square"):
        validation_points(W, H, [], n=10)
    with pytest.raises(ValueError, match="could not place"):
        validation_points(W, H, [(W / 2, H / 2)], min_separation_frac=50.0, max_tries=3)


def test_schedule_is_9_calibration_then_16_validation() -> None:
    plan = build_schedule(W, H, seed=0)
    assert [p.role for p in plan] == ["cal"] * 9 + ["val"] * 16


def test_dot_timing_total() -> None:
    assert DotTiming().total_s == pytest.approx(1.7)


def test_controller_sequencing_and_recording() -> None:
    session = synthetic_session()
    assert session.n_dots == 25
    assert session.dot_roles.tolist() == ["cal"] * 9 + ["val"] * 16

    assert np.array_equal(session.dot_t_on_ns[1:], session.dot_t_off_ns[:-1])
    durations = (session.dot_t_off_ns - session.dot_t_on_ns) / 1e9
    assert durations == pytest.approx(DotTiming().total_s, abs=0.05)
    assert session.conditions.participant == "p1"
    assert session.screen == TRUE_SCREEN
    assert np.all(np.diff(session.timestamps_ns) > 0)


def test_analysis_window_skips_latency_and_spans_one_second() -> None:
    session = synthetic_session()
    timing = DotTiming()
    for i in (0, 12, 24):
        sl = analysis_window(session, i, timing)
        t0 = session.timestamps_ns[sl][0] - session.dot_t_on_ns[i]
        t1 = session.timestamps_ns[sl][-1] - session.dot_t_on_ns[i]
        assert t0 >= 0.5e9 and t1 < 1.5e9
        assert (t1 - t0) / 1e9 == pytest.approx(1.0, abs=0.07)
        assert sl.stop - sl.start in (30, 29, 31)


def make_ready_controller() -> CalibrationController:
    c = CalibrationController(DotTiming(skip_s=0.1, window_s=0.1, tail_s=0.1), lead_in_s=1.0)
    c.on_frame(make_frame(0))
    return c


def test_controller_needs_a_camera_frame_before_starting() -> None:
    c = CalibrationController()
    assert not c.camera_ready
    with pytest.raises(RuntimeError, match="no camera frames"):
        c.start(0, build_schedule(W, H))


def test_controller_phase_progression_and_progress() -> None:
    c = make_ready_controller()
    assert c.phase == "waiting" and c.current_dot is None and c.progress(0) == 0.0
    plan = build_schedule(W, H)
    c.start(1_000, plan)
    with pytest.raises(RuntimeError, match="already started"):
        c.start(2_000, plan)
    c.update(1_000 + 999_999_999)
    assert str(c.phase) == "lead_in"
    c.update(1_000 + 1_000_000_000)
    assert str(c.phase) == "dot" and c.dot_number == 1 and c.current_dot == plan[0]
    t0 = 1_000 + 1_000_000_000
    assert c.progress(t0) == 0.0
    assert c.progress(t0 + 150_000_000) == pytest.approx(0.5)
    assert c.progress(t0 + 10**12) == 1.0
    c.update(t0 + 299_999_999)
    assert c.dot_number == 1
    c.update(t0 + 300_000_000)
    assert c.dot_number == 2


def test_controller_ignores_frames_before_start_and_after_abort() -> None:
    c = make_ready_controller()
    c.on_frame(make_frame(10))
    c.start(100, build_schedule(W, H))
    c.on_frame(make_frame(200))
    c.abort()
    assert c.phase == "aborted"
    c.on_frame(make_frame(300))
    c.update(10**12)
    assert c.phase == "aborted"
    with pytest.raises(RuntimeError, match="not complete"):
        c.build_session(Conditions(), TRUE_SCREEN)


def test_build_session_requires_completion() -> None:
    c = make_ready_controller()
    with pytest.raises(RuntimeError):
        c.build_session(Conditions(), TRUE_SCREEN)
    c.start(0, build_schedule(W, H))
    with pytest.raises(RuntimeError):
        c.build_session(Conditions(), TRUE_SCREEN)


def test_empty_schedule_rejected() -> None:
    c = make_ready_controller()
    with pytest.raises(ValueError):
        c.start(0, [])


def test_extra_points_are_the_four_inner_quadrant_centres() -> None:
    ext = extra_calibration_points(W, H)
    assert sorted((round(x / W, 3), round(y / H, 3)) for x, y in ext) == [
        (0.3, 0.3),
        (0.3, 0.7),
        (0.7, 0.3),
        (0.7, 0.7),
    ]


def test_13_point_schedule_has_roles_and_validation_avoids_all_calibration_points() -> None:
    plan = build_schedule(W, H, seed=2, n_calibration=13)
    assert [p.role for p in plan] == ["cal"] * 9 + ["ext"] * 4 + ["val"] * 16
    cal = np.array([p.position_px for p in plan if p.role != "val"])
    val = np.array([p.position_px for p in plan if p.role == "val"])
    d = np.hypot(val[:, None, 0] - cal[None, :, 0], val[:, None, 1] - cal[None, :, 1])
    assert d.min() >= 0.05 * H


def test_default_schedule_is_unchanged_by_the_new_options() -> None:
    assert build_schedule(W, H, seed=3) == build_schedule(W, H, seed=3, n_calibration=9)
    assert [p.role for p in build_schedule(W, H, n_validation=0)] == ["cal"] * 9
    with pytest.raises(ValueError, match="9 or 13"):
        build_schedule(W, H, n_calibration=5)


def test_calibration_dot_indices_for_5_9_13() -> None:
    session = synthetic_session(n_calibration=13)
    nine = calibration_dot_indices(session, 9)
    assert nine.tolist() == list(range(9))
    five = calibration_dot_indices(session, 5)
    fractions = {
        (round(session.dot_positions_px[i][0] / W, 2), round(session.dot_positions_px[i][1] / H, 2))
        for i in five
    }
    assert fractions == {(0.1, 0.1), (0.9, 0.1), (0.5, 0.5), (0.1, 0.9), (0.9, 0.9)}
    thirteen = calibration_dot_indices(session, 13)
    assert thirteen.tolist() == list(range(13))
    with pytest.raises(ValueError, match="extra calibration dots"):
        calibration_dot_indices(synthetic_session(), 13)
    with pytest.raises(ValueError, match="one of"):
        calibration_dot_indices(session, 7)


def test_fit_calibration_uses_the_requested_dots() -> None:
    session = synthetic_session(n_calibration=13)
    for points, expected in ((5, 5), (9, 9), (13, 13)):
        fit = fit_calibration(session, calibration=points)
        assert (fit.n_used, fit.n_total) == (expected, expected)
        assert fit.x.shape == (expected, 4) and fit.y.shape == (expected, 2)


def test_fit_calibration_errors() -> None:
    with pytest.raises(FitError, match="extra calibration dots"):
        fit_calibration(synthetic_session(), calibration=13)
    with pytest.raises(FitError, match="usable calibration dots"):
        fit_calibration(synthetic_session(blink_dots=frozenset(range(8))))
    with pytest.raises(ValueError, match="unknown feature set"):
        fit_calibration(synthetic_session(), feature_set="bogus")
