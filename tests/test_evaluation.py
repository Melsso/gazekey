import numpy as np
import pytest

from gazekey.core.calibration import DotTiming
from gazekey.core.evaluation import EvaluationError, evaluate_session, summarise
from gazekey.core.session import Conditions, SessionRecorder
from tests.synthetic import TRUE_SCREEN, make_frame, synthetic_session


def test_noiseless_session_is_recovered_almost_exactly() -> None:
    result = evaluate_session(synthetic_session())
    s = result.summary
    assert (result.n_cal_used, result.n_val_used) == (9, 16)
    assert s.mean_offset_deg < 0.02
    assert s.precision_rms_deg < 0.02 and s.precision_sd_deg < 0.02
    assert s.no_face_pct == 0.0
    assert s.size_for_hit_rate(0.9) < 0.1
    assert result.cal_offsets_deg.max() < 0.02


def test_noise_shows_up_as_precision_and_bounded_accuracy() -> None:
    clean = evaluate_session(synthetic_session()).summary
    noisy = evaluate_session(synthetic_session(noise=0.01)).summary
    assert noisy.precision_rms_deg > 10 * max(clean.precision_rms_deg, 1e-4)
    assert noisy.precision_sd_deg > 0.05
    assert 0.0 < noisy.mean_offset_deg < 0.6
    assert noisy.size_for_hit_rate(0.9) > clean.size_for_hit_rate(0.9)

    curve = noisy.hit_rates(list(range(1, 11)))
    assert np.all(np.diff(curve) >= 0) and curve[-1] == 1.0


def test_skip_window_keeps_saccades_out_of_precision() -> None:
    session = synthetic_session(latency_s=0.3)
    skipped = evaluate_session(session).summary
    unskipped = evaluate_session(session, timing=DotTiming(skip_s=0.0, window_s=1.0)).summary
    assert skipped.precision_rms_deg < 0.02
    assert unskipped.precision_rms_deg > 0.2
    assert unskipped.size_for_hit_rate(0.9) > 10 * skipped.size_for_hit_rate(0.9)


def test_accuracy_breaks_when_most_of_the_window_is_pre_saccade() -> None:
    session = synthetic_session(latency_s=0.8)
    bad = evaluate_session(session, timing=DotTiming(skip_s=0.0, window_s=1.0)).summary
    ok = evaluate_session(session, timing=DotTiming(skip_s=0.9, window_s=0.6)).summary
    assert bad.mean_offset_deg > 1.0
    assert ok.mean_offset_deg < 0.02


def test_mapper_is_fitted_on_calibration_dots_only() -> None:
    clean = evaluate_session(synthetic_session())
    biased = evaluate_session(synthetic_session(val_bias=0.15))
    x = np.random.default_rng(0).normal(size=(10, 4)) * 0.3
    assert np.array_equal(clean.fitted.predict(x), biased.fitted.predict(x))
    assert np.array_equal(clean.cal_offsets_deg, biased.cal_offsets_deg)
    assert biased.summary.mean_offset_deg > 1.0
    assert clean.summary.mean_offset_deg < 0.02


def test_dots_without_data_are_dropped_and_reported() -> None:
    result = evaluate_session(synthetic_session(blink_dots=frozenset({2, 10})))
    assert (result.n_cal_used, result.n_cal_total) == (8, 9)
    assert (result.n_val_used, result.n_val_total) == (15, 16)
    assert result.summary.no_face_pct > 5.0
    assert result.summary.not_valid_pct >= result.summary.no_face_pct


def test_too_few_calibration_dots_raises() -> None:
    session = synthetic_session(blink_dots=frozenset(range(7)))
    with pytest.raises(EvaluationError, match="calibration dots"):
        evaluate_session(session)


def test_no_usable_validation_dots_raises() -> None:
    session = synthetic_session(blink_dots=frozenset(range(9, 25)))
    with pytest.raises(EvaluationError, match="validation"):
        evaluate_session(session)


def test_session_without_screen_or_dots_raises() -> None:
    rec = SessionRecorder((1280, 720))
    for i in range(5):
        rec.add_frame(make_frame(i * 33_000_000))
    with pytest.raises(EvaluationError, match="screen geometry"):
        evaluate_session(rec.build())
    with pytest.raises(EvaluationError, match="both calibration and validation"):
        evaluate_session(rec.build(screen=TRUE_SCREEN))


def test_feature_sets_and_models_run_end_to_end() -> None:
    session = synthetic_session(noise=0.005)
    for model in ("ridge", "poly2", "poly3"):
        r = evaluate_session(session, mapper=model)
        assert r.mapper_name == model and np.isfinite(r.summary.mean_offset_deg)

    pose = evaluate_session(session, feature_set="iris+pose")
    assert np.isfinite(pose.summary.mean_offset_deg)
    with pytest.raises(ValueError, match="unknown feature set"):
        evaluate_session(session, feature_set="bogus")


def test_pooled_summary_combines_sessions_and_participants() -> None:
    a = evaluate_session(synthetic_session(noise=0.01, rng_seed=1))
    b = evaluate_session(synthetic_session(noise=0.01, rng_seed=2, conditions=Conditions("p2")))
    pooled = summarise([a, b])
    assert pooled.n_sessions == 2 and pooled.n_val_dots == 32
    assert pooled.cheb_deg.size == a.val_cheb_deg.size + b.val_cheb_deg.size
    lo, hi = sorted([a.summary.mean_offset_deg, b.summary.mean_offset_deg])
    assert lo <= pooled.mean_offset_deg <= hi
    assert pooled.mean_distance_mm == 600.0
    assert pooled.size_cm(10.0) == pytest.approx(2 * 60 * np.tan(np.radians(5.0)), rel=1e-9)
    with pytest.raises(ValueError):
        summarise([])


def test_calibration_point_variants_on_a_13_point_session() -> None:
    session = synthetic_session(n_calibration=13)
    for k in (5, 9, 13):
        r = evaluate_session(session, calibration=k)
        assert r.calibration_points == k
        assert (r.n_cal_used, r.n_cal_total) == (k, k)
        assert r.summary.mean_offset_deg < 0.05
    assert evaluate_session(session).n_cal_total == 9


def test_13_points_beat_5_points_under_noise_on_average() -> None:
    def mean_error(k: int) -> float:
        errors = []
        for seed in range(6):
            s = synthetic_session(n_calibration=13, noise=0.03, rng_seed=seed)
            errors.append(evaluate_session(s, calibration=k).summary.mean_offset_deg)
        return float(np.mean(errors))

    assert mean_error(13) < mean_error(5)


def test_13_point_variant_needs_extra_dots() -> None:
    with pytest.raises(EvaluationError, match="extra calibration dots"):
        evaluate_session(synthetic_session(), calibration=13)
    with pytest.raises(EvaluationError, match="one of"):
        evaluate_session(synthetic_session(), calibration=7)
