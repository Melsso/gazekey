import math

import numpy as np
import pytest

from gazekey.core.filters import GazeFilter, OneEuroFilter

NS = 1_000_000_000
STEP = round(NS / 30)


def alpha(cutoff: float, dt_s: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt_s)


def test_first_sample_passes_through_and_constant_input_is_unchanged() -> None:
    f = OneEuroFilter(min_cutoff=1.0, beta=0.5)
    assert f.update(0, [3.0, 4.0]).tolist() == [3.0, 4.0]
    for i in range(1, 50):
        assert f.update(i * STEP, [3.0, 4.0]) == pytest.approx([3.0, 4.0])


def test_first_order_step_response_matches_hand_value() -> None:
    f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    f.update(0, [0.0, 0.0])
    out = f.update(STEP, [1.0, 0.0])
    assert alpha(1.0, 1 / 30) == pytest.approx(0.17317, abs=1e-4)
    assert out[0] == pytest.approx(alpha(1.0, STEP * 1e-9))
    assert out[1] == 0.0


def test_speed_dependent_cutoff_matches_reference_formula() -> None:
    f = OneEuroFilter(min_cutoff=1.0, beta=0.5, d_cutoff=2.0)
    f.update(0, [0.0, 0.0])
    out = f.update(STEP, [1.0, 0.0])
    dt = STEP * 1e-9
    dx_hat = alpha(2.0, dt) * (1.0 / dt)
    cutoff = 1.0 + 0.5 * dx_hat
    assert out[0] == pytest.approx(alpha(cutoff, dt) * 1.0)


def test_beta_reduces_lag_on_fast_motion() -> None:
    def final_lag(beta: float) -> float:
        f = OneEuroFilter(min_cutoff=0.8, beta=beta)
        out = np.zeros(2)
        for i in range(60):
            x = 1000.0 * i / 30
            out = f.update(i * STEP, [x, 0.0])
        return 1000.0 * 59 / 30 - float(out[0])

    assert final_lag(0.0) > 5 * final_lag(0.05) > 0


def test_smoothing_reduces_jitter_at_rest() -> None:
    rng = np.random.default_rng(0)
    raw = np.array([500.0, 400.0]) + rng.normal(0.0, 20.0, size=(300, 2))
    f = OneEuroFilter(min_cutoff=0.8, beta=0.005)
    out = np.array([f.update(i * STEP, p) for i, p in enumerate(raw)])
    ratio = out[60:].std(axis=0) / raw[60:].std(axis=0)
    assert (ratio < 0.6).all()


def test_duplicate_and_backwards_timestamps_keep_output() -> None:
    f = OneEuroFilter()
    f.update(STEP, [0.0, 0.0])
    first = f.update(2 * STEP, [10.0, 0.0])
    assert f.update(2 * STEP, [99.0, 0.0]).tolist() == first.tolist()
    assert f.update(STEP, [99.0, 0.0]).tolist() == first.tolist()


def test_reset_and_validation() -> None:
    f = OneEuroFilter()
    f.update(0, [0.0, 0.0])
    f.update(STEP, [5.0, 5.0])
    f.reset()
    assert f.update(2 * STEP, [100.0, 100.0]).tolist() == [100.0, 100.0]
    for kwargs in ({"min_cutoff": 0.0}, {"d_cutoff": -1.0}, {"beta": -0.1}):
        with pytest.raises(ValueError):
            OneEuroFilter(**kwargs)


def test_gaze_filter_holds_through_short_dropouts_then_reports_lost() -> None:
    g = GazeFilter(hold_s=0.4)
    assert g.update(0, None) == (None, False)
    point, held = g.update(0, [10.0, 20.0])
    assert point is not None and point.tolist() == [10.0, 20.0] and not held
    held_point, held = g.update(round(0.2 * NS), None)
    assert held and held_point is not None and held_point.tolist() == [10.0, 20.0]
    assert g.update(round(0.5 * NS), None) == (None, False)


def test_gaze_filter_restarts_smoothing_after_a_long_gap() -> None:
    g = GazeFilter(reset_gap_s=1.0)
    g.update(0, [0.0, 0.0])
    near, _ = g.update(STEP, [100.0, 100.0])
    assert near is not None and near[0] < 100.0
    g2 = GazeFilter(reset_gap_s=1.0)
    g2.update(0, [0.0, 0.0])
    far, _ = g2.update(2 * NS, [100.0, 100.0])
    assert far is not None and far.tolist() == [100.0, 100.0]
