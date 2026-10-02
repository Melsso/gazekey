import math

import numpy as np
import pytest

from gazekey.core.metrics import (
    axis_offsets_deg,
    chebyshev_deg,
    hit_rates,
    radial_sd_deg,
    rms,
    sample_to_sample_deg,
    size_for_hit_rate,
    summarise_ms,
)
from tests.synthetic import TRUE_SCREEN

MM_PER_PX = 286.0 / 1440.0
CENTRE = (720.0, 450.0)


def shifted_px(dx_mm: float = 0.0, dy_mm: float = 0.0) -> tuple[float, float]:
    return CENTRE[0] + dx_mm / MM_PER_PX, CENTRE[1] + dy_mm / (179.0 / 900.0)


def test_rms() -> None:
    assert rms([3.0, 4.0]) == pytest.approx(math.sqrt(12.5))
    assert math.isnan(rms([]))


def test_axis_offsets_match_hand_geometry() -> None:
    off = axis_offsets_deg(TRUE_SCREEN, [shifted_px(dx_mm=100.0)], CENTRE)
    assert off[0, 0] == pytest.approx(math.degrees(math.atan(100.0 / 600.0)))
    assert off[0, 1] == pytest.approx(0.0, abs=1e-9)
    off_y = axis_offsets_deg(TRUE_SCREEN, [shifted_px(dy_mm=-50.0)], CENTRE)
    assert off_y[0, 0] == pytest.approx(0.0, abs=1e-9)
    assert off_y[0, 1] == pytest.approx(math.degrees(math.atan(50.0 / 600.0)))


def test_chebyshev_is_max_of_axes_and_broadcasts_target() -> None:
    pred = np.array([shifted_px(dx_mm=30.0, dy_mm=60.0), shifted_px(dx_mm=-90.0, dy_mm=10.0)])
    cheb = chebyshev_deg(TRUE_SCREEN, pred, CENTRE)
    assert cheb.shape == (2,)
    assert cheb[0] == pytest.approx(math.degrees(math.atan(60.0 / 600.0)), rel=1e-6)
    assert cheb[1] == pytest.approx(math.degrees(math.atan(90.0 / 600.0)), rel=1e-6)


def test_hit_rates_hand_values() -> None:
    cheb = [0.5, 1.0, 1.5, 2.0, 4.0]
    assert hit_rates(cheb, [1, 2, 3, 4, 8]).tolist() == [0.2, 0.4, 0.6, 0.8, 1.0]
    with pytest.raises(ValueError):
        hit_rates([], [1.0])


def test_size_for_hit_rate_hand_values_and_bounds() -> None:
    cheb = np.arange(1.0, 11.0)
    assert size_for_hit_rate(cheb, 0.9) == pytest.approx(18.0)
    assert size_for_hit_rate(cheb, 1.0) == pytest.approx(20.0)
    assert size_for_hit_rate(cheb, 0.1) == pytest.approx(2.0)
    for bad in (0.0, 1.5):
        with pytest.raises(ValueError):
            size_for_hit_rate(cheb, bad)
    with pytest.raises(ValueError):
        size_for_hit_rate([], 0.9)


def test_size_for_hit_rate_is_tight() -> None:
    cheb = np.random.default_rng(0).gamma(2.0, 1.0, size=500)
    s90 = size_for_hit_rate(cheb, 0.9)
    assert hit_rates(cheb, [s90])[0] >= 0.9
    assert hit_rates(cheb, [s90 - 1e-6])[0] < 0.9


def test_sample_to_sample_and_adjacency() -> None:
    xs = np.array([shifted_px(dx_mm=d) for d in (0.0, 10.0, 20.0, 50.0)])
    full = sample_to_sample_deg(TRUE_SCREEN, xs)
    assert full.shape == (3,)
    assert full[0] == pytest.approx(math.degrees(math.atan(10 / 600)), rel=1e-2)

    gapped = sample_to_sample_deg(TRUE_SCREEN, xs, frame_idx=[0, 1, 2, 5])
    assert np.allclose(gapped, full[:2])
    assert sample_to_sample_deg(TRUE_SCREEN, xs[:1]).size == 0


def test_radial_sd_hand_value_and_degenerate() -> None:
    pts = np.array([shifted_px(dx_mm=-20.0), shifted_px(dx_mm=20.0)])
    a = math.degrees(math.atan(20.0 / 600.0))
    assert radial_sd_deg(TRUE_SCREEN, pts) == pytest.approx(a * math.sqrt(2.0), rel=1e-6)
    assert math.isnan(radial_sd_deg(TRUE_SCREEN, pts[:1]))
    assert radial_sd_deg(TRUE_SCREEN, [CENTRE, CENTRE, CENTRE]) == pytest.approx(0.0)


def test_summarise_ms() -> None:
    assert summarise_ms([]) == "no data"
    text = summarise_ms([10.0] * 20 + [50.0])
    assert "median 10.0 ms" in text and "n=21" in text
