import math

import numpy as np
import pytest

from gazekey import config
from gazekey.core.geometry import (
    Axis,
    ScreenGeometry,
    estimate_distance_mm,
    focal_px_from_hfov,
    size_mm_for_angle,
    visual_angle_deg,
)


@pytest.fixture
def screen() -> ScreenGeometry:
    return ScreenGeometry(1440, 900, 286.0, 179.0, 600.0)


def test_one_degree_at_60cm_is_10_47_mm() -> None:
    assert size_mm_for_angle(1.0, 600.0) == pytest.approx(10.4720, abs=1e-3)


def test_size_angle_round_trip() -> None:
    for angle in (0.5, 1.0, 3.0, 10.0, 40.0):
        mm = size_mm_for_angle(angle, 550.0)
        assert visual_angle_deg(mm, 550.0) == pytest.approx(angle)


def test_known_right_angle_geometry() -> None:
    assert visual_angle_deg(1200.0, 600.0) == pytest.approx(90.0)


def test_invalid_distance_and_angle() -> None:
    with pytest.raises(ValueError):
        visual_angle_deg(10.0, 0.0)
    with pytest.raises(ValueError):
        size_mm_for_angle(180.0, 600.0)


def test_zero_distance_between_identical_points(screen: ScreenGeometry) -> None:
    p = np.array([[100.0, 200.0], [700.0, 450.0]])
    assert np.allclose(screen.angular_distance_deg(p, p), 0.0)


def test_angle_across_screen_centre_matches_hand_calculation(screen: ScreenGeometry) -> None:
    mm_per_px = 286.0 / 1440.0
    dx_px = 100.0 / mm_per_px
    left = (720.0 - dx_px, 450.0)
    right = (720.0 + dx_px, 450.0)
    expected = 2.0 * math.degrees(math.atan(100.0 / 600.0))
    assert float(screen.angular_distance_deg(left, right)) == pytest.approx(expected)


def test_angular_distance_is_symmetric_and_batched(screen: ScreenGeometry) -> None:
    rng = np.random.default_rng(0)
    a = rng.uniform([0, 0], [1440, 900], size=(50, 2))
    b = rng.uniform([0, 0], [1440, 900], size=(50, 2))
    ab = screen.angular_distance_deg(a, b)
    ba = screen.angular_distance_deg(b, a)
    assert ab.shape == (50,)
    assert np.allclose(ab, ba)


def test_off_axis_angle_is_smaller_than_on_axis_for_same_pixel_gap(screen: ScreenGeometry) -> None:
    centre = float(screen.angular_distance_deg((720.0, 450.0), (820.0, 450.0)))
    edge = float(screen.angular_distance_deg((1300.0, 450.0), (1400.0, 450.0)))
    assert edge < centre


def test_px_deg_round_trip(screen: ScreenGeometry) -> None:
    axes: tuple[Axis, Axis] = ("x", "y")
    for axis in axes:
        deg = screen.px_to_deg(80.0, axis)
        assert screen.deg_to_px(deg, axis) == pytest.approx(80.0)


def test_eye_offset_changes_angles() -> None:
    a = ScreenGeometry(1000, 800, 300.0, 240.0, 500.0)
    b = ScreenGeometry(1000, 800, 300.0, 240.0, 500.0, eye_offset_mm=(100.0, 0.0))
    p, q = (400.0, 400.0), (600.0, 400.0)
    assert float(a.angular_distance_deg(p, q)) != pytest.approx(float(b.angular_distance_deg(p, q)))


def test_screen_validation() -> None:
    with pytest.raises(ValueError):
        ScreenGeometry(0, 900, 286.0, 179.0, 600.0)
    with pytest.raises(ValueError):
        ScreenGeometry(1440, 900, -1.0, 179.0, 600.0)
    with pytest.raises(ValueError):
        ScreenGeometry(1440, 900, 286.0, 179.0, 0.0)
    with pytest.raises(ValueError):
        ScreenGeometry(1440, 900, 286.0, 179.0, 600.0).gaze_vectors_mm([1.0, 2.0, 3.0])


def test_iris_distance_estimate_round_trip() -> None:
    focal = focal_px_from_hfov(1280, 70.0)
    true_distance = 550.0
    diameter_px = focal * config.IRIS_DIAMETER_MM / true_distance
    assert estimate_distance_mm(diameter_px, focal) == pytest.approx(true_distance)


def test_focal_and_distance_validation() -> None:
    with pytest.raises(ValueError):
        focal_px_from_hfov(1280, 0.0)
    with pytest.raises(ValueError):
        estimate_distance_mm(0.0, 1000.0)
