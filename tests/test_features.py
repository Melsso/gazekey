from typing import Any

import numpy as np
import pytest

from gazekey.core import F64
from gazekey.core.features import (
    FEATURE_NAMES,
    IRIS_FEATURES,
    POSE_FEATURES,
    SCALE_FEATURES,
    compute_features,
    feature_index,
    head_pose_features,
    valid_mask,
)
from gazekey.core.landmarks import LEFT_EYE, NUM_LANDMARKS, RIGHT_EYE
from tests.synthetic import make_points, make_transform


def feats(**kwargs: Any) -> F64:
    out: F64 = compute_features(make_points(**kwargs)[None])[0]
    return out


def test_feature_layout_is_consistent() -> None:
    assert len(FEATURE_NAMES) == 16
    assert len(set(FEATURE_NAMES)) == 16
    assert IRIS_FEATURES + POSE_FEATURES + SCALE_FEATURES == FEATURE_NAMES
    assert compute_features(make_points()[None]).shape == (1, len(FEATURE_NAMES))


def test_landmark_indices_are_sane() -> None:
    for eye in (RIGHT_EYE, LEFT_EYE):
        assert eye.iris_ring == tuple(range(eye.iris_center + 1, eye.iris_center + 5))
        assert all(0 <= i < NUM_LANDMARKS for i in (*eye.iris_ring, eye.iris_center))
    assert (RIGHT_EYE.iris_center, LEFT_EYE.iris_center) == (468, 473)


def test_centred_iris_gives_zero_h_and_v() -> None:
    f = feats()
    for name in ("r_h", "r_v_corner", "l_h", "l_v_corner"):
        assert f[feature_index(name)] == pytest.approx(0.0, abs=1e-9)


def test_recovers_known_iris_offsets() -> None:
    f = feats(h_r=0.4, v_r=-0.1, h_l=-0.3, v_l=0.15)
    assert f[feature_index("r_h")] == pytest.approx(0.4)
    assert f[feature_index("r_v_corner")] == pytest.approx(-0.1)
    assert f[feature_index("l_h")] == pytest.approx(-0.3)
    assert f[feature_index("l_v_corner")] == pytest.approx(0.15)


def test_v_lid_and_openness_match_construction() -> None:
    open_ratio, half_w = 0.3, 30.0
    gap = open_ratio * 60.0
    f = feats(v_r=0.2, open_ratio=open_ratio)
    assert f[feature_index("r_open")] == pytest.approx(open_ratio)
    assert f[feature_index("r_v_lid")] == pytest.approx((0.2 * half_w + gap / 2) / gap)


def test_features_are_invariant_to_roll() -> None:
    base = feats(h_r=0.35, v_r=0.1, h_l=0.3, v_l=-0.05)
    rolled = feats(h_r=0.35, v_r=0.1, h_l=0.3, v_l=-0.05, roll_deg=17.0)
    for name in IRIS_FEATURES:
        assert rolled[feature_index(name)] == pytest.approx(base[feature_index(name)], abs=1e-9)


def test_features_are_invariant_to_image_scale_except_pixel_sizes() -> None:
    base = feats(h_r=0.2, v_l=0.1)
    big = feats(h_r=0.2, v_l=0.1, scale=1.7)
    for name in IRIS_FEATURES:
        assert big[feature_index(name)] == pytest.approx(base[feature_index(name)], abs=1e-9)
    assert big[feature_index("iris_diam")] == pytest.approx(1.7 * base[feature_index("iris_diam")])
    assert big[feature_index("interocular")] == pytest.approx(
        1.7 * base[feature_index("interocular")]
    )


def test_scale_features_match_construction() -> None:
    f = feats(iris_diam=22.0, ipd=200.0)
    assert f[feature_index("iris_diam")] == pytest.approx(22.0)
    assert f[feature_index("interocular")] == pytest.approx(200.0)


def test_pose_columns_are_nan_without_transforms() -> None:
    f = feats()
    assert np.isnan(f[[feature_index(n) for n in POSE_FEATURES]]).all()
    assert np.isfinite(f[[feature_index(n) for n in IRIS_FEATURES]]).all()


@pytest.mark.parametrize(
    ("yaw", "pitch", "roll"),
    [(0, 0, 0), (25, 0, 0), (-30, 10, 5), (12, -20, 15), (60, 25, -40)],
)
def test_head_pose_round_trip(yaw: float, pitch: float, roll: float) -> None:
    m = make_transform(yaw, pitch, roll, (1.0, -2.0, 45.0), scale=1.0)
    out = head_pose_features(m[None], 1)[0]
    assert out[:3] == pytest.approx([yaw, pitch, roll], abs=1e-9)
    assert out[3:] == pytest.approx([1.0, -2.0, 45.0])


def test_head_pose_ignores_uniform_scale() -> None:
    a = head_pose_features(make_transform(20, -5, 3, (0, 0, 40), scale=1.0)[None], 1)
    b = head_pose_features(make_transform(20, -5, 3, (0, 0, 40), scale=2.5)[None], 1)
    assert np.allclose(a, b)


def test_head_pose_accepts_transposed_matrices() -> None:
    m = make_transform(18, 7, -9, (3.0, 4.0, 50.0))
    normal = head_pose_features(m[None], 1)
    transposed = head_pose_features(m.T[None], 1)
    assert np.allclose(normal, transposed)


def test_head_pose_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        head_pose_features(np.eye(4)[None], 2)


def test_batch_matches_single_frame_and_handles_nan_rows() -> None:
    a = make_points(h_r=0.1)
    b = make_points(h_l=-0.2, v_r=0.05)
    nan_face = np.full((NUM_LANDMARKS, 3), np.nan)
    batch = compute_features(np.stack([a, nan_face, b]))
    assert batch.shape == (3, 16)
    assert np.allclose(batch[0], compute_features(a[None])[0], equal_nan=True)
    assert np.allclose(batch[2], compute_features(b[None])[0], equal_nan=True)
    assert np.isnan(batch[1]).all()
    assert list(valid_mask(batch)) == [True, False, True]


def test_blink_is_invalid() -> None:
    open_f = compute_features(make_points(open_ratio=0.3)[None])
    closed_f = compute_features(make_points(open_ratio=0.02)[None])
    assert valid_mask(open_f)[0]
    assert not valid_mask(closed_f)[0]


def test_degenerate_eye_width_gives_nan_not_error() -> None:
    p = make_points()
    p[RIGHT_EYE.corner_img_right] = p[RIGHT_EYE.corner_img_left]
    f = compute_features(p[None])[0]
    assert np.isnan(f[feature_index("r_h")])
    assert not valid_mask(f[None])[0]


def test_bad_input_shape_raises() -> None:
    with pytest.raises(ValueError):
        compute_features(np.zeros((5, 3)))
    with pytest.raises(ValueError):
        compute_features(np.zeros((1, 468, 3)))
