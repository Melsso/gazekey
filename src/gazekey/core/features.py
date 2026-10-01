from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import F32, F64
from .landmarks import LEFT_EYE, NUM_LANDMARKS, RIGHT_EYE, EyeIndices, normalized_to_pixels

FEATURE_NAMES: tuple[str, ...] = (
    "r_h",
    "r_v_corner",
    "r_v_lid",
    "r_open",
    "l_h",
    "l_v_corner",
    "l_v_lid",
    "l_open",
    "yaw",
    "pitch",
    "roll",
    "tx",
    "ty",
    "tz",
    "iris_diam",
    "interocular",
)
IRIS_FEATURES = FEATURE_NAMES[:8]
POSE_FEATURES = FEATURE_NAMES[8:14]
SCALE_FEATURES = FEATURE_NAMES[14:]

DEFAULT_OPEN_THRESHOLD = 0.10

_EPS = 1e-9


def feature_index(name: str) -> int:
    return FEATURE_NAMES.index(name)


def _safe_div(num: F64, den: F64, eps: float = _EPS) -> F64:
    out: F64 = np.full(np.broadcast(num, den).shape, np.nan)
    np.divide(num, den, out=out, where=np.abs(den) > eps)
    return out


def eye_features(points_px: NDArray[Any], eye: EyeIndices) -> F64:
    p = np.asarray(points_px, dtype=np.float64)[..., :2]
    left = p[:, eye.corner_img_left]
    right = p[:, eye.corner_img_right]
    iris = p[:, eye.iris_center]
    upper = p[:, eye.lid_upper]
    lower = p[:, eye.lid_lower]

    axis = right - left
    width = np.hypot(axis[:, 0], axis[:, 1])
    u = np.stack([_safe_div(axis[:, 0], width), _safe_div(axis[:, 1], width)], axis=-1)
    n = np.stack([-u[:, 1], u[:, 0]], axis=-1)

    centre = (left + right) / 2.0
    half = width / 2.0
    rel = iris - centre
    h = _safe_div(np.sum(rel * u, axis=-1), half)
    s_iris = np.sum(rel * n, axis=-1)
    s_upper = np.sum((upper - centre) * n, axis=-1)
    s_lower = np.sum((lower - centre) * n, axis=-1)
    gap = s_lower - s_upper

    v_corner = _safe_div(s_iris, half)
    v_lid = _safe_div(s_iris - s_upper, gap)
    openness = _safe_div(gap, width)
    return np.stack([h, v_corner, v_lid, openness], axis=-1)


def _max_abs_finite(x: F64) -> float:
    finite = x[np.isfinite(x)]
    return float(np.abs(finite).max()) if finite.size else 0.0


def head_pose_features(transforms: NDArray[Any] | None, n_frames: int) -> F64:
    if transforms is None:
        return np.full((n_frames, 6), np.nan)
    m = np.asarray(transforms, dtype=np.float64)
    if m.shape != (n_frames, 4, 4):
        raise ValueError(f"transforms must have shape ({n_frames}, 4, 4), got {m.shape}")
    if _max_abs_finite(m[:, 3, :3]) > 1e-4 and _max_abs_finite(m[:, :3, 3]) <= 1e-4:
        m = m.transpose(0, 2, 1)

    rot = m[:, :3, :3]
    with np.errstate(invalid="ignore", divide="ignore"):
        rot = rot / np.linalg.norm(rot, axis=1)[:, None, :]
    pitch = np.arcsin(np.clip(-rot[:, 1, 2], -1.0, 1.0))
    yaw = np.arctan2(rot[:, 0, 2], rot[:, 2, 2])
    roll = np.arctan2(rot[:, 1, 0], rot[:, 1, 1])
    angles = np.degrees(np.stack([yaw, pitch, roll], axis=-1))
    out: F64 = np.concatenate([angles, m[:, :3, 3]], axis=-1)
    return out


def _iris_diameter_px(p: F64, eye: EyeIndices) -> F64:
    r = eye.iris_ring
    d_a = np.linalg.norm(p[:, r[0]] - p[:, r[2]], axis=-1)
    d_b = np.linalg.norm(p[:, r[1]] - p[:, r[3]], axis=-1)
    out: F64 = (d_a + d_b) / 2.0
    return out


def scale_features(points_px: NDArray[Any]) -> F64:
    p = np.asarray(points_px, dtype=np.float64)[..., :2]
    diameter = (_iris_diameter_px(p, RIGHT_EYE) + _iris_diameter_px(p, LEFT_EYE)) / 2.0
    interocular = np.linalg.norm(p[:, RIGHT_EYE.iris_center] - p[:, LEFT_EYE.iris_center], axis=-1)
    return np.stack([diameter, interocular], axis=-1)


def compute_features(points_px: NDArray[Any], transforms: NDArray[Any] | None = None) -> F64:
    pts = np.asarray(points_px, dtype=np.float64)
    if pts.ndim != 3 or pts.shape[1] != NUM_LANDMARKS or pts.shape[2] < 2:
        raise ValueError(f"points_px must have shape (N, {NUM_LANDMARKS}, 2|3), got {pts.shape}")
    out: F64 = np.concatenate(
        [
            eye_features(pts, RIGHT_EYE),
            eye_features(pts, LEFT_EYE),
            head_pose_features(transforms, pts.shape[0]),
            scale_features(pts),
        ],
        axis=1,
    )
    return out


def features_from_normalized(
    points: F32, image_size: tuple[int, int], transforms: NDArray[Any] | None = None
) -> F64:
    return compute_features(normalized_to_pixels(points, image_size), transforms)


def valid_mask(features: F64, min_openness: float = DEFAULT_OPEN_THRESHOLD) -> NDArray[np.bool_]:
    """True for frames usable for gaze: all iris features finite and both eyes open."""
    finite = np.isfinite(features[:, : len(IRIS_FEATURES)]).all(axis=1)
    open_cols = [feature_index("r_open"), feature_index("l_open")]
    eyes_open = (features[:, open_cols] >= min_openness).all(axis=1)
    out: NDArray[np.bool_] = finite & eyes_open
    return out
