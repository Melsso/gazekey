import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

from . import F64
from .geometry import ScreenGeometry


def rms(values: ArrayLike) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(arr**2))) if arr.size else float("nan")


def axis_offsets_deg(screen: ScreenGeometry, pred_px: ArrayLike, target_px: ArrayLike) -> F64:
    pred = np.asarray(pred_px, dtype=np.float64)
    target = np.broadcast_to(np.asarray(target_px, dtype=np.float64), pred.shape)
    same_row = np.stack([pred[..., 0], target[..., 1]], axis=-1)
    same_col = np.stack([target[..., 0], pred[..., 1]], axis=-1)
    out: F64 = np.stack(
        [
            screen.angular_distance_deg(target, same_row),
            screen.angular_distance_deg(target, same_col),
        ],
        axis=-1,
    )
    return out


def chebyshev_deg(screen: ScreenGeometry, pred_px: ArrayLike, target_px: ArrayLike) -> F64:
    out: F64 = axis_offsets_deg(screen, pred_px, target_px).max(axis=-1)
    return out


def hit_rates(cheb_deg: ArrayLike, sizes_deg: Sequence[float]) -> F64:
    cheb = np.asarray(cheb_deg, dtype=np.float64)
    if cheb.size == 0:
        raise ValueError("no samples")
    half = np.asarray(sizes_deg, dtype=np.float64)[:, None] / 2.0
    out: F64 = (cheb[None, :] <= half + 1e-12).mean(axis=1)
    return out


def size_for_hit_rate(cheb_deg: ArrayLike, rate: float = 0.9) -> float:
    if not 0.0 < rate <= 1.0:
        raise ValueError("rate must be in (0, 1]")
    cheb = np.sort(np.asarray(cheb_deg, dtype=np.float64))
    if cheb.size == 0:
        raise ValueError("no samples")
    k = max(math.ceil(rate * cheb.size - 1e-9), 1) - 1
    return float(2.0 * cheb[k])


def sample_to_sample_deg(
    screen: ScreenGeometry, points_px: ArrayLike, frame_idx: ArrayLike | None = None
) -> F64:
    pts = np.asarray(points_px, dtype=np.float64)
    if len(pts) < 2:
        return np.empty(0)
    d: F64 = screen.angular_distance_deg(pts[:-1], pts[1:])
    if frame_idx is not None:
        adjacent = np.diff(np.asarray(frame_idx)) == 1
        d = d[adjacent]
    return d


def radial_sd_deg(screen: ScreenGeometry, points_px: ArrayLike) -> float:
    pts = np.asarray(points_px, dtype=np.float64)
    if len(pts) < 2:
        return float("nan")
    d = screen.angular_distance_deg(pts, pts.mean(axis=0))
    return float(np.sqrt(np.sum(d**2) / (len(pts) - 1)))


def summarise_ms(values_ms: Sequence[float]) -> str:
    if len(values_ms) == 0:
        return "no data"
    arr = np.asarray(values_ms, dtype=np.float64)
    return f"median {np.median(arr):.1f} ms, p95 {np.percentile(arr, 95):.1f} ms (n={arr.size})"
