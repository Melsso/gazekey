import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from .. import config
from . import F64

Axis = Literal["x", "y"]


def visual_angle_deg(size_mm: float, distance_mm: float) -> float:
    if distance_mm <= 0:
        raise ValueError("distance_mm must be positive")
    return math.degrees(2.0 * math.atan(size_mm / (2.0 * distance_mm)))


def size_mm_for_angle(angle_deg: float, distance_mm: float) -> float:
    if distance_mm <= 0:
        raise ValueError("distance_mm must be positive")
    if not 0.0 <= angle_deg < 180.0:
        raise ValueError("angle_deg must be in [0, 180)")
    return 2.0 * distance_mm * math.tan(math.radians(angle_deg) / 2.0)


@dataclass(frozen=True, slots=True)
class ScreenGeometry:
    width_px: int
    height_px: int
    width_mm: float
    height_mm: float
    viewing_distance_mm: float
    eye_offset_mm: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        if self.width_px <= 0 or self.height_px <= 0:
            raise ValueError("screen size in pixels must be positive")
        if self.width_mm <= 0 or self.height_mm <= 0:
            raise ValueError("screen size in mm must be positive")
        if self.viewing_distance_mm <= 0:
            raise ValueError("viewing_distance_mm must be positive")

    @property
    def mm_per_px(self) -> tuple[float, float]:
        return self.width_mm / self.width_px, self.height_mm / self.height_px

    def gaze_vectors_mm(self, points_px: ArrayLike) -> F64:
        p = np.asarray(points_px, dtype=np.float64)
        if p.shape[-1] != 2:
            raise ValueError("points must have last dimension 2 (x, y)")
        mm_x, mm_y = self.mm_per_px
        x = (p[..., 0] - self.width_px / 2.0) * mm_x - self.eye_offset_mm[0]
        y = (p[..., 1] - self.height_px / 2.0) * mm_y - self.eye_offset_mm[1]
        z = np.full_like(x, self.viewing_distance_mm)
        out: F64 = np.stack([x, y, z], axis=-1)
        return out

    def angular_distance_deg(self, a_px: ArrayLike, b_px: ArrayLike) -> F64:
        va = self.gaze_vectors_mm(a_px)
        vb = self.gaze_vectors_mm(b_px)
        cross = np.linalg.norm(np.cross(va, vb), axis=-1)
        dot = np.sum(va * vb, axis=-1)
        out: F64 = np.degrees(np.arctan2(cross, dot))
        return out

    def px_to_deg(self, size_px: float, axis: Axis = "x") -> float:
        return visual_angle_deg(size_px * self._mm_per_px(axis), self.viewing_distance_mm)

    def deg_to_px(self, angle_deg: float, axis: Axis = "x") -> float:
        """Size in pixels of a centred segment that subtends `angle_deg`."""
        return size_mm_for_angle(angle_deg, self.viewing_distance_mm) / self._mm_per_px(axis)

    def _mm_per_px(self, axis: Axis) -> float:
        mm_x, mm_y = self.mm_per_px
        return mm_x if axis == "x" else mm_y


def focal_px_from_hfov(image_width_px: int, hfov_deg: float) -> float:
    if not 0.0 < hfov_deg < 180.0:
        raise ValueError("hfov_deg must be in (0, 180)")
    return (image_width_px / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)


def estimate_distance_mm(iris_diameter_px: float, focal_px: float) -> float:
    if iris_diameter_px <= 0 or focal_px <= 0:
        raise ValueError("iris_diameter_px and focal_px must be positive")
    return focal_px * config.IRIS_DIAMETER_MM / iris_diameter_px
