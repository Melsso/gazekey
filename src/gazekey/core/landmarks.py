from dataclasses import dataclass
from typing import Protocol, Self

import numpy as np

from . import F32, U8

NUM_LANDMARKS = 478


@dataclass(frozen=True, slots=True)
class EyeIndices:
    iris_center: int
    iris_ring: tuple[int, int, int, int]
    corner_img_left: int
    corner_img_right: int
    lid_upper: int
    lid_lower: int


RIGHT_EYE = EyeIndices(
    iris_center=468,
    iris_ring=(469, 470, 471, 472),
    corner_img_left=33,
    corner_img_right=133,
    lid_upper=159,
    lid_lower=145,
)

LEFT_EYE = EyeIndices(
    iris_center=473,
    iris_ring=(474, 475, 476, 477),
    corner_img_left=362,
    corner_img_right=263,
    lid_upper=386,
    lid_lower=374,
)


class LandmarkError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True, eq=False)
class LandmarkFrame:
    t_ns: int
    image_size: tuple[int, int]
    points: F32 | None
    transform: F32 | None

    @property
    def valid(self) -> bool:
        return self.points is not None


def normalized_to_pixels(points: F32, image_size: tuple[int, int]) -> F32:
    width, height = image_size
    scale = np.array([width, height, width], dtype=np.float32)
    out: F32 = points * scale
    return out


class LandmarkSource(Protocol):
    def process(self, image: U8, t_ns: int) -> LandmarkFrame: ...

    def close(self) -> None: ...

    def __enter__(self) -> Self: ...

    def __exit__(self, *exc: object) -> None: ...
