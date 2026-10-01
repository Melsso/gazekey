import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Protocol, Self

from .. import config
from ..core import U8

_PROP_FRAME_WIDTH = 3
_PROP_FRAME_HEIGHT = 4


class CameraError(RuntimeError):
    pass


class VideoCaptureLike(Protocol):
    def isOpened(self) -> bool: ...  # noqa: N802
    def read(self) -> tuple[bool, Any]: ...
    def set(self, prop_id: int, value: float, /) -> bool: ...
    def release(self) -> None: ...


def open_avfoundation(index: int) -> VideoCaptureLike:
    import cv2

    cap: VideoCaptureLike = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    return cap


@dataclass(frozen=True, slots=True)
class Frame:
    image: U8
    t_ns: int


class Camera:
    def __init__(
        self,
        index: int | None = None,
        width: int = config.CAMERA_WIDTH,
        height: int = config.CAMERA_HEIGHT,
        *,
        warmup_frames: int = config.CAMERA_WARMUP_FRAMES,
        max_read_failures: int = config.CAMERA_MAX_READ_FAILURES,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        capture_factory: Callable[[int], VideoCaptureLike] = open_avfoundation,
    ) -> None:
        self.index = config.default_camera_index() if index is None else index
        self.width = width
        self.height = height
        self._warmup_frames = warmup_frames
        self._max_read_failures = max_read_failures
        self._clock_ns = clock_ns
        self._capture_factory = capture_factory
        self._cap: VideoCaptureLike | None = None
        self._failures = 0

    def open(self) -> None:
        if self._cap is not None:
            return
        cap = self._capture_factory(self.index)
        if not cap.isOpened():
            cap.release()
            raise CameraError(
                f"Could not open camera {self.index}. Check the index "
                f"({config.CAMERA_ENV_VAR}) and macOS camera permissions."
            )
        cap.set(_PROP_FRAME_WIDTH, self.width)
        cap.set(_PROP_FRAME_HEIGHT, self.height)
        self._cap = cap
        self._failures = 0
        for _ in range(self._warmup_frames):
            cap.read()

    def read(self) -> Frame | None:
        if self._cap is None:
            raise CameraError("Camera is not open.")
        ok, image = self._cap.read()
        t_ns = self._clock_ns()
        if not ok:
            self._failures += 1
            if self._failures >= self._max_read_failures:
                raise CameraError("Camera stopped delivering frames.")
            return None
        self._failures = 0
        return Frame(image=image, t_ns=t_ns)

    def frames(self, timeout: float | None = None) -> Iterator[Frame]:
        """Yield frames until `timeout` seconds pass (or forever if None)."""
        deadline_ns = None if timeout is None else self._clock_ns() + int(timeout * 1e9)
        while deadline_ns is None or self._clock_ns() < deadline_ns:
            frame = self.read()
            if frame is not None:
                yield frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> Self:
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()
