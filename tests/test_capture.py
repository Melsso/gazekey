from typing import Any

import numpy as np
import pytest

from gazekey.vision.capture import Camera, CameraError


class FakeCapture:
    def __init__(self, script: list[bool], opened: bool = True) -> None:
        self.script = list(script)
        self.opened = opened
        self.reads = 0
        self.props: dict[int, float] = {}
        self.released = 0

    def isOpened(self) -> bool:
        return self.opened

    def read(self) -> tuple[bool, Any]:
        self.reads += 1
        ok = self.script.pop(0) if self.script else True
        return ok, (np.zeros((4, 4, 3), dtype=np.uint8) if ok else None)

    def set(self, prop_id: int, value: float, /) -> bool:
        self.props[prop_id] = value
        return True

    def release(self) -> None:
        self.released += 1


class TickClock:
    """Each call advances time by `step_ns`."""

    def __init__(self, step_ns: int = 10_000_000) -> None:
        self.t = 0
        self.step = step_ns

    def __call__(self) -> int:
        self.t += self.step
        return self.t


def make_camera(cap: FakeCapture, clock: TickClock | None = None, **kw: Any) -> Camera:
    return Camera(0, 640, 480, clock_ns=clock or TickClock(), capture_factory=lambda _i: cap, **kw)


def test_open_sets_resolution_and_discards_warmup() -> None:
    cap = FakeCapture([])
    make_camera(cap, warmup_frames=5).open()
    assert cap.reads == 5
    assert cap.props == {3: 640, 4: 480}


def test_open_failure_raises_and_releases() -> None:
    cap = FakeCapture([], opened=False)
    with pytest.raises(CameraError, match="camera 0"):
        make_camera(cap).open()
    assert cap.released == 1


def test_read_stamps_after_the_read_using_injected_clock() -> None:
    clock = TickClock(step_ns=7)
    cam = make_camera(FakeCapture([]), clock, warmup_frames=0)
    cam.open()
    frame = cam.read()
    assert frame is not None
    assert frame.t_ns == 7
    frame2 = cam.read()
    assert frame2 is not None and frame2.t_ns == 14


def test_transient_failures_return_none_then_recover() -> None:
    cam = make_camera(FakeCapture([False, False, True]), warmup_frames=0, max_read_failures=5)
    cam.open()
    assert cam.read() is None
    assert cam.read() is None
    assert cam.read() is not None


def test_too_many_consecutive_failures_raise() -> None:
    cam = make_camera(FakeCapture([False] * 10), warmup_frames=0, max_read_failures=3)
    cam.open()
    assert cam.read() is None
    assert cam.read() is None
    with pytest.raises(CameraError, match="stopped delivering"):
        cam.read()


def test_success_resets_failure_counter() -> None:
    cam = make_camera(
        FakeCapture([False, False, True, False, False, True]), warmup_frames=0, max_read_failures=3
    )
    cam.open()
    results = [cam.read() for _ in range(6)]
    assert [r is not None for r in results] == [False, False, True, False, False, True]


def test_read_before_open_raises() -> None:
    with pytest.raises(CameraError, match="not open"):
        make_camera(FakeCapture([])).read()


def test_frames_stops_at_timeout_using_injected_clock() -> None:
    clock = TickClock(step_ns=10_000_000)
    cam = make_camera(FakeCapture([]), clock, warmup_frames=0)
    cam.open()
    got = list(cam.frames(timeout=0.1))
    assert 1 <= len(got) < 10
    assert all(b.t_ns > a.t_ns for a, b in zip(got, got[1:], strict=False))


def test_context_manager_releases_and_release_is_idempotent() -> None:
    cap = FakeCapture([])
    with make_camera(cap, warmup_frames=0) as cam:
        assert cam.read() is not None
    assert cap.released == 1
    cam.release()
    assert cap.released == 1
