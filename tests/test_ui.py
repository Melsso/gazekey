import os
import queue
import threading
import time
from collections.abc import Callable, Iterator

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)
from PySide6.QtCore import QCoreApplication, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from gazekey.core.calibration import CalibrationController, DotTiming  # noqa: E402
from gazekey.core.landmarks import LandmarkFrame  # noqa: E402
from gazekey.core.session import Conditions  # noqa: E402
from gazekey.ui.calibration_app import (  # noqa: E402
    CalibrationWindow,
    CaptureWorker,
    ScreenInfo,
)
from tests.synthetic import make_frame  # noqa: E402

INFO = ScreenInfo("test", 800, 600, 286.0, 214.5, 2.0, "override")
FAST = DotTiming(skip_s=0.05, window_s=0.1, tail_s=0.05)


@pytest.fixture(scope="module")
def app() -> QCoreApplication:
    existing = QApplication.instance()
    return existing if existing is not None else QApplication([])


class FakeClock:
    def __init__(self) -> None:
        self.t = 5_000_000_000

    def __call__(self) -> int:
        return self.t


def make_window(
    app: QCoreApplication, worker: CaptureWorker | None = None
) -> tuple[CalibrationWindow, CalibrationController, FakeClock, "queue.SimpleQueue[LandmarkFrame]"]:
    clock = FakeClock()
    frames: queue.SimpleQueue[LandmarkFrame] = queue.SimpleQueue()
    controller = CalibrationController(FAST, lead_in_s=0.1)
    worker = worker or CaptureWorker(lambda stop: iter(()), frames)
    window = CalibrationWindow(
        controller,
        frames,
        worker,
        info=INFO,
        distance_mm=600.0,
        seed=0,
        clock_ns=clock,
    )
    window.resize(800, 600)
    window.show()
    return window, controller, clock, frames


def test_full_run_with_fake_clock_reaches_done_and_paints_every_phase(
    app: QCoreApplication,
) -> None:
    window, controller, clock, frames = make_window(app)
    assert not window.grab().isNull()

    window.tick()
    QTest.keyClick(window, Qt.Key.Key_Space)
    assert controller.phase == "waiting"

    frames.put(make_frame(clock(), image_size=(640, 480)))
    window.tick()
    assert controller.camera_ready
    QTest.keyClick(window, Qt.Key.Key_Space)
    assert str(controller.phase) == "lead_in"
    assert window.screen_geometry is not None
    assert window.screen_geometry.width_px == 800
    assert window.screen_geometry.mm_per_px[0] == pytest.approx(286.0 / 800.0)

    seen_phases = set()
    for _ in range(2000):
        clock.t += 20_000_000
        frames.put(make_frame(clock(), image_size=(640, 480)))
        window.tick()
        seen_phases.add(controller.phase)
        if window.outcome != "running":
            break
        assert not window.grab().isNull()
    assert window.outcome == "done"
    assert {"lead_in", "dot"} <= seen_phases
    session = controller.build_session(Conditions(), window.screen_geometry)
    assert session.n_dots == 25 and session.n_frames > 100


def test_escape_aborts_and_discards(app: QCoreApplication) -> None:
    window, controller, clock, frames = make_window(app)
    frames.put(make_frame(clock(), image_size=(640, 480)))
    window.tick()
    QTest.keyClick(window, Qt.Key.Key_Space)
    QTest.keyClick(window, Qt.Key.Key_Escape)
    assert window.outcome == "aborted" and controller.phase == "aborted"


def test_worker_error_is_surfaced_and_stops_the_run(app: QCoreApplication) -> None:
    frames: queue.SimpleQueue[LandmarkFrame] = queue.SimpleQueue()

    def broken(stop: object) -> Iterator[LandmarkFrame]:
        raise RuntimeError("camera exploded")
        yield  # pragma: no cover

    worker = CaptureWorker(broken, frames)
    worker.start()
    worker.join(timeout=2.0)
    window, _, _, _ = make_window(app, worker)
    window.tick()
    assert window.outcome == "error" and "camera exploded" in window.error


def test_capture_worker_streams_until_stopped() -> None:
    frames: queue.SimpleQueue[LandmarkFrame] = queue.SimpleQueue()
    started = threading.Event()

    def stream(should_stop: Callable[[], bool]) -> Iterator[LandmarkFrame]:
        i = 0
        while not should_stop():
            started.set()
            yield make_frame(i * 1_000_000)
            i += 1
            time.sleep(0.001)

    worker = CaptureWorker(stream, frames)
    worker.start()
    assert started.wait(timeout=2.0)
    worker.stop()
    worker.join(timeout=2.0)
    assert not worker.is_alive() and worker.error is None
    assert not frames.empty()
