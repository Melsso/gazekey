import os
import queue

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)
from PySide6.QtCore import QCoreApplication, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from gazekey.core.keyboard import DwellSelector  # noqa: E402
from gazekey.core.landmarks import LandmarkFrame  # noqa: E402
from gazekey.core.tracking import build_tracker  # noqa: E402
from gazekey.ui.calibration_app import CaptureWorker  # noqa: E402
from gazekey.ui.keyboard_app import KeyboardWindow  # noqa: E402
from tests.synthetic import gaze_features, make_frame, synthetic_session  # noqa: E402

STEP = round(1e9 / 30)
POINT = (120.0, 220.0)


@pytest.fixture(scope="module")
def app() -> QCoreApplication:
    existing = QApplication.instance()
    return existing if existing is not None else QApplication([])


def make_window(
    app: QCoreApplication,
) -> tuple[KeyboardWindow, "queue.SimpleQueue[LandmarkFrame]"]:
    frames: queue.SimpleQueue[LandmarkFrame] = queue.SimpleQueue()
    tracker, _ = build_tracker(synthetic_session())
    worker = CaptureWorker(lambda stop: iter(()), frames)
    window = KeyboardWindow(tracker, frames, worker, origin_pt=(0, 0), dwell=DwellSelector())
    window.resize(1440, 900)
    window.show()
    return window, frames


def feed(
    window: KeyboardWindow, frames: "queue.SimpleQueue[LandmarkFrame]", n: int, start: int = 0
) -> None:
    for i in range(n):
        frames.put(make_frame((start + i) * STEP, **gaze_features(*POINT)))
    window.tick()


def test_looking_at_a_key_types_it(app: QCoreApplication) -> None:
    window, frames = make_window(app)
    assert not window.grab().isNull()
    feed(window, frames, 90)
    assert window.model.text and set(window.model.text) == {"A"}
    assert window.outcome == "running"
    assert not window.grab().isNull()


def test_escape_ends_the_keyboard(app: QCoreApplication) -> None:
    window, _ = make_window(app)
    QTest.keyClick(window, Qt.Key.Key_Escape)
    assert window.outcome == "done"
