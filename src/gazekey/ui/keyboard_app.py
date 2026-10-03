import queue
import signal
import time
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QKeyEvent, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import QApplication, QWidget

from ..core.keyboard import DwellSelector, DwellState, KeyboardModel
from ..core.landmarks import LandmarkFrame
from ..core.tracking import GazeTracker
from .calibration_app import CaptureWorker, FrameStream

Outcome = Literal["running", "done", "error"]

BACKGROUND = QColor(20, 20, 20)
KEY = QColor(52, 56, 66)
KEY_ACTIVE = QColor(78, 96, 140)
TEXT = QColor(225, 225, 225)
PROGRESS = QColor(255, 200, 70)
GAZE = QColor(255, 60, 60, 190)


class KeyboardError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class KeyboardResult:
    text: str


def _font(points: int) -> QFont:
    font = QFont()
    font.setPointSize(points)
    return font


class KeyboardWindow(QWidget):
    def __init__(
        self,
        tracker: GazeTracker,
        frames: "queue.SimpleQueue[LandmarkFrame]",
        worker: CaptureWorker,
        *,
        origin_pt: tuple[int, int],
        dwell: DwellSelector,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        tick_ms: int = 8,
    ) -> None:
        super().__init__()
        self._tracker = tracker
        self._frames = frames
        self._worker = worker
        self._origin = origin_pt
        self._dwell = dwell
        self._clock_ns = clock_ns
        self.model = KeyboardModel(float(self.width()), float(self.height()))
        self._state = DwellState(None, 0.0, None)
        self._gaze: QPointF | None = None
        self.outcome: Outcome = "running"
        self.error = ""

        self.setWindowTitle("gazekey keyboard")
        self.setCursor(Qt.CursorShape.BlankCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._timer = QTimer(self)
        self._timer.setInterval(tick_ms)
        self._timer.timeout.connect(self.tick)
        self._timer.start()

    def _sync_size(self) -> None:
        self.model.width = float(self.width())
        self.model.height = float(self.height())

    def tick(self) -> None:
        if self.outcome != "running":
            return
        self._sync_size()
        while True:
            try:
                frame = self._frames.get_nowait()
            except queue.Empty:
                break
            self._on_frame(frame)
        if self._worker.error is not None:
            self._finish("error", self._worker.error)
            return
        self.update()

    def _on_frame(self, frame: LandmarkFrame) -> None:
        estimate = self._tracker.update(frame)
        key_id = None
        self._gaze = None
        if estimate.point_px is not None:
            self._gaze = self.mapFromGlobal(
                QPointF(
                    self._origin[0] + estimate.point_px[0], self._origin[1] + estimate.point_px[1]
                )
            )
            key_id = self.model.key_at((self._gaze.x(), self._gaze.y()))
        self._state = self._dwell.update(frame.t_ns, key_id)
        if self._state.selected is not None:
            self.model.press(self._state.selected)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Escape:
            self.stop()

    def stop(self) -> None:
        self._finish("done")

    def _finish(self, outcome: Outcome, error: str = "") -> None:
        self._timer.stop()
        self.outcome, self.error = outcome, error
        self.close()

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        self._sync_size()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), BACKGROUND)
        self._paint_text(painter)
        self._paint_keys(painter)
        self._paint_gaze(painter)
        painter.end()

    def _paint_text(self, painter: QPainter) -> None:
        top = min(k.rect.y for k in self.model.keys)
        painter.setPen(TEXT)
        painter.setFont(_font(28))
        painter.drawText(
            QRectF(0, 0, float(self.width()), top),
            Qt.AlignmentFlag.AlignCenter,
            self.model.text[-40:] + "_",
        )

    def _paint_keys(self, painter: QPainter) -> None:
        for key in self.model.keys:
            r = QRectF(key.rect.x, key.rect.y, key.rect.w, key.rect.h).adjusted(6, 6, -6, -6)
            active = self._state.key_id == key.id
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(KEY_ACTIVE if active else KEY)
            painter.drawRoundedRect(r, 12.0, 12.0)
            painter.setPen(TEXT)
            painter.setFont(_font(30))
            painter.drawText(r.adjusted(0, 0, 0, -56), Qt.AlignmentFlag.AlignCenter, key.label)
            if active and self._state.progress > 0.0:
                centre = QPointF(r.center().x(), r.bottom() - 30)
                ring = QRectF(centre.x() - 18, centre.y() - 18, 36, 36)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QPen(PROGRESS, 6))
                painter.drawArc(ring, 90 * 16, -round(self._state.progress * 360 * 16))

    def _paint_gaze(self, painter: QPainter) -> None:
        if self._gaze is None:
            return
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(GAZE)
        painter.drawEllipse(self._gaze, 10.0, 10.0)


def run_keyboard(
    *,
    tracker: GazeTracker,
    landmarker_path: Path,
    camera_index: int | None,
    origin_pt: tuple[int, int],
    dwell: DwellSelector,
    frame_stream: FrameStream | None = None,
    clock_ns: Callable[[], int] = time.monotonic_ns,
) -> KeyboardResult:
    app = QApplication.instance() or QApplication([])
    frames: queue.SimpleQueue[LandmarkFrame] = queue.SimpleQueue()
    with ExitStack() as stack:
        stream = frame_stream
        if stream is None:
            from ..vision.capture import Camera
            from ..vision.landmarker import MediaPipeLandmarkSource, stream_landmarks

            camera = stack.enter_context(Camera(camera_index))
            source = stack.enter_context(MediaPipeLandmarkSource(landmarker_path))

            def stream(should_stop: Callable[[], bool]) -> Iterator[LandmarkFrame]:
                return stream_landmarks(camera, source, should_stop)

        worker = CaptureWorker(stream, frames)
        window = KeyboardWindow(
            tracker,
            frames,
            worker,
            origin_pt=origin_pt,
            dwell=dwell,
            clock_ns=clock_ns,
        )
        previous_handler = signal.signal(signal.SIGINT, lambda *_: window.stop())
        worker.start()
        window.showFullScreen()
        window.activateWindow()
        window.setFocus()
        try:
            app.exec()
        finally:
            signal.signal(signal.SIGINT, previous_handler)
            worker.stop()
            worker.join(timeout=5.0)

    if window.outcome == "error":
        raise KeyboardError(window.error)
    return KeyboardResult(window.model.text)
