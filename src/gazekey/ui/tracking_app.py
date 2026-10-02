import queue
import signal
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Literal

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QGuiApplication, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import QApplication, QWidget

from ..core.landmarks import LandmarkFrame
from ..core.tracking import GazeEstimate, GazeTracker, TrackingStats
from .calibration_app import CaptureWorker, FrameStream

DOT = QColor(255, 60, 60, 190)
DOT_HELD = QColor(255, 170, 40, 150)
OUTLINE = QColor(255, 255, 255, 230)
RAW = QColor(60, 140, 255, 200)
RADIUS = 14.0
RAW_RADIUS = 4.0


class TrackingError(RuntimeError):
    pass


class TrackWindow(QWidget):
    def __init__(
        self,
        tracker: GazeTracker,
        frames: "queue.SimpleQueue[LandmarkFrame]",
        worker: CaptureWorker,
        *,
        origin_pt: tuple[int, int],
        show_raw: bool = False,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        tick_ms: int = 8,
    ) -> None:
        flags = (
            Qt.WindowType.Window
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.WindowTransparentForInput
        )
        super().__init__(None, flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setWindowTitle("gazekey overlay")

        self._tracker = tracker
        self._frames = frames
        self._worker = worker
        self._origin = origin_pt
        self._show_raw = show_raw
        self._clock_ns = clock_ns
        self.stats = TrackingStats()
        self.latest: GazeEstimate | None = None
        self.outcome: Literal["running", "done", "error"] = "running"
        self.error = ""
        self._dirty = QRectF()

        self._timer = QTimer(self)
        self._timer.setInterval(tick_ms)
        self._timer.timeout.connect(self.tick)
        self._timer.start()

    def tick(self) -> None:
        if self.outcome != "running":
            return
        while True:
            try:
                frame = self._frames.get_nowait()
            except queue.Empty:
                break
            estimate = self._tracker.update(frame)
            self.stats.record(frame, estimate, (self._clock_ns() - frame.t_ns) / 1e6)
            self.latest = estimate
            self._invalidate(estimate)
        if self._worker.error is not None:
            self.finish("error", self._worker.error)

    def finish(self, outcome: Literal["done", "error"] = "done", error: str = "") -> None:
        self._timer.stop()
        self.outcome, self.error = outcome, error
        self.close()

    def local_point(self, point_px: tuple[float, float]) -> QPointF:
        return self.mapFromGlobal(
            QPointF(self._origin[0] + point_px[0], self._origin[1] + point_px[1])
        )

    def _invalidate(self, estimate: GazeEstimate) -> None:
        margin = RADIUS + 6.0
        new = QRectF()
        for p in (estimate.point_px, estimate.raw_px if self._show_raw else None):
            if p is not None:
                c = self.local_point(p)
                new = new.united(QRectF(c.x() - margin, c.y() - margin, 2 * margin, 2 * margin))
        self.update(self._dirty.united(new).toAlignedRect())
        self._dirty = new

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
        painter.fillRect(event.rect(), Qt.GlobalColor.transparent)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        est = self.latest
        if est is not None:
            if self._show_raw and est.raw_px is not None:
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(RAW)
                painter.drawEllipse(self.local_point(est.raw_px), RAW_RADIUS, RAW_RADIUS)
            if est.point_px is not None:
                painter.setPen(QPen(OUTLINE, 2))
                painter.setBrush(DOT_HELD if est.held else DOT)
                painter.drawEllipse(self.local_point(est.point_px), RADIUS, RADIUS)
        painter.end()


def run_tracking(
    *,
    tracker: GazeTracker,
    landmarker_path: Path,
    camera_index: int | None,
    origin_pt: tuple[int, int],
    seconds: float | None = None,
    show_raw: bool = False,
    frame_stream: FrameStream | None = None,
    clock_ns: Callable[[], int] = time.monotonic_ns,
) -> TrackingStats:
    from contextlib import ExitStack

    app = QApplication.instance() or QApplication([])
    screen = QGuiApplication.primaryScreen()
    if screen is None:
        raise TrackingError("no display available")

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
        window = TrackWindow(
            tracker, frames, worker, origin_pt=origin_pt, show_raw=show_raw, clock_ns=clock_ns
        )
        window.setGeometry(screen.geometry())
        previous_handler = signal.signal(signal.SIGINT, lambda *_: window.finish())
        if seconds is not None:
            QTimer.singleShot(round(seconds * 1000), window.finish)
        worker.start()
        window.show()
        try:
            app.exec()
        finally:
            signal.signal(signal.SIGINT, previous_handler)
            worker.stop()
            worker.join(timeout=5.0)

    if window.outcome == "error":
        raise TrackingError(window.error)
    return window.stats
