import queue
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import (
    QColor,
    QFont,
    QGuiApplication,
    QKeyEvent,
    QPainter,
    QPaintEvent,
    QPen,
    QScreen,
)
from PySide6.QtWidgets import QApplication, QWidget

from ..core.calibration import DEFAULT_TIMING, CalibrationController, DotTiming, build_schedule
from ..core.geometry import ScreenGeometry
from ..core.landmarks import LandmarkFrame
from ..core.session import Conditions, Session

FrameStream = Callable[[Callable[[], bool]], Iterator[LandmarkFrame]]
Outcome = Literal["running", "done", "aborted", "error"]

BACKGROUND = QColor(20, 20, 20)
TEXT = QColor(200, 200, 200)
WARN = QColor(255, 120, 90)
RING = QColor(240, 240, 240)
CENTER = QColor(255, 70, 70)


class CalibrationError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class CalibrationRun:
    session: Session
    origin_pt: tuple[int, int]


@dataclass(frozen=True, slots=True)
class ScreenInfo:
    name: str
    width_pt: int
    height_pt: int
    width_mm: float
    height_mm: float
    pixel_ratio: float
    mm_source: str

    def geometry(self, width_pt: int, height_pt: int, distance_mm: float) -> ScreenGeometry:
        return ScreenGeometry(
            width_px=width_pt,
            height_px=height_pt,
            width_mm=width_pt * self.width_mm / self.width_pt,
            height_mm=height_pt * self.height_mm / self.height_pt,
            viewing_distance_mm=distance_mm,
        )

    def describe(self) -> str:
        return (
            f"display '{self.name}': {self.width_pt}x{self.height_pt} points "
            f"(pixel ratio {self.pixel_ratio:g}), {self.width_mm:.0f}x{self.height_mm:.0f} mm "
            f"[{self.mm_source}]. Check the size against your Mac's spec; override with -S WxH."
        )


def read_screen_info(screen: QScreen, override_mm: tuple[float, float] | None = None) -> ScreenInfo:
    size, physical = screen.size(), screen.physicalSize()
    width_mm, height_mm = override_mm or (physical.width(), physical.height())
    if width_mm <= 0 or height_mm <= 0:
        raise CalibrationError("could not read the physical screen size; pass -S WIDTHxHEIGHT (mm)")
    return ScreenInfo(
        name=screen.name(),
        width_pt=size.width(),
        height_pt=size.height(),
        width_mm=float(width_mm),
        height_mm=float(height_mm),
        pixel_ratio=float(screen.devicePixelRatio()),
        mm_source="override" if override_mm else "Qt/macOS",
    )


class CaptureWorker(threading.Thread):
    def __init__(self, stream: FrameStream, out: "queue.SimpleQueue[LandmarkFrame]") -> None:
        super().__init__(daemon=True, name="gazekey-capture")
        self._stream = stream
        self._out = out
        self._stop_event = threading.Event()
        self.error: str | None = None

    def run(self) -> None:
        try:
            for frame in self._stream(self._stop_event.is_set):
                self._out.put(frame)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    def stop(self) -> None:
        self._stop_event.set()


class CalibrationWindow(QWidget):
    def __init__(
        self,
        controller: CalibrationController,
        frames: "queue.SimpleQueue[LandmarkFrame]",
        worker: CaptureWorker,
        *,
        info: ScreenInfo,
        distance_mm: float,
        seed: int,
        n_calibration: int = 9,
        n_validation: int = 16,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        tick_ms: int = 10,
    ) -> None:
        super().__init__()
        self._controller = controller
        self._frames = frames
        self._worker = worker
        self._info = info
        self._distance_mm = distance_mm
        self._seed = seed
        self._n_calibration = n_calibration
        self._n_validation = n_validation
        self._clock_ns = clock_ns
        self.origin_pt: tuple[int, int] = (0, 0)
        self.outcome: Outcome = "running"
        self.error = ""
        self.screen_geometry: ScreenGeometry | None = None

        self.setWindowTitle("gazekey calibration")
        self.setCursor(Qt.CursorShape.BlankCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
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
            self._controller.on_frame(frame)
        if self._worker.error is not None:
            self._finish("error", self._worker.error)
            return
        self._controller.update(self._clock_ns())
        if self._controller.phase == "done":
            self._finish("done")
            return
        self.update()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Escape:
            self._controller.abort()
            self._finish("aborted")
        elif event.key() == Qt.Key.Key_Space and self._controller.phase == "waiting":
            self._start()

    def _start(self) -> None:
        if not self._controller.camera_ready:
            return
        w, h = self.width(), self.height()
        self.screen_geometry = self._info.geometry(w, h, self._distance_mm)

        origin = self.mapToGlobal(QPoint(0, 0))
        self.origin_pt = (origin.x(), origin.y())
        schedule = build_schedule(
            w,
            h,
            seed=self._seed,
            n_calibration=self._n_calibration,
            n_validation=self._n_validation,
        )
        self._controller.start(self._clock_ns(), schedule)

    def _finish(self, outcome: Outcome, error: str = "") -> None:
        self._timer.stop()
        self.outcome, self.error = outcome, error
        self.close()

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), BACKGROUND)
        phase = self._controller.phase
        if phase == "waiting":
            self._paint_intro(painter)
        elif phase == "lead_in":
            self._paint_text(painter, ["Get ready", "Look at each dot until it disappears"], TEXT)
        elif (dot := self._controller.current_dot) is not None:
            self._paint_dot(painter, dot.position_px)
        painter.end()

    def _paint_text(self, painter: QPainter, lines: list[str], color: QColor) -> None:
        painter.setPen(color)
        font = QFont()
        font.setPointSize(18)
        painter.setFont(font)
        line_h = 34.0
        top = self.height() / 2 - line_h * len(lines) / 2
        for i, text in enumerate(lines):
            rect = QRectF(0, top + i * line_h, self.width(), line_h)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

    def _paint_intro(self, painter: QPainter) -> None:
        c = self._controller
        n = self._n_calibration + self._n_validation
        if not c.camera_ready:
            status, color = "waiting for camera frames...", WARN
        elif not c.face_visible:
            status, color = "NO FACE DETECTED: move into view", WARN
        else:
            status, color = "face found: ready", TEXT
        self._paint_text(
            painter,
            [
                "gazekey calibration",
                "Sit at your normal distance. Look at each dot until it disappears.",
                f"{n} dots, about {round(n * DEFAULT_TIMING.total_s + 1.5)} seconds. "
                "Do not chase dots before they appear.",
                status,
                "Space: start     Esc: abort",
            ],
            color,
        )

    def _paint_dot(self, painter: QPainter, position: tuple[float, float]) -> None:
        progress = self._controller.progress(self._clock_ns())
        r_max = min(self.width(), self.height()) * 0.04
        radius = r_max * (1.0 - 0.85 * progress)
        centre = QPointF(*position)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(RING, 2))
        painter.drawEllipse(centre, radius, radius)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(CENTER)
        painter.drawEllipse(centre, 4.0, 4.0)


def run_calibration(
    *,
    conditions: Conditions,
    landmarker_path: Path,
    camera_index: int | None,
    distance_mm: float,
    seed: int,
    n_calibration: int = 9,
    n_validation: int = 16,
    screen_mm_override: tuple[float, float] | None = None,
    report: Callable[[str], None] = print,
    frame_stream: FrameStream | None = None,
    clock_ns: Callable[[], int] = time.monotonic_ns,
    timing: DotTiming = DEFAULT_TIMING,
) -> CalibrationRun | None:
    app = QApplication.instance() or QApplication(sys.argv[:1])
    screen = QGuiApplication.primaryScreen()
    if screen is None:
        raise CalibrationError("no display available")
    info = read_screen_info(screen, screen_mm_override)
    report(info.describe())
    report(f"viewing distance {distance_mm / 10:g} cm (as given; measure it with a ruler)")

    frames: queue.SimpleQueue[LandmarkFrame] = queue.SimpleQueue()
    controller = CalibrationController(timing)
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
        window = CalibrationWindow(
            controller,
            frames,
            worker,
            info=info,
            distance_mm=distance_mm,
            seed=seed,
            n_calibration=n_calibration,
            n_validation=n_validation,
            clock_ns=clock_ns,
        )
        worker.start()
        window.showFullScreen()
        window.activateWindow()
        window.setFocus()
        try:
            app.exec()
        finally:
            worker.stop()
            worker.join(timeout=5.0)

    if window.outcome == "error":
        raise CalibrationError(window.error)
    if window.outcome != "done" or window.screen_geometry is None:
        return None
    session = controller.build_session(conditions, window.screen_geometry)
    return CalibrationRun(session, window.origin_pt)
