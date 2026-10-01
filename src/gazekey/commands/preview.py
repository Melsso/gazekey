import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from .. import config
from ..core.features import FEATURE_NAMES, compute_features, feature_index, valid_mask
from ..core.landmarks import LEFT_EYE, RIGHT_EYE, LandmarkFrame, normalized_to_pixels
from ..core.metrics import RateMeter, summarise_ms
from ..core.session import Conditions, SessionRecorder, save_session
from ..vision.capture import Camera, CameraError
from ..vision.landmarker import MediaPipeLandmarkSource

WINDOW = "gazekey preview (q / Esc to quit)"
GREEN, BLUE, YELLOW, RED = (0, 255, 0), (255, 128, 0), (0, 255, 255), (0, 0, 255)


def status_lines(
    feats: np.ndarray[Any, Any] | None,
    fps: float | None,
    latency_ms: float,
    landmark_ms: float,
    recording: bool,
) -> list[str]:
    lines = [
        f"capture {fps:.1f} fps" if fps is not None else "capture -- fps",
        f"latency capture->display {latency_ms:.0f} ms (landmarks {landmark_ms:.0f} ms)",
    ]
    if recording:
        lines.append("REC landmarks (no video)")
    if feats is None:
        lines.append("no face")
        return lines
    ok = bool(valid_mask(feats[None])[0])
    lines.append("gaze-valid" if ok else "INVALID (blink / eyes closed)")
    for name in FEATURE_NAMES:
        lines.append(f"{name:>11}: {feats[feature_index(name)]:8.3f}")
    return lines


def _draw(
    image: np.ndarray[Any, Any], frame: LandmarkFrame, lines: list[str]
) -> np.ndarray[Any, Any]:
    import cv2

    width = image.shape[1]
    canvas = np.ascontiguousarray(image[:, ::-1])

    if frame.points is not None:
        px = normalized_to_pixels(frame.points, frame.image_size)
        for eye in (RIGHT_EYE, LEFT_EYE):
            ring = [px[i] for i in eye.iris_ring]
            radius = max(2, int(round(np.linalg.norm(ring[0][:2] - ring[2][:2]) / 2)))
            cx, cy = px[eye.iris_center][:2]
            cv2.circle(canvas, (width - 1 - int(cx), int(cy)), radius, GREEN, 1, cv2.LINE_AA)
            cv2.circle(canvas, (width - 1 - int(cx), int(cy)), 2, GREEN, -1, cv2.LINE_AA)
            for idx, color in (
                (eye.corner_img_left, BLUE),
                (eye.corner_img_right, BLUE),
                (eye.lid_upper, YELLOW),
                (eye.lid_lower, YELLOW),
            ):
                x, y = px[idx][:2]
                cv2.circle(canvas, (width - 1 - int(x), int(y)), 2, color, -1, cv2.LINE_AA)

    for row, text in enumerate(lines):
        color = RED if text.startswith(("INVALID", "no face")) else GREEN
        cv2.putText(
            canvas, text, (10, 22 + 18 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
        )
    return canvas


def run_preview(
    *,
    camera_index: int | None,
    landmarker_path: Path,
    record_path: Path | None,
    seconds: float | None,
    clock_ns: Callable[[], int] = time.monotonic_ns,
) -> int:
    import cv2

    if not landmarker_path.is_file():
        print(
            f"error: FaceLandmarker model not found at {landmarker_path}. "
            "Run: uv run python scripts/download_model.py",
            file=sys.stderr,
        )
        return 1

    rate = RateMeter()
    latency_ms: list[float] = []
    landmark_ms: list[float] = []
    n_frames = n_face = n_valid = 0
    recorder: SessionRecorder | None = None

    try:
        with Camera(camera_index) as camera, MediaPipeLandmarkSource(landmarker_path) as source:
            for frame in camera.frames(timeout=seconds):
                t0 = clock_ns()
                lm = source.process(frame.image, frame.t_ns)
                lm_ms = (clock_ns() - t0) / 1e6
                fps = rate.update(frame.t_ns)

                feats = None
                if lm.points is not None:
                    px = normalized_to_pixels(lm.points, lm.image_size)
                    tf = None if lm.transform is None else lm.transform[None]
                    feats = compute_features(px[None], tf)[0]
                    n_face += 1
                    n_valid += int(valid_mask(feats[None])[0])

                if record_path is not None:
                    if recorder is None:
                        recorder = SessionRecorder(lm.image_size)
                    recorder.add_frame(lm)

                lat_ms = (clock_ns() - frame.t_ns) / 1e6
                lines = status_lines(feats, fps, lat_ms, lm_ms, recorder is not None)
                cv2.imshow(WINDOW, _draw(frame.image, lm, lines))
                latency_ms.append(lat_ms)
                landmark_ms.append(lm_ms)
                n_frames += 1
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    except (CameraError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        cv2.destroyAllWindows()

    _print_summary(n_frames, n_face, n_valid, rate.rate, latency_ms, landmark_ms)
    if record_path is not None and recorder is not None:
        session = recorder.build(conditions=Conditions())
        saved = save_session(session, record_path)
        print(f"saved {session.n_frames} frames of landmarks (no video) to {saved}")
    return 0


def _print_summary(
    n_frames: int,
    n_face: int,
    n_valid: int,
    fps: float | None,
    latency_ms: list[float],
    landmark_ms: list[float],
) -> None:
    if n_frames == 0:
        print("no frames processed")
        return
    print("--- preview summary (paste this back) ---")
    print(
        f"frames: {n_frames}   face found: {100 * n_face / n_frames:.1f}%   "
        f"gaze-valid: {100 * n_valid / n_frames:.1f}%   data loss: "
        f"{100 * (1 - n_face / n_frames):.1f}%"
    )
    print(f"capture rate (smoothed, last): {fps:.1f} fps" if fps else "capture rate: n/a")
    print(f"latency capture->display: {summarise_ms(latency_ms)}")
    print(f"landmark inference: {summarise_ms(landmark_ms)}")
    print(
        f"camera default index {config.default_camera_index()}, "
        f"{config.CAMERA_WIDTH}x{config.CAMERA_HEIGHT} requested"
    )
