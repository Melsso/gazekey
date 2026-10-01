from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, Self

import numpy as np

from ..core import F32, U8
from ..core.landmarks import NUM_LANDMARKS, LandmarkError, LandmarkFrame
from .capture import Camera


class MediaPipeLandmarkSource:
    def __init__(
        self,
        model_path: Path,
        *,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(
                f"FaceLandmarker model not found at {model_path}. "
                "Run: uv run python scripts/download_model.py"
            )
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,
        )
        self._mp: Any = mp
        self._landmarker: Any = vision.FaceLandmarker.create_from_options(options)
        self._last_ts_ms = -1

    def process(self, image: U8, t_ns: int) -> LandmarkFrame:
        if self._landmarker is None:
            raise LandmarkError("Landmark source is closed.")
        height, width = image.shape[:2]
        rgb = np.ascontiguousarray(image[:, :, ::-1])
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)

        ts_ms = max(t_ns // 1_000_000, self._last_ts_ms + 1)
        self._last_ts_ms = ts_ms
        result = self._landmarker.detect_for_video(mp_image, ts_ms)

        if not result.face_landmarks:
            return LandmarkFrame(t_ns, (width, height), None, None)

        face = result.face_landmarks[0]
        if len(face) != NUM_LANDMARKS:
            raise LandmarkError(
                f"Expected {NUM_LANDMARKS} landmarks (with iris), got {len(face)}. "
                "Is this the face_landmarker.task model?"
            )
        points: F32 = np.array([(p.x, p.y, p.z) for p in face], dtype=np.float32)

        transform: F32 | None = None
        matrices = result.facial_transformation_matrixes
        if matrices is not None and len(matrices) > 0:
            transform = np.asarray(matrices[0], dtype=np.float32).reshape(4, 4)
        return LandmarkFrame(t_ns, (width, height), points, transform)

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def stream_landmarks(
    camera: Camera, source: MediaPipeLandmarkSource, should_stop: Callable[[], bool]
) -> Iterator[LandmarkFrame]:
    for frame in camera.frames():
        if should_stop():
            return
        yield source.process(frame.image, frame.t_ns)
