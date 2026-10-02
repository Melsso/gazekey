import os
from collections.abc import Mapping
from pathlib import Path

CAMERA_ENV_VAR = "GAZEKEY_CAMERA"
MODEL_ENV_VAR = "GAZEKEY_MODEL"

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_WARMUP_FRAMES = 10
CAMERA_MAX_READ_FAILURES = 30

MODEL_FILENAME = "face_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_SHA256: str | None = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"

IRIS_DIAMETER_MM = 11.7
SESSIONS_DIR = Path("sessions")


def default_camera_index(environ: Mapping[str, str] | None = None) -> int:
    env = os.environ if environ is None else environ
    raw = env.get(CAMERA_ENV_VAR, "0")
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{CAMERA_ENV_VAR} must be an integer, got {raw!r}") from exc


def default_model_path(environ: Mapping[str, str] | None = None) -> Path:
    """`$GAZEKEY_MODEL` if set, else `models/face_landmarker.task` relative to the CWD."""
    env = os.environ if environ is None else environ
    override = env.get(MODEL_ENV_VAR)
    return Path(override) if override else Path("models") / MODEL_FILENAME
