import json
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from .. import __version__
from . import F32, F64, I64
from .geometry import ScreenGeometry
from .landmarks import NUM_LANDMARKS, LandmarkFrame

SCHEMA_VERSION = 1
DotRole = Literal["cal", "ext", "val"]
_DOT_ROLES = ("cal", "ext", "val")
_HEADS = ("still", "free")
_LIGHTS = ("normal", "dim")


class SessionFormatError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Conditions:
    participant: str = "p0"
    head: str = "free"
    glasses: bool = False
    light: str = "normal"
    distance_cm: float | None = None
    minutes_since_calibration: float = 0.0
    notes: str = ""
    origin_x_pt: int = 0
    origin_y_pt: int = 0

    def __post_init__(self) -> None:
        if self.head not in _HEADS:
            raise ValueError(f"head must be one of {_HEADS}, got {self.head!r}")
        if self.light not in _LIGHTS:
            raise ValueError(f"light must be one of {_LIGHTS}, got {self.light!r}")
        if self.distance_cm is not None and self.distance_cm <= 0:
            raise ValueError("distance_cm must be positive")
        if self.minutes_since_calibration < 0:
            raise ValueError("minutes_since_calibration must be >= 0")


@dataclass(frozen=True, slots=True, eq=False)
class Session:
    timestamps_ns: I64
    landmarks: F32
    transforms: F32
    image_size: tuple[int, int]
    dot_positions_px: F64
    dot_roles: np.ndarray[Any, np.dtype[np.str_]]
    dot_t_on_ns: I64
    dot_t_off_ns: I64
    conditions: Conditions = field(default_factory=Conditions)
    screen: ScreenGeometry | None = None
    created_utc: str = ""
    app_version: str = __version__

    def __post_init__(self) -> None:
        n = self.timestamps_ns.shape[0]
        m = self.dot_positions_px.shape[0]
        if self.timestamps_ns.ndim != 1:
            raise ValueError("timestamps_ns must be 1-D")
        if self.landmarks.shape != (n, NUM_LANDMARKS, 3):
            raise ValueError(f"landmarks must be ({n}, {NUM_LANDMARKS}, 3)")
        if self.transforms.shape != (n, 4, 4):
            raise ValueError(f"transforms must be ({n}, 4, 4)")
        if self.dot_positions_px.shape != (m, 2):
            raise ValueError("dot_positions_px must be (M, 2)")
        for name in ("dot_roles", "dot_t_on_ns", "dot_t_off_ns"):
            if getattr(self, name).shape != (m,):
                raise ValueError(f"{name} must have shape ({m},)")
        if n > 1 and np.any(np.diff(self.timestamps_ns) < 0):
            raise ValueError("timestamps_ns must be non-decreasing")
        if not set(self.dot_roles.tolist()) <= set(_DOT_ROLES):
            raise ValueError(f"dot roles must be in {_DOT_ROLES}")
        if np.any(self.dot_t_off_ns < self.dot_t_on_ns):
            raise ValueError("a dot ends before it starts")

    @property
    def n_frames(self) -> int:
        return int(self.timestamps_ns.shape[0])

    @property
    def n_dots(self) -> int:
        return int(self.dot_positions_px.shape[0])

    @property
    def face_found(self) -> np.ndarray[Any, np.dtype[np.bool_]]:
        return cast("np.ndarray[Any, np.dtype[np.bool_]]", np.isfinite(self.landmarks[:, 0, 0]))

    def dot_indices(self, role: DotRole) -> np.ndarray[Any, np.dtype[np.intp]]:
        return np.flatnonzero(self.dot_roles == role)

    def frame_slice(self, t_start_ns: int, t_end_ns: int) -> slice:
        lo = int(np.searchsorted(self.timestamps_ns, t_start_ns, side="left"))
        hi = int(np.searchsorted(self.timestamps_ns, t_end_ns, side="left"))
        return slice(lo, hi)


class SessionRecorder:
    def __init__(
        self,
        image_size: tuple[int, int],
        *,
        now_utc: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._image_size = image_size
        self._now_utc = now_utc
        self._t_ns: list[int] = []
        self._points: list[F32] = []
        self._transforms: list[F32] = []
        self._dot_pos: list[tuple[float, float]] = []
        self._dot_role: list[str] = []
        self._dot_on: list[int] = []
        self._dot_off: list[int] = []
        self._dot_open = False

    def add_frame(self, frame: LandmarkFrame) -> None:
        if frame.image_size != self._image_size:
            raise ValueError(f"frame image_size {frame.image_size} != {self._image_size}")
        if self._t_ns and frame.t_ns < self._t_ns[-1]:
            raise ValueError("frame timestamps must be non-decreasing")
        self._t_ns.append(frame.t_ns)
        self._points.append(
            frame.points
            if frame.points is not None
            else np.full((NUM_LANDMARKS, 3), np.nan, dtype=np.float32)
        )
        self._transforms.append(
            frame.transform
            if frame.transform is not None
            else np.full((4, 4), np.nan, dtype=np.float32)
        )

    def begin_dot(self, position_px: tuple[float, float], role: DotRole, t_ns: int) -> None:
        if self._dot_open:
            raise RuntimeError("previous dot has not ended")
        if role not in _DOT_ROLES:
            raise ValueError(f"role must be one of {_DOT_ROLES}")
        self._dot_pos.append((float(position_px[0]), float(position_px[1])))
        self._dot_role.append(role)
        self._dot_on.append(t_ns)
        self._dot_off.append(t_ns)
        self._dot_open = True

    def end_dot(self, t_ns: int) -> None:
        if not self._dot_open:
            raise RuntimeError("no dot in progress")
        self._dot_off[-1] = t_ns
        self._dot_open = False

    def build(
        self, *, conditions: Conditions | None = None, screen: ScreenGeometry | None = None
    ) -> Session:
        if self._dot_open:
            raise RuntimeError("a dot is still in progress")
        n = len(self._t_ns)
        return Session(
            timestamps_ns=np.asarray(self._t_ns, dtype=np.int64),
            landmarks=(
                np.stack(self._points) if n else np.empty((0, NUM_LANDMARKS, 3), dtype=np.float32)
            ),
            transforms=np.stack(self._transforms) if n else np.empty((0, 4, 4), dtype=np.float32),
            image_size=self._image_size,
            dot_positions_px=np.asarray(self._dot_pos, dtype=np.float64).reshape(-1, 2),
            dot_roles=np.asarray(self._dot_role, dtype="U3"),
            dot_t_on_ns=np.asarray(self._dot_on, dtype=np.int64),
            dot_t_off_ns=np.asarray(self._dot_off, dtype=np.int64),
            conditions=conditions or Conditions(),
            screen=screen,
            created_utc=self._now_utc().isoformat(),
            app_version=__version__,
        )


def save_session(session: Session, path: Path | str) -> Path:
    final = Path(path)
    if final.suffix != ".npz":
        final = final.with_name(final.name + ".npz")
    final.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": session.created_utc,
        "app_version": session.app_version,
        "image_size": list(session.image_size),
        "conditions": asdict(session.conditions),
        "screen": asdict(session.screen) if session.screen is not None else None,
    }
    tmp = final.with_name(final.name + ".tmp")
    try:
        with tmp.open("wb") as fh:
            np.savez_compressed(
                fh,
                timestamps_ns=session.timestamps_ns,
                landmarks=session.landmarks,
                transforms=session.transforms,
                dot_positions_px=session.dot_positions_px,
                dot_roles=session.dot_roles,
                dot_t_on_ns=session.dot_t_on_ns,
                dot_t_off_ns=session.dot_t_off_ns,
                meta=np.array(json.dumps(meta)),
            )
        os.replace(tmp, final)
    finally:
        tmp.unlink(missing_ok=True)
    return final


def load_session(path: Path | str) -> Session:
    try:
        with np.load(Path(path), allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            version = meta.get("schema_version")
            if version != SCHEMA_VERSION:
                raise SessionFormatError(
                    f"unsupported session schema {version!r} (this build reads {SCHEMA_VERSION})"
                )
            screen_raw = meta["screen"]
            screen = None
            if screen_raw is not None:
                screen_raw["eye_offset_mm"] = tuple(screen_raw["eye_offset_mm"])
                screen = ScreenGeometry(**screen_raw)
            return Session(
                timestamps_ns=data["timestamps_ns"],
                landmarks=data["landmarks"],
                transforms=data["transforms"],
                image_size=cast("tuple[int, int]", tuple(meta["image_size"])),
                dot_positions_px=data["dot_positions_px"],
                dot_roles=data["dot_roles"],
                dot_t_on_ns=data["dot_t_on_ns"],
                dot_t_off_ns=data["dot_t_off_ns"],
                conditions=Conditions(**meta["conditions"]),
                screen=screen,
                created_utc=meta["created_utc"],
                app_version=meta["app_version"],
            )
    except KeyError as exc:
        raise SessionFormatError(f"session file is missing field {exc}") from exc
