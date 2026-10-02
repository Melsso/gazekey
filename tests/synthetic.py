import math
from typing import Any

import numpy as np

from gazekey.core import F32, F64
from gazekey.core.calibration import (
    DEFAULT_TIMING,
    CalibrationController,
    DotTiming,
    build_schedule,
)
from gazekey.core.geometry import ScreenGeometry
from gazekey.core.landmarks import (
    LEFT_EYE,
    NUM_LANDMARKS,
    RIGHT_EYE,
    EyeIndices,
    LandmarkFrame,
)
from gazekey.core.session import Conditions, Session


def make_points(
    *,
    h_r: float = 0.0,
    v_r: float = 0.0,
    h_l: float = 0.0,
    v_l: float = 0.0,
    eye_width: float = 60.0,
    open_ratio: float = 0.3,
    iris_diam: float = 22.0,
    ipd: float = 200.0,
    centre: tuple[float, float] = (640.0, 360.0),
    roll_deg: float = 0.0,
    scale: float = 1.0,
) -> F64:
    pts = np.zeros((NUM_LANDMARKS, 3), dtype=np.float64)
    half = eye_width / 2.0
    gap = open_ratio * eye_width

    def place(eye: EyeIndices, cx: float, h: float, v: float) -> None:
        cy = centre[1]
        pts[eye.corner_img_left, :2] = (cx - half, cy)
        pts[eye.corner_img_right, :2] = (cx + half, cy)
        pts[eye.lid_upper, :2] = (cx, cy - gap / 2.0)
        pts[eye.lid_lower, :2] = (cx, cy + gap / 2.0)
        ix, iy = cx + h * half, cy + v * half
        pts[eye.iris_center, :2] = (ix, iy)
        r = iris_diam / 2.0
        offsets = [(r, 0.0), (0.0, r), (-r, 0.0), (0.0, -r)]
        for idx, (dx, dy) in zip(eye.iris_ring, offsets, strict=True):
            pts[idx, :2] = (ix + dx, iy + dy)

    place(RIGHT_EYE, centre[0] - ipd / 2.0, h_r, v_r)
    place(LEFT_EYE, centre[0] + ipd / 2.0, h_l, v_l)

    theta = math.radians(roll_deg)
    rot = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    c = np.asarray(centre)
    pts[:, :2] = ((pts[:, :2] - c) @ rot.T) * scale + c
    return pts


def to_normalized(points_px: F64, image_size: tuple[int, int] = (1280, 720)) -> F32:
    w, h = image_size
    out = points_px.astype(np.float32).copy()
    out[:, 0] /= w
    out[:, 1] /= h
    out[:, 2] /= w
    return out


def make_frame(
    t_ns: int,
    *,
    face: bool = True,
    image_size: tuple[int, int] = (1280, 720),
    with_transform: bool = True,
    **kwargs: Any,
) -> LandmarkFrame:
    points = to_normalized(make_points(**kwargs), image_size) if face else None
    transform = np.eye(4, dtype=np.float32) if (face and with_transform) else None
    return LandmarkFrame(t_ns=t_ns, image_size=image_size, points=points, transform=transform)


def rotation(yaw_deg: float, pitch_deg: float, roll_deg: float) -> F64:
    y, p, r = (math.radians(a) for a in (yaw_deg, pitch_deg, roll_deg))
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    out: F64 = ry @ rx @ rz
    return out


def make_transform(
    yaw: float, pitch: float, roll: float, t: tuple[float, float, float], scale: float = 1.0
) -> F64:
    m = np.eye(4)
    m[:3, :3] = rotation(yaw, pitch, roll) * scale
    m[:3, 3] = t
    return m


TRUE_SCREEN = ScreenGeometry(1440, 900, 286.0, 179.0, 600.0)


def gaze_features(x: float, y: float, screen: ScreenGeometry = TRUE_SCREEN) -> dict[str, Any]:
    u = x / screen.width_px - 0.5
    v = y / screen.height_px - 0.5
    return {
        "h_r": 1.2 * u + 0.10,
        "h_l": 1.2 * u - 0.10,
        "v_r": 0.4 * v - 0.10,
        "v_l": 0.4 * v - 0.08,
    }


def synthetic_session(
    *,
    screen: ScreenGeometry = TRUE_SCREEN,
    seed: int = 0,
    noise: float = 0.0,
    fps: float = 30.0,
    latency_s: float = 0.3,
    conditions: Conditions | None = None,
    blink_dots: frozenset[int] = frozenset(),
    val_bias: float = 0.0,
    rng_seed: int = 1,
    timing: DotTiming = DEFAULT_TIMING,
    n_calibration: int = 9,
    n_validation: int = 16,
) -> Session:
    rng = np.random.default_rng(rng_seed)
    controller = CalibrationController(timing)
    step = round(1e9 / fps)
    size = (1280, 720)
    centre = (screen.width_px / 2.0, screen.height_px / 2.0)

    def frame(t_ns: int, pos: tuple[float, float], *, face: bool, bias: float) -> LandmarkFrame:
        f = gaze_features(*pos, screen)
        kw: dict[str, Any] = {
            k: v + float(rng.normal(0.0, noise)) if noise else v for k, v in f.items()
        }
        kw["h_r"] += bias
        kw["h_l"] += bias
        return make_frame(t_ns, face=face, image_size=size, **kw)

    t = 1_000_000_000
    controller.on_frame(frame(t, centre, face=True, bias=0.0))
    schedule = build_schedule(
        screen.width_px,
        screen.height_px,
        seed=seed,
        n_calibration=n_calibration,
        n_validation=n_validation,
    )
    controller.start(t, schedule)
    prev, cur, seen = centre, centre, 0
    while controller.phase in ("lead_in", "dot"):
        t += step
        dot = controller.current_dot
        if dot is not None and controller.dot_number != seen:
            seen, prev, cur = controller.dot_number, cur, dot.position_px
        elapsed = controller.progress(t) * timing.total_s
        pos = cur if dot is not None and elapsed >= latency_s else (prev if dot else centre)
        dot_index = controller.dot_number - 1
        controller.on_frame(
            frame(
                t,
                pos,
                face=dot_index not in blink_dots or dot is None,
                bias=val_bias if dot is not None and dot.role == "val" else 0.0,
            )
        )
        controller.update(t)
    return controller.build_session(
        conditions or Conditions(participant="p1", distance_cm=60.0), screen
    )
