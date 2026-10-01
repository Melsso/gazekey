import platform
import sys
import time
import warnings
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from .. import config
from ..core import F64
from ..core.features import FEATURE_NAMES, feature_index, features_from_normalized, valid_mask
from ..core.session import Session, load_session
from ..vision.model import sha256_file

SUPPORTED_PYTHON = ((3, 11), (3, 13))


def check_python(info: tuple[int, int] | None = None) -> tuple[bool, str]:
    major, minor = info if info is not None else sys.version_info[:2]
    lo, hi = SUPPORTED_PYTHON
    ok = lo <= (major, minor) < hi
    detail = f"Python {major}.{minor}"
    if not ok:
        detail += f" (supported: >= {lo[0]}.{lo[1]}, < {hi[0]}.{hi[1]})"
    return ok, detail


def check_platform() -> tuple[bool, str]:
    system, machine = platform.system(), platform.machine()
    ok = not (system == "Darwin" and machine != "arm64")
    note = "" if ok else " (x86_64 Python under Rosetta? use a native arm64 Python)"
    return ok, f"{system} {machine}{note}"


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def check_packages() -> tuple[bool, str]:
    names = ("numpy", "opencv-contrib-python", "mediapipe")
    versions = {n: _package_version(n) for n in names}
    ok = all(v != "not installed" for v in versions.values())
    stray = _package_version("opencv-python")
    detail = ", ".join(f"{n} {v}" for n, v in versions.items())
    if stray != "not installed":
        ok = False
        detail += f"; WARNING: opencv-python {stray} is also installed and will clash"
    return ok, detail


def check_model(path: Path) -> tuple[bool, str]:
    if not path.is_file():
        return False, f"{path} missing (run: uv run python scripts/download_model.py)"
    digest = sha256_file(path)
    pinned = config.MODEL_SHA256
    if pinned is not None and digest != pinned:
        return False, f"{path} hash {digest} != pinned {pinned}"
    return True, f"{path} sha256={digest[:16]}..." + ("" if pinned else " (hash not pinned)")


def check_landmarker(path: Path) -> tuple[bool, str]:
    from ..vision.landmarker import MediaPipeLandmarkSource

    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    with MediaPipeLandmarkSource(path) as source:
        t0 = time.monotonic_ns()
        frame = source.process(blank, t0)
        elapsed_ms = (time.monotonic_ns() - t0) / 1e6
    ok = not frame.valid
    return ok, f"FaceLandmarker ran on a blank frame in {elapsed_ms:.0f} ms (first call, warm-up)"


def check_camera(index: int) -> tuple[bool, str]:
    from ..vision.capture import Camera

    n = 30
    with Camera(index) as cam:
        stamps: list[int] = []
        shape: tuple[int, ...] = ()
        for frame in cam.frames(timeout=10.0):
            stamps.append(frame.t_ns)
            shape = frame.image.shape
            if len(stamps) >= n:
                break
    if len(stamps) < 2:
        return False, "camera opened but delivered fewer than 2 frames"
    fps = 1e9 * (len(stamps) - 1) / (stamps[-1] - stamps[0])
    return True, f"camera {index}: {shape[1]}x{shape[0]} at {fps:.1f} fps (measured over {n})"


def run_doctor(model_path: Path, camera_index: int | None) -> int:
    checks: list[tuple[str, Callable[[], tuple[bool, str]]]] = [
        ("python", check_python),
        ("platform", check_platform),
        ("packages", check_packages),
        ("model file", lambda: check_model(model_path)),
    ]
    failed = False
    for name, fn in checks:
        failed |= not _report(name, fn)

    if not failed:
        failed |= not _report("landmarker", lambda: check_landmarker(model_path))
    if camera_index is not None and not failed:
        failed |= not _report("camera", lambda: check_camera(camera_index))
    print("\nAll checks passed." if not failed else "\nSome checks FAILED.")
    return 1 if failed else 0


def _report(name: str, fn: Callable[[], tuple[bool, str]]) -> bool:
    try:
        ok, detail = fn()
    except Exception as exc:
        ok, detail = False, f"{type(exc).__name__}: {exc}"
    print(f"[{'ok' if ok else 'FAIL'}] {name}: {detail}")
    return ok


_THRESHOLDS = (0.06, 0.08, 0.10, 0.12, 0.15, 0.18)
_TABLE_COLUMNS = ("r_h", "l_h", "r_v_corner", "l_v_corner", "r_v_lid", "yaw", "pitch")


def _nanmedian(x: F64) -> float:
    finite = x[np.isfinite(x)]
    return float(np.median(finite)) if finite.size else float("nan")


def _percentiles(x: F64, qs: tuple[float, ...] = (1, 5, 25, 50, 75, 95)) -> str:
    finite = x[np.isfinite(x)]
    if not finite.size:
        return "no data"
    return "  ".join(f"p{int(q)}={np.percentile(finite, q):.3f}" for q in qs)


def session_report(session: Session, window_s: float = 2.0) -> str:
    feats = features_from_normalized(session.landmarks, session.image_size, session.transforms)
    n = session.n_frames
    lines = [f"frames: {n}   image: {session.image_size[0]}x{session.image_size[1]}"]
    if n < 2:
        return "\n".join([*lines, "too few frames"])
    t = (session.timestamps_ns - session.timestamps_ns[0]) / 1e9
    lines.append(f"duration: {t[-1]:.1f} s   face found: {100 * session.face_found.mean():.1f}%")

    lines.append("\nvalid-frame share by openness threshold (blink threshold candidates):")
    for thr in _THRESHOLDS:
        lines.append(f"  >= {thr:.2f}: {100 * valid_mask(feats, thr).mean():.1f}%")
    for name in ("r_open", "l_open"):
        lines.append(f"{name}: {_percentiles(feats[:, feature_index(name)])}")

    finite_tf = np.flatnonzero(np.isfinite(session.transforms[:, 0, 0]))
    if finite_tf.size:
        lines.append("\nraw facial transformation matrix (first frame with one):")
        for row in session.transforms[int(finite_tf[0])]:
            lines.append("  " + "  ".join(f"{v:9.4f}" for v in row))
    else:
        lines.append("\nno facial transformation matrices recorded")

    lines.append("\nfeature ranges over the recording (min / median / max):")
    for name in FEATURE_NAMES:
        col = feats[:, feature_index(name)]
        finite = col[np.isfinite(col)]
        if finite.size:
            lo, mid, hi = finite.min(), np.median(finite), finite.max()
            lines.append(f"  {name:>11}: {lo:9.3f} {mid:9.3f} {hi:9.3f}")

    lines.append(f"\nmedian per {window_s:g}s window (gaze-valid frames only):")
    lines.append("  t(s)  " + "".join(f"{c:>12}" for c in _TABLE_COLUMNS))
    ok = valid_mask(feats)
    idx = {c: feature_index(c) for c in _TABLE_COLUMNS}
    for start in np.arange(0.0, t[-1], window_s):
        sel = ok & (t >= start) & (t < start + window_s)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            cells = "".join(f"{_nanmedian(feats[sel, idx[c]]):12.3f}" for c in _TABLE_COLUMNS)
        lines.append(f"  {start:4.0f}  {cells}")
    return "\n".join(lines)


def run_inspect(path: Path, window_s: float) -> int:
    print(session_report(load_session(path), window_s))
    return 0
