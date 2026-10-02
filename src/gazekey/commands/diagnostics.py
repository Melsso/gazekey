import platform
import sys
import time
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from .. import config
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


def run_doctor(model_path: Path, camera_index: int) -> int:
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
    if not failed:
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
