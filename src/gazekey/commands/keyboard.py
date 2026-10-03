import sys
from pathlib import Path

from .. import config
from ..core.calibration import FitError
from ..core.geometry import ScreenGeometry
from ..core.keyboard import DwellSelector, describe_keys
from ..core.tracking import GazeTracker, build_tracker
from ..ui.keyboard_app import KeyboardError, run_keyboard
from .track import find_profile


def _load_tracker(
    sessions_dir: Path,
) -> tuple[GazeTracker, tuple[int, int], ScreenGeometry | None] | None:
    found = find_profile(sessions_dir)
    if found is None:
        print(
            f"error: no calibration found in {sessions_dir}/. Run: uv run gazekey calibrate",
            file=sys.stderr,
        )
        return None
    path, session = found
    try:
        tracker, fit = build_tracker(session)
    except FitError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return None
    print(f"using calibration {path} ({fit.n_used}/{fit.n_total} dots)")
    origin = (session.conditions.origin_x_pt, session.conditions.origin_y_pt)
    return tracker, origin, session.screen


def run_type(*, landmarker_path: Path, sessions_dir: Path = config.SESSIONS_DIR) -> int:
    loaded = _load_tracker(sessions_dir)
    if loaded is None:
        return 1
    tracker, origin, screen = loaded
    if screen is not None:
        print(f"keyboard: {describe_keys(screen)}")
        print("compare with the 90% hit-rate target that `gazekey eval` prints")
    print("look at a key until its ring fills. Esc quits.")
    try:
        result = run_keyboard(
            tracker=tracker,
            landmarker_path=landmarker_path,
            camera_index=None,
            origin_pt=origin,
            dwell=DwellSelector(config.DWELL_S, config.GRACE_S, config.COOLDOWN_S),
        )
    except (KeyboardError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"typed: {result.text!r}")
    return 0
