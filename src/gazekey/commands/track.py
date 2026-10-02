import sys
from pathlib import Path

from .. import config
from ..core.calibration import FitError
from ..core.session import Session, SessionFormatError, load_session
from ..core.tracking import build_tracker, calibration_error_deg


def find_profile(directory: Path) -> tuple[Path, Session] | None:
    """The newest session in `directory` that holds a usable calibration."""
    files = sorted(directory.glob("*.npz"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in files:
        try:
            session = load_session(path)
        except (SessionFormatError, OSError, ValueError):
            continue
        if session.screen is not None and session.dot_indices("cal").size > 0:
            return path, session
    return None


def run_track(
    *,
    landmarker_path: Path,
    camera_index: int | None = None,
    sessions_dir: Path = config.SESSIONS_DIR,
) -> int:
    from ..ui.tracking_app import TrackingError, run_tracking

    found = find_profile(sessions_dir)
    if found is None:
        print(
            f"error: no calibration found in {sessions_dir}/. Run: uv run gazekey calibrate",
            file=sys.stderr,
        )
        return 1
    path, session = found
    print(
        f"using calibration {path} (assumes the same display and that you sit as you did then; "
        "recalibrate if the dot is off)"
    )

    try:
        tracker, fit = build_tracker(session)
    except FitError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    err = calibration_error_deg(fit, session)
    err_text = f", training fit error {err:.2f}\u00b0 (expect more on new points)" if err else ""
    print(f"calibration used {fit.n_used}/{fit.n_total} dots{err_text}")
    print("tracking: a red dot follows your gaze. Stop with Ctrl+C in this terminal.")
    try:
        stats = run_tracking(
            tracker=tracker,
            landmarker_path=landmarker_path,
            camera_index=camera_index,
            origin_pt=(0, 0),
        )
    except (TrackingError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print("\n" + stats.summary())
    return 0
