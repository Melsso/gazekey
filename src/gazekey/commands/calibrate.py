import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from .. import config
from ..core.evaluation import EvaluationError, evaluate_session
from ..core.session import Conditions, save_session
from ..ui.calibration_app import CalibrationError, run_calibration
from .evaluate import format_result


def default_output(participant: str, now: datetime | None = None) -> Path:
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    return config.SESSIONS_DIR / f"{participant}_{stamp}.npz"


def run_calibrate(
    *,
    distance_cm: float,
    participant: str,
    note: str,
    camera_index: int | None,
    landmarker_path: Path,
) -> int:
    conditions = Conditions(participant=participant, distance_cm=distance_cm, notes=note)
    try:
        run = run_calibration(
            conditions=conditions,
            landmarker_path=landmarker_path,
            camera_index=camera_index,
            distance_mm=distance_cm * 10.0,
            seed=0,
            report=print,
        )
    except (CalibrationError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if run is None:
        print("calibration aborted; nothing saved.")
        return 1

    session = replace(
        run.session,
        conditions=replace(conditions, origin_x_pt=run.origin_pt[0], origin_y_pt=run.origin_pt[1]),
    )
    saved = save_session(session, default_output(participant))
    print(f"saved session (landmarks only, no video) to {saved}")
    try:
        print("\n" + format_result(str(saved), evaluate_session(session)))
    except EvaluationError as exc:
        print(f"could not evaluate the session: {exc}", file=sys.stderr)
        return 1
    return 0
