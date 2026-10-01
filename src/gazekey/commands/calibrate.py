import re
import sys
from datetime import datetime
from pathlib import Path

from ..core.evaluation import EvaluationError, evaluate_session
from ..core.session import Conditions, save_session
from .evaluate import format_result


def parse_screen_mm(text: str) -> tuple[float, float]:
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*[xX]\s*(\d+(?:\.\d+)?)\s*", text)
    if match is None:
        raise ValueError(f"expected WIDTHxHEIGHT in mm, e.g. 286x179, got {text!r}")
    return float(match.group(1)), float(match.group(2))


def default_output(participant: str, now: datetime | None = None) -> Path:
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    return Path("sessions") / f"{participant}_{stamp}.npz"


def run_calibrate(
    *,
    distance_cm: float,
    output: Path | None,
    participant: str,
    head: str,
    glasses: bool,
    light: str,
    minutes: float,
    seed: int,
    camera_index: int | None,
    landmarker_path: Path,
    screen_mm: tuple[float, float] | None,
) -> int:
    from ..ui.calibration_app import CalibrationError, run_calibration

    conditions = Conditions(
        participant=participant,
        head=head,
        glasses=glasses,
        light=light,
        distance_cm=distance_cm,
        minutes_since_calibration=minutes,
        notes=f"validation_seed={seed}",
    )
    try:
        session = run_calibration(
            conditions=conditions,
            landmarker_path=landmarker_path,
            camera_index=camera_index,
            distance_mm=distance_cm * 10.0,
            seed=seed,
            screen_mm_override=screen_mm,
            report=print,
        )
    except (CalibrationError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if session is None:
        print("calibration aborted; nothing saved.")
        return 1

    saved = save_session(session, output or default_output(participant))
    print(f"saved session (landmarks only, no video) to {saved}")
    try:
        print("\n" + format_result(str(saved), evaluate_session(session)))
    except EvaluationError as exc:
        print(f"could not evaluate the session: {exc}", file=sys.stderr)
        return 1
    return 0
