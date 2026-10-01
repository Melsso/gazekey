import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__, config
from .core.mapper import FEATURE_SETS, MAPPER_DEGREES


def _landmarker_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-l",
        "--landmarker",
        type=Path,
        default=None,
        metavar="TASK_FILE",
        help="FaceLandmarker .task file (default: $GAZEKEY_MODEL or models/face_landmarker.task)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gazekey", description="Webcam eye tracking and a hands-free dwell keyboard."
    )
    parser.add_argument("-V", "--version", action="version", version=f"gazekey {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    p = sub.add_parser("preview", help="live landmarks and gaze features, for tuning")
    p.add_argument("-c", "--camera", type=int, default=None, metavar="INDEX")
    _landmarker_flag(p)
    p.add_argument(
        "-r",
        "--record",
        type=Path,
        default=None,
        metavar="FILE",
        help="also save the landmarks (never video) to a .npz session",
    )
    p.add_argument(
        "-s",
        "--seconds",
        type=float,
        default=None,
        metavar="SEC",
        help="stop automatically after this many seconds",
    )

    p = sub.add_parser("calibrate", help="calibration + validation, saves a session")
    p.add_argument(
        "-d",
        "--distance-cm",
        type=float,
        required=True,
        metavar="CM",
        help="eye-to-screen distance, measured with a ruler",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        metavar="FILE",
        help="default: sessions/<participant>_<timestamp>.npz",
    )
    p.add_argument("-p", "--participant", default="p0", metavar="ID", help="pseudonymous id")
    p.add_argument("-H", "--head", choices=("still", "free"), default="free")
    p.add_argument("-g", "--glasses", action="store_true", help="the participant wears glasses")
    p.add_argument("-L", "--light", choices=("normal", "dim"), default="normal")
    p.add_argument(
        "-t",
        "--minutes",
        type=float,
        default=0.0,
        metavar="MIN",
        help="minutes since the previous calibration (drift sweeps)",
    )
    p.add_argument("-e", "--seed", type=int, default=0, help="seed for the validation points")
    p.add_argument(
        "-S",
        "--screen-mm",
        default=None,
        metavar="WxH",
        help="override the physical screen size in mm, e.g. 286x179",
    )
    p.add_argument("-c", "--camera", type=int, default=None, metavar="INDEX")
    _landmarker_flag(p)

    p = sub.add_parser("eval", help="replay sessions offline, print metrics")
    p.add_argument("files", nargs="+", type=Path, metavar="FILES")
    p.add_argument(
        "-m",
        "--model",
        choices=tuple(MAPPER_DEGREES),
        default="ridge",
        help="gaze mapper (default: ridge = linear)",
    )
    p.add_argument(
        "-f",
        "--features",
        choices=tuple(FEATURE_SETS),
        default="iris",
        help="feature set (default: iris)",
    )

    p = sub.add_parser("type", help="hands-free keyboard (Phase 4)")
    p.add_argument("-d", "--dwell-ms", type=int, default=800, metavar="DWELL_MS")

    p = sub.add_parser("study", help="text-entry test with the phrase list (Phase 4)")
    p.add_argument("-u", "--ui", choices=("flat", "hierarchical"), default="flat", metavar="UI")

    p = sub.add_parser("doctor", help="check the install: python, wheels, model, camera")
    _landmarker_flag(p)
    p.add_argument(
        "-c",
        "--camera",
        type=int,
        default=None,
        metavar="INDEX",
        help="also grab frames from this camera and measure fps",
    )

    p = sub.add_parser("inspect", help="diagnostic report for a recorded session")
    p.add_argument("file", type=Path)
    p.add_argument("-w", "--window", type=float, default=2.0, metavar="SEC")
    return parser


_STUB_PHASE = {"type": 4, "study": 4}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command: str = args.command

    if command in _STUB_PHASE:
        print(
            f"gazekey {command}: not implemented yet (planned for Phase {_STUB_PHASE[command]}).",
            file=sys.stderr,
        )
        return 2

    if command == "eval":
        from .commands.evaluate import run_eval

        return run_eval(args.files, args.model, args.features)
    if command == "inspect":
        from .commands.diagnostics import run_inspect

        return run_inspect(args.file, args.window)

    landmarker = args.landmarker or config.default_model_path()
    if command == "doctor":
        from .commands.diagnostics import run_doctor

        return run_doctor(landmarker, args.camera)
    if command == "calibrate":
        from .commands.calibrate import parse_screen_mm, run_calibrate

        try:
            screen_mm = parse_screen_mm(args.screen_mm) if args.screen_mm else None
        except ValueError as exc:
            build_parser().error(str(exc))
        return run_calibrate(
            distance_cm=args.distance_cm,
            output=args.output,
            participant=args.participant,
            head=args.head,
            glasses=args.glasses,
            light=args.light,
            minutes=args.minutes,
            seed=args.seed,
            camera_index=args.camera,
            landmarker_path=landmarker,
            screen_mm=screen_mm,
        )

    from .commands.preview import run_preview

    return run_preview(
        camera_index=args.camera,
        landmarker_path=landmarker,
        record_path=args.record,
        seconds=args.seconds,
    )


if __name__ == "__main__":
    sys.exit(main())
