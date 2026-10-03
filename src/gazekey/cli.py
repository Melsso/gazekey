import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__, config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gazekey", description="Webcam eye tracking, with honest measurement."
    )
    parser.add_argument("-V", "--version", action="version", version=f"gazekey {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    sub.add_parser("doctor", help="check the install: python, packages, model, camera")

    p = sub.add_parser(
        "calibrate", help="calibrate + validate: saves your profile and prints accuracy"
    )
    p.add_argument(
        "-d",
        "--distance-cm",
        type=float,
        default=60.0,
        metavar="CM",
        help="eye-to-screen distance, measured with a ruler (default 60)",
    )
    p.add_argument("-p", "--participant", default="me", metavar="NAME")
    p.add_argument(
        "-m",
        "--note",
        default="",
        metavar="TEXT",
        help='conditions worth remembering, e.g. "glasses, dim light"',
    )

    sub.add_parser("track", help="live gaze dot over your screen (uses your latest calibration)")
    sub.add_parser("type", help="hands-free on-screen keyboard (uses your latest calibration)")

    p = sub.add_parser("eval", help="re-score saved sessions offline and print metrics")
    p.add_argument(
        "files",
        nargs="*",
        type=Path,
        metavar="FILE",
        help="default: every session in sessions/",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command: str = args.command

    if command == "eval":
        from .commands.evaluate import run_eval

        files = args.files or sorted(config.SESSIONS_DIR.glob("*.npz"))
        if not files:
            print(
                f"error: no sessions found in {config.SESSIONS_DIR}/ (run: gazekey calibrate)",
                file=sys.stderr,
            )
            return 1
        return run_eval(files)

    landmarker = config.default_model_path()
    if command == "doctor":
        from .commands.diagnostics import run_doctor

        return run_doctor(landmarker, config.default_camera_index())
    if command == "calibrate":
        from .commands.calibrate import run_calibrate

        return run_calibrate(
            distance_cm=args.distance_cm,
            participant=args.participant,
            note=args.note,
            camera_index=None,
            landmarker_path=landmarker,
        )
    if command == "type":
        from .commands.keyboard import run_type

        return run_type(landmarker_path=landmarker)

    from .commands.track import run_track

    return run_track(landmarker_path=landmarker, camera_index=None)


if __name__ == "__main__":
    sys.exit(main())
