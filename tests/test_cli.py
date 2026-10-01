from pathlib import Path

import pytest

from gazekey import config
from gazekey.cli import build_parser, main


def test_no_command_is_an_error() -> None:
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2


def test_help_and_version_exit_cleanly(capsys: pytest.CaptureFixture[str]) -> None:
    for flag in ("-h", "-V"):
        with pytest.raises(SystemExit) as exc:
            main([flag])
        assert exc.value.code == 0
    assert "gazekey" in capsys.readouterr().out


@pytest.mark.parametrize("argv", [["type"], ["study"]])
def test_future_phase_commands_are_stubs(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(argv) == 2
    assert "not implemented yet" in capsys.readouterr().err


def test_short_flags_parse() -> None:
    parser = build_parser()
    a = parser.parse_args(["preview", "-c", "1", "-l", "m.task", "-r", "out.npz", "-s", "5"])
    assert (a.camera, a.landmarker) == (1, Path("m.task"))
    assert (a.record, a.seconds) == (Path("out.npz"), 5.0)
    assert parser.parse_args(["doctor", "-c", "0", "-l", "m.task"]).camera == 0
    c = parser.parse_args(
        ["calibrate", "-d", "60", "-o", "s.npz", "-p", "p2", "-H", "still", "-g", "-L", "dim"]
        + ["-t", "5", "-e", "3", "-S", "286x179", "-c", "1", "-l", "m.task"]
    )
    assert (c.distance_cm, c.output, c.participant) == (60.0, Path("s.npz"), "p2")
    assert (c.head, c.glasses, c.light, c.minutes, c.seed) == ("still", True, "dim", 5.0, 3)
    assert (c.screen_mm, c.camera, c.landmarker) == ("286x179", 1, Path("m.task"))
    e = parser.parse_args(["eval", "a.npz", "b.npz", "-m", "poly2", "-f", "iris+pose"])
    assert e.files == [Path("a.npz"), Path("b.npz")]
    assert (e.model, e.features) == ("poly2", "iris+pose")
    assert parser.parse_args(["inspect", "x.npz", "-w", "1"]).window == 1.0
    assert parser.parse_args(["type", "-d", "600"]).dwell_ms == 600
    assert parser.parse_args(["study", "-u", "hierarchical"]).ui == "hierarchical"
    with pytest.raises(SystemExit):
        parser.parse_args(["study", "-u", "bogus"])
    with pytest.raises(SystemExit):
        parser.parse_args(["calibrate"])
    with pytest.raises(SystemExit):
        parser.parse_args(["eval", "a.npz", "-m", "svm"])


def test_calibrate_rejects_bad_screen_size() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["calibrate", "-d", "60", "-S", "wide"])
    assert exc.value.code == 2


def test_config_env_lookups() -> None:
    assert config.default_camera_index({}) == 0
    assert config.default_camera_index({config.CAMERA_ENV_VAR: "2"}) == 2
    with pytest.raises(ValueError):
        config.default_camera_index({config.CAMERA_ENV_VAR: "front"})
    assert config.default_model_path({}) == Path("models") / config.MODEL_FILENAME
    assert config.default_model_path({config.MODEL_ENV_VAR: "/x/y.task"}) == Path("/x/y.task")
