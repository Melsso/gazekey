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


def test_commands_and_flags_parse() -> None:
    parser = build_parser()
    c = parser.parse_args(["calibrate"])
    assert (c.distance_cm, c.participant, c.note) == (60.0, "me", "")
    c = parser.parse_args(["calibrate", "-d", "70", "-p", "ann", "-m", "glasses"])
    assert (c.distance_cm, c.participant, c.note) == (70.0, "ann", "glasses")
    assert parser.parse_args(["eval"]).files == []
    assert parser.parse_args(["eval", "a.npz", "b.npz"]).files == [Path("a.npz"), Path("b.npz")]
    assert parser.parse_args(["doctor"]).command == "doctor"
    assert parser.parse_args(["track"]).command == "track"


def test_removed_commands_and_flags_are_gone() -> None:
    parser = build_parser()
    for argv in (["preview"], ["inspect", "x.npz"], ["type"], ["study"], ["calibrate", "-n", "13"]):
        with pytest.raises(SystemExit):
            parser.parse_args(argv)


def test_eval_without_sessions_explains_itself(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["eval"]) == 1
    assert "no sessions found" in capsys.readouterr().err


def test_config_env_lookups() -> None:
    assert config.default_camera_index({}) == 0
    assert config.default_camera_index({config.CAMERA_ENV_VAR: "2"}) == 2
    with pytest.raises(ValueError):
        config.default_camera_index({config.CAMERA_ENV_VAR: "front"})
    assert config.default_model_path({}) == Path("models") / config.MODEL_FILENAME
    assert config.default_model_path({config.MODEL_ENV_VAR: "/x/y.task"}) == Path("/x/y.task")
