from datetime import datetime
from pathlib import Path

import pytest

from gazekey.commands.calibrate import default_output, parse_screen_mm
from gazekey.commands.evaluate import format_result, run_eval
from gazekey.core.evaluation import evaluate_session
from gazekey.core.session import Conditions, SessionRecorder, save_session
from tests.synthetic import make_frame, synthetic_session


def test_parse_screen_mm() -> None:
    assert parse_screen_mm("286x179") == (286.0, 179.0)
    assert parse_screen_mm(" 286.5 X 179.25 ") == (286.5, 179.25)
    for bad in ("286", "wide", "286x", "x179", "-286x179"):
        with pytest.raises(ValueError, match="WIDTHxHEIGHT"):
            parse_screen_mm(bad)


def test_default_output_name() -> None:
    assert default_output("p1", datetime(2026, 1, 2, 3, 4, 5)) == Path(
        "sessions/p1_20260102-030405.npz"
    )


def test_format_result_reports_every_required_metric() -> None:
    cond = Conditions(participant="p7", head="still", glasses=True, distance_cm=60.0)
    text = format_result("a.npz", evaluate_session(synthetic_session(noise=0.01, conditions=cond)))
    for expected in (
        "participant=p7",
        "head=still",
        "glasses=yes",
        "calibration dots 9/9",
        "validation dots 16/16",
        "accuracy   mean",
        "precision  RMS sample-to-sample",
        "data loss  no face",
        "90% hit-rate target:",
        "cm at 60 cm",
        "1\u00b0:",
        "10\u00b0:",
    ):
        assert expected in text, expected


def test_run_eval_prints_per_session_participant_and_pooled_blocks(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    files = []
    for name, seed in (("p1", 1), ("p1", 2), ("p2", 3)):
        session = synthetic_session(noise=0.01, rng_seed=seed, conditions=Conditions(name))
        files.append(save_session(session, tmp_path / f"{name}_{seed}.npz"))
    assert run_eval(files, "ridge", "iris") == 0
    out = capsys.readouterr().out
    assert out.count("model=ridge") == 3
    assert "== participant p1: 2 session(s) pooled ==" in out
    assert "== participant p2: 1 session(s) pooled ==" in out
    assert "== ALL: 3 sessions pooled ==" in out


def test_run_eval_single_session_has_no_pooled_block(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = save_session(synthetic_session(), tmp_path / "s.npz")
    assert run_eval([path], "poly2", "iris+pose") == 0
    out = capsys.readouterr().out
    assert "model=poly2" in out and "features=iris+pose" in out and "pooled" not in out


def test_run_eval_reports_bad_files_and_keeps_going(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    good = save_session(synthetic_session(), tmp_path / "good.npz")
    rec = SessionRecorder((1280, 720))
    rec.add_frame(make_frame(0))
    no_screen = save_session(rec.build(), tmp_path / "preview_only.npz")
    garbage = tmp_path / "garbage.npz"
    garbage.write_bytes(b"not a zip")
    code = run_eval([tmp_path / "missing.npz", no_screen, garbage, good], "ridge", "iris")
    captured = capsys.readouterr()
    assert code == 1
    assert captured.err.count("error:") == 3
    assert "screen geometry" in captured.err
    assert "good.npz" in captured.out
