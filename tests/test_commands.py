import os
from datetime import datetime
from pathlib import Path

import pytest

from gazekey.commands.calibrate import default_output
from gazekey.commands.evaluate import format_result, run_eval
from gazekey.commands.track import find_profile
from gazekey.core.evaluation import evaluate_session
from gazekey.core.session import Conditions, SessionRecorder, save_session
from tests.synthetic import make_frame, synthetic_session


def test_default_output_name() -> None:
    assert default_output("p1", datetime(2026, 1, 2, 3, 4, 5)) == Path(
        "sessions/p1_20260102-030405.npz"
    )


def test_format_result_reports_every_required_metric() -> None:
    cond = Conditions(participant="p7", distance_cm=60.0, notes="glasses")
    text = format_result("a.npz", evaluate_session(synthetic_session(noise=0.01, conditions=cond)))
    for expected in (
        "participant=p7",
        "note: glasses",
        "calibration dots 9/9",
        "validation dots 16/16",
        "accuracy   mean",
        "precision  RMS sample-to-sample",
        "data loss  no face",
        "90% hit-rate target:",
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
    assert run_eval(files) == 0
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
    code = run_eval([tmp_path / "missing.npz", no_screen, garbage, good])
    captured = capsys.readouterr()
    assert code == 1
    assert captured.err.count("error:") == 3
    assert "screen geometry" in captured.err
    assert "good.npz" in captured.out


def test_run_eval_calibration_variants_and_missing_extra_dots(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dev = save_session(synthetic_session(n_calibration=13), tmp_path / "dev.npz")
    plain = save_session(synthetic_session(), tmp_path / "plain.npz")
    assert run_eval([dev], "ridge", "iris", calibration=13) == 0
    assert "cal-points=13" in capsys.readouterr().out
    assert run_eval([plain], "ridge", "iris", calibration=13) == 1
    assert "extra calibration dots" in capsys.readouterr().err


def test_find_profile_picks_the_newest_usable_session(tmp_path: Path) -> None:
    assert find_profile(tmp_path) is None
    rec = SessionRecorder((1280, 720))
    rec.add_frame(make_frame(0))
    save_session(rec.build(), tmp_path / "no_calibration.npz")
    (tmp_path / "garbage.npz").write_bytes(b"not a zip")
    assert find_profile(tmp_path) is None

    old = save_session(synthetic_session(), tmp_path / "old.npz")
    new = save_session(synthetic_session(), tmp_path / "new.npz")
    os.utime(old, (1_000, 1_000))
    os.utime(new, (2_000, 2_000))
    found = find_profile(tmp_path)
    assert found is not None and found[0] == new
