from gazekey.commands.diagnostics import check_python, session_report
from gazekey.core.session import SessionRecorder
from tests.synthetic import make_frame


def test_session_report_shows_window_medians_matching_construction() -> None:
    rec = SessionRecorder((1280, 720))
    step = 33_000_000
    for i in range(120):
        h = -0.3 if i < 60 else 0.3
        rec.add_frame(make_frame(i * step, h_r=h, h_l=h, v_r=0.1))
    text = session_report(rec.build(), window_s=2.0)
    assert "frames: 120" in text
    assert "-0.300" in text and "0.300" in text
    assert ">= 0.12: 100.0%" in text
    assert "raw facial transformation matrix" in text


def test_session_report_handles_no_face_and_tiny_sessions() -> None:
    rec = SessionRecorder((1280, 720))
    rec.add_frame(make_frame(0, face=False))
    assert "too few frames" in session_report(rec.build())
    rec2 = SessionRecorder((1280, 720))
    for i in range(5):
        rec2.add_frame(make_frame(i * 33_000_000, face=False))
    text = session_report(rec2.build())
    assert "face found: 0.0%" in text and "no facial transformation" in text


def test_doctor_python_check() -> None:
    assert check_python((3, 11))[0] and check_python((3, 12))[0]
    assert not check_python((3, 10))[0] and not check_python((3, 13))[0]
