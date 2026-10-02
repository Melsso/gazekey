from gazekey.commands.diagnostics import check_python


def test_doctor_python_check() -> None:
    assert check_python((3, 11))[0] and check_python((3, 12))[0]
    assert not check_python((3, 10))[0] and not check_python((3, 13))[0]
