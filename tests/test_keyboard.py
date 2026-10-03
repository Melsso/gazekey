import pytest

from gazekey.core.geometry import ScreenGeometry
from gazekey.core.keyboard import (
    BACKSPACE,
    KEY_CHARS,
    DwellSelector,
    KeyboardModel,
    describe_keys,
)

W, H = 1710.0, 1112.0
MS = 1_000_000


def test_every_character_has_one_key_and_keys_tile_the_area_below_the_text_strip() -> None:
    letters = KEY_CHARS.replace(" ", "").replace(".", "").replace(",", "").replace(BACKSPACE, "")
    assert letters == "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    keys = KeyboardModel(W, H).keys
    assert len(keys) == len(KEY_CHARS) == 30
    assert len({k.id for k in keys}) == 30
    top = min(k.rect.y for k in keys)
    for k in keys:
        r = k.rect
        assert r.x >= 0 and r.x + r.w <= W + 1e-9 and top <= r.y and r.y + r.h <= H + 1e-9
    assert sum(k.rect.w * k.rect.h for k in keys) == pytest.approx(W * (H - top))


def test_key_at_hits_the_right_key_and_misses_the_text_strip() -> None:
    model = KeyboardModel(W, H)
    assert model.key_at((10.0, H * 0.3)) == "c:A"
    assert model.key_at((W - 10.0, H - 10.0)) == f"c:{BACKSPACE}"
    assert model.key_at((W / 2, 10.0)) is None
    assert model.key_at(None) is None


def test_typing_and_backspace() -> None:
    model = KeyboardModel(W, H)
    for key_id in ("c:H", "c:I", "c: "):
        model.press(key_id)
    assert model.text == "HI "
    model.press(f"c:{BACKSPACE}")
    model.press(f"c:{BACKSPACE}")
    assert model.text == "H"
    model.press(f"c:{BACKSPACE}")
    model.press(f"c:{BACKSPACE}")
    assert model.text == ""


def test_press_rejects_unknown_keys() -> None:
    model = KeyboardModel(W, H)
    for bad in ("zzz", "c:", "c:AB", "c:1", "g0"):
        with pytest.raises(ValueError):
            model.press(bad)


def test_describe_keys_reports_size_in_cm_and_degrees() -> None:
    text = describe_keys(ScreenGeometry(1710, 1069, 291.0, 182.0, 450.0))
    assert text.startswith("30 keys") and "cm" in text and "\u00b0" in text


def test_dwell_fires_after_the_dwell_time_and_not_before() -> None:
    sel = DwellSelector(dwell_s=0.8)
    assert sel.update(0, "a").selected is None
    mid = sel.update(400 * MS, "a")
    assert mid.selected is None and mid.progress == pytest.approx(0.5)
    assert sel.update(799 * MS, "a").selected is None
    assert sel.update(800 * MS, "a").selected == "a"


def test_leaving_the_key_resets_but_a_brief_slip_is_tolerated() -> None:
    sel = DwellSelector(dwell_s=0.8, grace_s=0.15)
    sel.update(0, "a")
    sel.update(300 * MS, None)
    assert sel.update(310 * MS, "a").progress == 0.0

    sel = DwellSelector(dwell_s=0.8, grace_s=0.15)
    sel.update(0, "a")
    sel.update(100 * MS, "b")
    assert sel.update(400 * MS, "a").progress == pytest.approx(0.5)
    assert sel.update(800 * MS, "a").selected == "a"


def test_cooldown_blocks_chain_selection_then_allows_the_same_key_again() -> None:
    sel = DwellSelector(dwell_s=0.8, cooldown_s=0.5)
    sel.update(0, "a")
    assert sel.update(800 * MS, "a").selected == "a"
    assert sel.update(1000 * MS, "a").key_id is None
    assert sel.update(1300 * MS, "a").progress == 0.0
    assert sel.update(2100 * MS, "a").selected == "a"


def test_dwell_selector_validates_arguments() -> None:
    with pytest.raises(ValueError):
        DwellSelector(dwell_s=0.0)
