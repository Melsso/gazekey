import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from gazekey.core.geometry import ScreenGeometry
from gazekey.core.session import (
    Conditions,
    Session,
    SessionFormatError,
    SessionRecorder,
    load_session,
    save_session,
)
from tests.synthetic import make_frame

SIZE = (1280, 720)


def build_session() -> Session:
    rec = SessionRecorder(SIZE, now_utc=lambda: datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC))
    rec.begin_dot((100.0, 200.0), "cal", 0)
    for i in range(5):
        rec.add_frame(make_frame(i * 1_000, h_r=0.1 * i))
    rec.end_dot(4_500)
    rec.begin_dot((300.5, 50.25), "val", 5_000)
    rec.add_frame(make_frame(5_000, face=False))
    rec.add_frame(make_frame(6_000, with_transform=False))
    rec.end_dot(7_000)
    screen = ScreenGeometry(1440, 900, 286.0, 179.0, 600.0, eye_offset_mm=(5.0, -3.0))
    return rec.build(
        conditions=Conditions(participant="p1", head="still", glasses=True, distance_cm=60.0),
        screen=screen,
    )


def test_round_trip_preserves_everything(tmp_path: Path) -> None:
    original = build_session()
    path = save_session(original, tmp_path / "s.npz")
    loaded = load_session(path)
    assert np.array_equal(loaded.timestamps_ns, original.timestamps_ns)
    assert np.array_equal(loaded.landmarks, original.landmarks, equal_nan=True)
    assert np.array_equal(loaded.transforms, original.transforms, equal_nan=True)
    assert np.array_equal(loaded.dot_positions_px, original.dot_positions_px)
    assert loaded.dot_roles.tolist() == ["cal", "val"]
    assert np.array_equal(loaded.dot_t_on_ns, original.dot_t_on_ns)
    assert np.array_equal(loaded.dot_t_off_ns, original.dot_t_off_ns)
    assert loaded.image_size == SIZE
    assert loaded.conditions == original.conditions
    assert loaded.screen == original.screen
    assert loaded.created_utc == "2026-01-02T03:04:05+00:00"
    assert loaded.landmarks.dtype == np.float32
    assert loaded.timestamps_ns.dtype == np.int64


def test_save_appends_suffix_and_leaves_no_temp_file(tmp_path: Path) -> None:
    path = save_session(build_session(), tmp_path / "sub" / "run1")
    assert path.name == "run1.npz"
    assert sorted(p.name for p in path.parent.iterdir()) == ["run1.npz"]


def test_saved_file_contains_no_pickled_objects(tmp_path: Path) -> None:
    path = save_session(build_session(), tmp_path / "s.npz")
    with np.load(path, allow_pickle=False) as data:
        assert {"landmarks", "timestamps_ns", "meta"} <= set(data.files)
        assert not any(name in data.files for name in ("image", "video", "frames"))


def test_missing_face_frames_are_nan_and_flagged() -> None:
    s = build_session()
    assert s.n_frames == 7 and s.n_dots == 2
    assert list(s.face_found) == [True] * 5 + [False, True]
    assert np.isnan(s.transforms[5]).all() and np.isnan(s.transforms[6]).all()
    assert np.isfinite(s.transforms[0]).all()


def test_frame_slice_and_dot_indices() -> None:
    s = build_session()
    assert s.frame_slice(1_000, 4_000) == slice(1, 4)
    assert s.frame_slice(0, 10**9) == slice(0, 7)
    assert s.frame_slice(10**9, 2 * 10**9) == slice(7, 7)
    assert s.dot_indices("cal").tolist() == [0]
    assert s.dot_indices("val").tolist() == [1]


def test_empty_session_round_trips(tmp_path: Path) -> None:
    s = SessionRecorder(SIZE).build()
    loaded = load_session(save_session(s, tmp_path / "empty.npz"))
    assert loaded.n_frames == 0 and loaded.n_dots == 0 and loaded.screen is None


def test_recorder_rejects_bad_usage() -> None:
    rec = SessionRecorder(SIZE)
    rec.add_frame(make_frame(1_000))
    with pytest.raises(ValueError):
        rec.add_frame(make_frame(500))
    with pytest.raises(ValueError):
        rec.add_frame(make_frame(2_000, image_size=(640, 480)))
    with pytest.raises(RuntimeError):
        rec.end_dot(0)
    rec.begin_dot((1, 1), "cal", 0)
    with pytest.raises(RuntimeError):
        rec.begin_dot((2, 2), "cal", 0)
    with pytest.raises(RuntimeError):
        rec.build()
    with pytest.raises(ValueError):
        SessionRecorder(SIZE).begin_dot((1, 1), "xyz", 0)  # type: ignore[arg-type]


def test_session_validates_shapes_and_order() -> None:
    good = build_session()
    with pytest.raises(ValueError):
        Session(
            timestamps_ns=good.timestamps_ns[::-1].copy(),
            landmarks=good.landmarks,
            transforms=good.transforms,
            image_size=SIZE,
            dot_positions_px=good.dot_positions_px,
            dot_roles=good.dot_roles,
            dot_t_on_ns=good.dot_t_on_ns,
            dot_t_off_ns=good.dot_t_off_ns,
        )
    with pytest.raises(ValueError):
        Session(
            timestamps_ns=good.timestamps_ns,
            landmarks=good.landmarks[:3],
            transforms=good.transforms,
            image_size=SIZE,
            dot_positions_px=good.dot_positions_px,
            dot_roles=good.dot_roles,
            dot_t_on_ns=good.dot_t_on_ns,
            dot_t_off_ns=good.dot_t_off_ns,
        )


def test_load_rejects_unknown_schema(tmp_path: Path) -> None:
    path = save_session(build_session(), tmp_path / "s.npz")
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    meta = json.loads(str(arrays["meta"]))
    meta["schema_version"] = 99
    arrays["meta"] = np.array(json.dumps(meta))
    bad = tmp_path / "bad.npz"
    np.savez_compressed(bad, **arrays)
    with pytest.raises(SessionFormatError):
        load_session(bad)


def test_load_rejects_missing_fields(tmp_path: Path) -> None:
    bad = tmp_path / "bad.npz"
    np.savez_compressed(bad, meta=np.array(json.dumps({"schema_version": 1})))
    with pytest.raises(SessionFormatError):
        load_session(bad)


def test_conditions_validation() -> None:
    with pytest.raises(ValueError):
        Conditions(head="sideways")
    with pytest.raises(ValueError):
        Conditions(light="blinding")
    with pytest.raises(ValueError):
        Conditions(distance_cm=-5.0)
    with pytest.raises(ValueError):
        Conditions(minutes_since_calibration=-1.0)


def test_extra_calibration_role_round_trips(tmp_path: Path) -> None:
    rec = SessionRecorder(SIZE)
    for role, t in (("cal", 0), ("ext", 1_000), ("val", 2_000)):
        rec.begin_dot((10.0, 20.0), role, t)  # type: ignore[arg-type]
        rec.add_frame(make_frame(t))
        rec.end_dot(t + 500)
    loaded = load_session(save_session(rec.build(), tmp_path / "s.npz"))
    assert loaded.dot_roles.tolist() == ["cal", "ext", "val"]
    assert loaded.dot_indices("ext").tolist() == [1]
