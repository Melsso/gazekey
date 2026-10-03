# gazekey

Webcam eye tracking and a hands-free dwell-to-select on-screen keyboard, built as an
accessibility-motivated CV/ML engineering project. Dedicated eye trackers cost thousands of
dollars; an ordinary laptop camera costs nothing. The point of this repository is honest
**measurement**: gaze error in degrees of visual angle and a "how big must a button be" curve,
which the keyboard's key size is derived from.

## Scope honesty

- **Not a medical device, and not validated with people who have motor impairments.**
  Nothing here should be read as a claim otherwise.
- **Accuracy is not assumed.** Webcam trackers are commonly reported in the low single-digit
  degrees, versus under 1 degree for dedicated hardware. This project reports what it
  actually measures, and designs the UI around it.
- **No results yet.** Every number in this README will be a measured one; until then there
  are none.

## Status

| Phase | Content | State |
|---|---|---|
| 0-1 | Packaging, CI, capture, landmarks, features, sessions | done, verified on Apple Silicon |
| 2 | Calibration/validation UI, ridge mapper, degree metrics, `eval`, live `track` | done; calibration accuracy still needs work |
| 3 | Improve accuracy, ablations | parked |
| 4 | Flat dwell keyboard (`type`) | done |
| 5 | Robustness and failure analysis | not started |
| 6 | Demo, write-up | not started |

## Quick start (macOS, Apple Silicon)

```sh
uv sync                                    # Python 3.11 is pinned in .python-version
uv run python scripts/download_model.py    # fetches the FaceLandmarker model (not committed)
uv run gazekey doctor                      # checks packages, model and camera

uv run gazekey calibrate                   # sit at your normal distance, ~50 s; prints metrics
uv run gazekey track                       # live gaze dot, uses your latest calibration
uv run gazekey type                        # hands-free keyboard, uses your latest calibration
uv run gazekey eval                        # re-score every saved session: the numbers for this README
```

`calibrate` assumes you sit 60 cm from the screen (`-d CM` to change it; measure with a ruler)
and prints the screen size it read from macOS, which you should check against your Mac's
spec. Tag a session with its conditions via `-m "glasses, dim light"` and a person via
`-p NAME`. Use `GAZEKEY_CAMERA` and `GAZEKEY_MODEL` to pick another camera or model file.
macOS asks for camera permission for your terminal on first use.

Development: `uv run ruff check . && uv run mypy && uv run pytest`.

## Layout

```
src/gazekey/
  config.py  cli.py
  core/      pure logic (no cv2 / mediapipe / Qt): landmarks, features, geometry, session,
             calibration, mapper, metrics, evaluation, filters, tracking, keyboard
  vision/    hardware adapters: capture, landmarker (MediaPipe), model download
  commands/  what each CLI command does: calibrate, track, keyboard (type), evaluate,
             diagnostics
  ui/        PySide6: the calibration app, the gaze overlay and the keyboard
```

## Privacy

Face and gaze data are personal data (GDPR). Everything stays local: no uploads, no
telemetry. Sessions store **landmark time series only, never video**, and `sessions/`,
`*.npz` and recordings are gitignored. Sessions are plain files you can delete.

## Design notes

Short explanations of each choice; they double as the write-up.

- **uv, Python 3.11.** One lockfile, and uv pins the interpreter, which matters because the
  MediaPipe wheels are version-sensitive. MediaPipe is pinned to 0.10.21, the newest release
  with explicit cp311 macOS wheels.
- **`opencv-contrib-python`, not `opencv-python`.** MediaPipe depends on the contrib build;
  installing both makes them overwrite each other's files. `gazekey doctor` warns if both exist.
- **Landmarks, not video.** Sessions are `.npz` files of landmark arrays, timestamps, dot
  positions, screen geometry and a conditions tag. Every analysis can be re-run offline without
  the camera, and no image of the user is ever stored.
- **MediaPipe behind `LandmarkSource`.** Only `vision/landmarker.py` imports it, so features,
  geometry, mapper and metrics are testable with synthetic landmarks.
- **Landmark indices are verified, not remembered.** Eye corners, lids and iris rings were
  checked against MediaPipe's own mesh definition; a test guards the iris block layout.
- **Eye-local features.** The iris is projected onto each eye's corner-to-corner axis, so
  horizontal and vertical iris position are invariant to head roll and image scale.
- **Timestamps at capture, monotonic clock.** Each frame is stamped right after the driver
  returns it. This is an *upper bound* on exposure time (driver buffering is invisible), so
  reported latencies are lower bounds.
- **Degrees everywhere.** `geometry.py` converts pixels, millimetres and visual angle
  (exact 3-D angle between lines of sight, not a small-angle approximation) and is
  unit-tested against hand-computed values.
- **Provisional constants.** The blink threshold (`DEFAULT_OPEN_THRESHOLD`, 0.10) is tuned only
  roughly: in a real recording, openness varied with gaze direction (looking down lowers the
  lids), so a first guess of 0.12 discarded lowered lids as well as blinks.
- **Calibration protocol.** 3x3 grid for calibration; 16 validation points, one per cell of a
  4x4 grid, jittered with a seeded RNG and rejected if they fall near any calibration point, so
  the mapper is never scored on data it was fitted to. Per dot the first 500 ms (saccade
  latency) is ignored and the next 1 s analysed. The median is used because it shrugs off a
  partly late saccade or a blink.
- **Ridge with exact leave-one-dot-out selection.** With 9 calibration dots, a polynomial model
  over 4 features is underdetermined, so the baseline is linear. Ridge strength is chosen on the
  calibration dots alone via the closed-form leave-one-out error (verified against brute force).
- **Hit rate is an angular square.** A sample counts as a hit for a target of size s degrees if
  its horizontal and vertical angular offsets are each at most s/2. Near the screen normal this
  equals a physical square of side 2 D tan(s/2), and unlike a millimetre square it can be pooled
  across sessions with different screens or viewing distances. The headline (smallest target
  with 90% hits) is computed exactly from the sample quantile.
- **Smoothing for the live dot.** `track` and `type` use a One Euro filter (steady at rest,
  responsive when the eyes move fast) and hold the last point through short dropouts such as
  blinks.
- **Calibration window origin.** Calibration runs in a full-screen window that macOS may place
  below the menu bar, so its top-left is not the screen's. The session stores that origin, and
  the overlay and keyboard convert gaze points through it.
- **Flat keyboard sized from the accuracy limit.** Group-then-letter layouts were tried first
  (6 to 16 groups, two selections per character) and typing stayed usable down to the smallest
  groups, so the keyboard is flat: 30 large keys in a 6x5 grid (A-Z, space, `.`, `,`,
  backspace), one selection per character. `type` prints the key size in cm and degrees at
  startup, to compare against the measured hit-rate curve.
- **Dwell selection.** A key fires after the gaze stays on it for 0.8 s, shown as a ring that
  fills. Leaving the key for under 0.15 s is ignored so jitter at a key edge does not restart
  the timer. After a selection nothing can start for 0.5 s, so whatever appears under the gaze
  is not selected by accident (the Midas-touch problem).
- **Time is injected everywhere.** The calibration sequencer and dwell selector are pure state
  machines driven by timestamps; tests run a full 25-dot calibration and a typing session
  against a synthetic user with a known gaze mapping.
- **Qt for the UI.** PySide6 gives logical-point coordinates (Retina handled by Qt) and the
  physical screen size via `QScreen`. Dots are in points and never mirrored; the camera is read
  on a worker thread and drained by the UI thread, which owns the recorder.