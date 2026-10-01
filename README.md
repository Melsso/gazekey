# gazekey

Webcam eye tracking and a hands-free, dwell-to-select on-screen keyboard, built as an
accessibility-motivated CV/ML engineering project. Dedicated eye trackers cost thousands of
dollars; an ordinary laptop camera costs nothing. The point of this repository is honest
**measurement**: gaze error in degrees of visual angle, a "how big must a button be" curve,
and a text-entry study. Measurement and write-up quality matter more than feature count.

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
| 0 | Repo, packaging, CI, model download, CLI skeleton | done, verified on Apple Silicon |
| 1 | Capture, landmarks, features, session I/O, live overlay | done, verified on Apple Silicon |
| 2 | Calibration/validation UI, ridge mapper, degree metrics, `eval` | code written; first real accuracy number pending |
| 3 | Ablations, hit-rate curve | not started |
| 4 | Keyboard and text-entry study | not started |
| 5 | Robustness and failure analysis | not started |
| 6 | Demo, write-up | not started |

## Quick start (macOS, Apple Silicon)

```sh
uv sync                                    # Python 3.11 is pinned in .python-version
uv run python scripts/download_model.py    # fetches the FaceLandmarker model (not committed)
uv run gazekey doctor -c 0                 # checks wheels, model, and the camera
uv run gazekey preview -s 20 -r /tmp/smoke.npz   # live overlay; saves landmarks only

# Measure: sit at your normal distance, measure eye-to-screen with a ruler (say 60 cm)
uv run gazekey calibrate -d 60 -p me -H still      # 9 calibration + 16 validation dots, ~50 s
uv run gazekey eval sessions/*.npz                 # replay offline: accuracy, precision, hit rates
uv run gazekey inspect sessions/me_*.npz           # feature diagnostics for a recording
```

`calibrate` prints the screen size it read from macOS; check it against your Mac's spec and
override with `-S WIDTHxHEIGHT` (mm) if it is wrong. Tag every session with its conditions
(`-H still|free`, `-g` glasses, `-L normal|dim`, `-t` minutes since the last calibration).

macOS will ask for camera permission for your terminal on first use.

Development: `uv run ruff check . && uv run mypy && uv run pytest`.

## Layout

```
src/gazekey/
  config.py  cli.py
  core/      pure logic (no cv2 / mediapipe / Qt): landmarks, features, geometry, session,
             calibration, mapper, metrics, evaluation
  vision/    hardware adapters: capture, landmarker (MediaPipe), model download
  commands/  what each CLI command does: calibrate, evaluate, preview, diagnostics
  ui/        the PySide6 calibration app
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
  positions, screen geometry and condition tags. Every ablation can be re-run offline without
  the camera, and no image of the user is ever stored.
- **MediaPipe behind `LandmarkSource`.** Only `landmarks/mediapipe_source.py` imports it, so
  features, geometry, mapper and metrics are testable with synthetic landmarks.
- **Landmark indices are verified, not remembered.** Eye corners, lids and iris rings were
  checked against MediaPipe's own mesh definition; a test guards the iris block layout.
- **Eye-local features.** The iris is projected onto each eye's corner-to-corner axis, so
  horizontal and vertical iris position are invariant to head roll and image scale.
  Three vertical/openness variants are kept so the ablation can compare them.
- **Timestamps at capture, monotonic clock.** Each frame is stamped right after the driver
  returns it. This is an *upper bound* on exposure time (driver buffering is invisible), so
  reported latencies are lower bounds.
- **Degrees everywhere.** `geometry.py` converts pixels, millimetres and visual angle
  (exact 3-D angle between lines of sight, not a small-angle approximation) and is
  unit-tested against hand-computed values.
- **Provisional constants.** The blink threshold (`DEFAULT_OPEN_THRESHOLD`, now 0.10) is tuned
  only roughly: in a real recording, openness varied with gaze direction (looking down lowers
  the lids), so the first guess of 0.12 discarded lowered lids as well as blinks. It gets its
  final value from error-vs-openness on dot data in Phase 3. The facial transformation matrix
  was confirmed to be column-vector convention (translation in the last column).
- **Calibration protocol.** 3x3 grid for calibration; 16 validation points, one per cell of a
  4x4 grid, jittered with a seeded RNG and rejected if they fall near any calibration point, so
  the mapper is never scored on data it was fitted to. Per dot the first 500 ms (saccade
  latency) is ignored and the next 1 s analysed. The median is used because it shrugs off a
  partly late saccade or a blink (tests show accuracy survives 30% contamination of a window,
  while precision does not, which is why the skip window exists).
- **Ridge with exact leave-one-dot-out selection.** With 9 calibration dots, a polynomial model
  over 4 features is underdetermined, so the baseline is linear. Ridge strength is chosen on the
  calibration dots alone via the closed-form leave-one-out error (verified against brute force).
- **Hit rate is an angular square.** A sample counts as a hit for a target of size s degrees if
  its horizontal and vertical angular offsets are each at most s/2. Near the screen normal this
  equals a physical square of side 2 D tan(s/2), and unlike a millimetre square it can be pooled
  across sessions with different screens or viewing distances. The headline (smallest target
  with 90% hits) is computed exactly from the sample quantile, not read off a coarse curve.
- **Time is injected everywhere.** The calibration sequencer is a pure state machine driven by
  a clock and a stream of frames; tests run a full 25-dot calibration against a synthetic user
  with a known gaze mapping and check that the evaluation recovers it.
- **Qt for the UI.** PySide6 gives logical-point coordinates (Retina handled by Qt) and the
  physical screen size via `QScreen`. Dots are in points and never mirrored; the camera is read
  on a worker thread and drained by the UI thread, which owns the recorder.
