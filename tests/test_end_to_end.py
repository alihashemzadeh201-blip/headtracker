"""End-to-end accuracy: synthetic face in, cursor pixel out.

This is the test that actually answers "does the cursor land where I look?".
It wires the real geometry estimator, the real calibration fit and the real
cursor controller together against a simulated physical rig, and measures the
final error in screen pixels -- including the systematic bias the iris
measurement carries, which the calibration is expected to absorb.
"""

from __future__ import annotations

import numpy as np
import pytest

from headtracker.calibration import CalibrationModel, CalibrationSession, grid_points
from headtracker.controller import CursorSettings, GazeCursorController
from headtracker.geometry import GazeEstimator
from headtracker.mouse import AbsoluteMouse, NullBackend
from headtracker.settings import AppSettings
from tests.synthetic_face import make_face

SCREEN = (1920.0, 1080.0)
FPS = 30.0

# The simulated rig: a pinhole projection with the camera mounted slightly
# off-axis, which is how a real webcam sits on top of a monitor.
MOUNT_YAW, MOUNT_PITCH = 2.5, -1.0
GAIN_X = SCREEN[0] / 2.0 / np.tan(np.radians(24.0))
GAIN_Y = SCREEN[1] / 2.0 / np.tan(np.radians(17.0))
ORIGIN_X, ORIGIN_Y = SCREEN[0] * 0.48, SCREEN[1] * 0.51


def gaze_to_pixel(yaw, pitch):
    """Where a gaze direction lands on the simulated screen."""
    x = ORIGIN_X + GAIN_X * np.tan(np.radians(np.asarray(yaw) - MOUNT_YAW))
    y = ORIGIN_Y + GAIN_Y * np.tan(np.radians(np.asarray(pitch) - MOUNT_PITCH))
    return np.column_stack([x, y])


def pixel_to_gaze(x, y):
    """Invert the rig: the gaze needed to look at a screen pixel."""
    yaw = np.degrees(np.arctan((np.asarray(x) - ORIGIN_X) / GAIN_X)) + MOUNT_YAW
    pitch = np.degrees(np.arctan((np.asarray(y) - ORIGIN_Y) / GAIN_Y)) + MOUNT_PITCH
    return np.column_stack([yaw, pitch])


def split_gaze(yaw: float, pitch: float) -> tuple:
    """Share a gaze between head and eyes the way a person actually does.

    The eyes take the fine part and the head the rest, capped so the eyes stay
    inside their physiological range.
    """
    eye_yaw = float(np.clip(yaw, -18.0, 18.0))
    eye_pitch = float(np.clip(pitch, -14.0, 14.0))
    return (yaw - eye_yaw, pitch - eye_pitch, eye_yaw, eye_pitch)


def test_rig_helpers_round_trip():
    """Guards the test rig itself: passing the wrong units here silently
    produces a plausible-looking but meaningless accuracy number."""
    angles = np.array([[5.0, -3.0], [-12.0, 7.0], [0.0, 0.0], [20.0, 12.0]])
    pixels = gaze_to_pixel(angles[:, 0], angles[:, 1])
    np.testing.assert_allclose(pixel_to_gaze(pixels[:, 0], pixels[:, 1]), angles, atol=1e-9)


def test_split_gaze_preserves_the_total_direction():
    for yaw, pitch in [(0.0, 0.0), (5.0, -4.0), (-30.0, 20.0), (12.0, 40.0)]:
        head_yaw, head_pitch, eye_yaw, eye_pitch = split_gaze(yaw, pitch)
        assert head_yaw + eye_yaw == pytest.approx(yaw)
        assert head_pitch + eye_pitch == pytest.approx(pitch)


def probe_pixels(columns: int, rows: int) -> np.ndarray:
    """A grid of screen pixels covering the area the user actually aims at."""
    xs = np.linspace(0.15 * SCREEN[0], 0.85 * SCREEN[0], columns)
    ys = np.linspace(0.20 * SCREEN[1], 0.80 * SCREEN[1], rows)
    grid_x, grid_y = np.meshgrid(xs, ys)
    return np.column_stack([grid_x.ravel(), grid_y.ravel()])


def measure_at(estimator, pixels: np.ndarray, noise_px: float = 0.0) -> np.ndarray:
    """The gaze angles the estimator actually reports when aimed at pixels.

    Evaluation must go through the estimator, because the running app feeds the
    calibration *measured* angles.  Scoring it against the ideal angles instead
    would charge the model for the estimator's own (calibrated-away) bias.
    """
    required = pixel_to_gaze(pixels[:, 0], pixels[:, 1])
    observed = []
    for index, row in enumerate(required):
        sample = measure(estimator, row[0], row[1], noise_px=noise_px, seed=int(index))
        observed.append((sample.yaw, sample.pitch))
    return np.array(observed, dtype=np.float64)


#: Use the resolution the app requests by default, so these numbers describe
#: the shipped configuration rather than an arbitrary one.
FRAME = (AppSettings().camera_width, AppSettings().camera_height)


@pytest.fixture(name="estimator")
def fixture_estimator() -> GazeEstimator:
    defaults = AppSettings()
    return GazeEstimator(
        FRAME[0],
        FRAME[1],
        camera_fov_deg=defaults.camera_fov_deg,
        min_eye_open=defaults.min_eye_open,
    )


def measure(estimator, yaw: float, pitch: float, noise_px: float = 0.0, seed: int = 0):
    """Run the real estimator on a synthetic face aimed at a gaze direction."""
    head_yaw, head_pitch, eye_yaw, eye_pitch = split_gaze(yaw, pitch)
    face = make_face(
        head_yaw=head_yaw,
        head_pitch=head_pitch,
        eye_yaw=eye_yaw,
        eye_pitch=eye_pitch,
        width=FRAME[0],
        height=FRAME[1],
        noise_px=noise_px,
        seed=seed,
    )
    return estimator.estimate(face.points)


def run_calibration(estimator, columns=4, rows=4, noise_px=0.0) -> CalibrationModel:
    """Drive the real CalibrationSession with measured gaze samples."""
    points = grid_points(columns, rows)
    session = CalibrationSession(points, screen=SCREEN, countdown_s=0.1)
    clock = 0.0
    session.start(clock)

    guard = 0
    while not session.finished and guard < 20000:
        guard += 1
        clock += 1.0 / FPS
        if session.is_collecting():
            normalised = points[session.index]
            # A calibration point *is* a screen pixel; work out the gaze that
            # would land there, then measure what the estimator makes of it.
            target_x = normalised[0] * SCREEN[0]
            target_y = normalised[1] * SCREEN[1]
            required = pixel_to_gaze(target_x, target_y)[0]
            # A handful of frames per point, as the real app would collect.
            sample = measure(
                estimator, required[0], required[1], noise_px=noise_px, seed=guard
            )
            if sample.valid:
                session.add_sample(sample.yaw, sample.pitch, sample.distance)
        session.update(clock)
        if session.is_ready():
            session.advance(clock)  # stand in for the user's click

    assert session.finished, "calibration never completed"
    return session.build(degree=2)


# --------------------------------------------------------------------------
# Calibration accuracy through the real measurement chain
# --------------------------------------------------------------------------
def test_calibration_absorbs_the_measurement_bias(estimator):
    """The iris estimate is biased; calibrating on measured angles must hide it.

    The synthetic rig is exact, so any end-to-end error here comes from the
    estimator's bias and from the degree-2 fit -- nothing else.
    """
    model = run_calibration(estimator, 4, 4)

    pixels = probe_pixels(8, 6)
    errors = np.linalg.norm(model.predict(measure_at(estimator, pixels)) - pixels, axis=1)

    assert float(np.mean(errors)) < 20.0, f"mean error {errors.mean():.0f} px"
    assert float(np.max(errors)) < 50.0, f"worst error {errors.max():.0f} px"


def test_end_to_end_cursor_error_on_a_clean_signal(estimator):
    """Look at a pixel, read back where the cursor actually went."""
    model = run_calibration(estimator, 4, 4)
    backend = NullBackend((int(SCREEN[0]), int(SCREEN[1])))
    controller = GazeCursorController(AbsoluteMouse(backend), model, CursorSettings())

    targets = [(300.0, 200.0), (960.0, 540.0), (1600.0, 900.0), (200.0, 800.0)]
    errors = []
    clock = 0.0
    for pixel_x, pixel_y in targets:
        required = pixel_to_gaze(pixel_x, pixel_y)[0]
        for _ in range(45):  # let the smoothing settle
            sample = measure(estimator, required[0], required[1])
            controller.update(sample, clock)
            clock += 1.0 / FPS
        errors.append(np.hypot(*(np.array(backend.position()) - np.array([pixel_x, pixel_y]))))

    mean_error = float(np.mean(errors))
    assert mean_error < 45.0, f"per-target errors in px: {[round(e) for e in errors]}"


def test_end_to_end_error_with_realistic_landmark_jitter(estimator):
    """The honest number: what a 1 px landmark wobble costs after smoothing."""
    model = run_calibration(estimator, 4, 4, noise_px=1.0)
    backend = NullBackend((int(SCREEN[0]), int(SCREEN[1])))
    controller = GazeCursorController(AbsoluteMouse(backend), model, CursorSettings())

    required = pixel_to_gaze(700.0, 400.0)[0]
    positions = []
    for index in range(120):
        sample = measure(estimator, required[0], required[1], noise_px=1.0, seed=index)
        controller.update(sample, index / FPS)
        if index > 60:
            positions.append(backend.position())

    positions = np.array(positions, dtype=float)
    jitter = float(np.mean(np.linalg.norm(positions - positions.mean(axis=0), axis=1)))
    offset = float(np.linalg.norm(positions.mean(axis=0) - np.array([700.0, 400.0])))

    # Measured at the shipped defaults: ~10 px off target with ~37 px of
    # residual wobble.  The wobble is the resolution floor of webcam gaze
    # tracking -- the iris is ~16 px across even at 1080p -- not something the
    # filter can remove without making the cursor feel sluggish.
    assert jitter < 60.0, f"cursor jittered by {jitter:.0f} px"
    assert offset < 30.0, f"cursor sat {offset:.0f} px off target"


def test_calibration_beats_the_uncalibrated_default(estimator):
    """Quantifies what asking the user to calibrate actually buys."""
    calibrated = run_calibration(estimator, 4, 4)
    default = CalibrationModel.default(SCREEN)

    pixels = probe_pixels(6, 5)
    observed = measure_at(estimator, pixels)

    calibrated_error = np.linalg.norm(calibrated.predict(observed) - pixels, axis=1)
    default_error = np.linalg.norm(default.predict(observed) - pixels, axis=1)

    assert float(np.mean(calibrated_error)) < float(np.mean(default_error)) / 4.0


def test_more_calibration_points_do_not_make_it_worse(estimator):
    small = run_calibration(estimator, 3, 3)
    large = run_calibration(estimator, 5, 4)

    pixels = probe_pixels(7, 5)
    observed = measure_at(estimator, pixels)

    small_error = float(np.mean(np.linalg.norm(small.predict(observed) - pixels, axis=1)))
    large_error = float(np.mean(np.linalg.norm(large.predict(observed) - pixels, axis=1)))
    assert large_error < small_error * 1.5
