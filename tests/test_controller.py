"""Cursor control: absolute positioning, head compensation, tracking loss."""

from __future__ import annotations

import math

import numpy as np
import pytest

from headtracker.calibration import CalibrationModel
from headtracker.controller import CursorSettings, GazeCursorController
from headtracker.geometry import GazeSample
from headtracker.mouse import AbsoluteMouse, NullBackend

SCREEN = (1920.0, 1080.0)


#: The eye position the calibration was taken from.  Head compensation is a
#: displacement from here, so it has to be set for those tests to exercise
#: anything.
REFERENCE_EYE = (0.0, 0.0, 4000.0)


def make_model() -> CalibrationModel:
    """A simple linear map: 40 deg of yaw spans the screen width."""
    model = CalibrationModel.default(SCREEN)
    model.coefficients = np.array(
        [
            [SCREEN[0] / 2.0, SCREEN[0] / 40.0, 0.0],
            [SCREEN[1] / 2.0, 0.0, SCREEN[1] / 24.0],
        ],
        dtype=np.float64,
    ).T
    model.reference = (0.0, 0.0)
    model.reference_eye = REFERENCE_EYE
    return model


def make_controller(**settings) -> tuple:
    backend = NullBackend(SCREEN)  # type: ignore[arg-type]
    mouse = AbsoluteMouse(backend)
    controller = GazeCursorController(mouse, make_model(), CursorSettings(**settings))
    return controller, backend


def sample(yaw: float, pitch: float, valid: bool = True, head_shift=None) -> GazeSample:
    """A gaze sample, optionally with the head displaced from the calibration pose.

    ``head_shift`` is ``(dx, dy, dz)`` in face-model units, applied to
    ``REFERENCE_EYE``.  It moves the pose *translation*, which is what the
    controller compensates; a turned head is already covered by the angle.
    """
    translation = REFERENCE_EYE
    if head_shift is not None:
        translation = tuple(a + b for a, b in zip(REFERENCE_EYE, head_shift))
    return GazeSample(
        yaw=yaw, pitch=pitch, head_translation=translation, valid=valid,
        reason="" if valid else "lost",
    )


# --------------------------------------------------------------------------
# Absolute positioning
# --------------------------------------------------------------------------
def test_cursor_reaches_the_mapped_position():
    controller, backend = make_controller()
    # Let the filter settle on a fixed gaze.
    for index in range(60):
        controller.update(sample(10.0, -5.0), index / 30.0)

    expected = make_model().predict_one(10.0, -5.0)
    assert backend.position()[0] == pytest.approx(expected[0], abs=1.5)
    assert backend.position()[1] == pytest.approx(expected[1], abs=1.5)


def test_no_drift_under_repeated_identical_frames():
    """The core failure of relative movement: error must not accumulate.

    ``moveRel`` integrates every frame's rounding error, so the cursor wanders
    away from the target over time.  Recomputing the absolute target each frame
    makes that impossible.
    """
    controller, backend = make_controller()
    for index in range(300):
        controller.update(sample(7.5, 3.25), index / 30.0)
    settled = backend.position()

    for index in range(300, 900):
        controller.update(sample(7.5, 3.25), index / 30.0)

    assert backend.position() == settled


def test_returning_to_a_gaze_returns_to_the_same_pixel():
    """A relative controller cannot guarantee this; an absolute one must."""
    controller, backend = make_controller()
    for index in range(60):
        controller.update(sample(0.0, 0.0), index / 30.0)
    home = backend.position()

    clock = 60
    for offset in (12.0, -18.0, 5.0, -3.0):
        for _ in range(40):
            controller.update(sample(offset, offset / 2.0), clock / 30.0)
            clock += 1

    for _ in range(60):
        controller.update(sample(0.0, 0.0), clock / 30.0)
        clock += 1

    assert abs(backend.position()[0] - home[0]) <= 1
    assert abs(backend.position()[1] - home[1]) <= 1


def test_output_is_clamped_to_the_screen():
    controller, backend = make_controller()
    for index in range(40):
        controller.update(sample(500.0, 500.0), index / 30.0)
    x, y = backend.position()
    assert 0 <= x <= SCREEN[0] - 1
    assert 0 <= y <= SCREEN[1] - 1


# --------------------------------------------------------------------------
# Gain and distance compensation
# --------------------------------------------------------------------------
def test_gain_scales_the_offset_from_the_calibration_centre():
    plain, plain_backend = make_controller(gain=1.0)
    doubled, doubled_backend = make_controller(gain=2.0)

    for index in range(60):
        plain.update(sample(8.0, 0.0), index / 30.0)
        doubled.update(sample(8.0, 0.0), index / 30.0)

    centre = SCREEN[0] / 2.0
    plain_offset = plain_backend.position()[0] - centre
    doubled_offset = doubled_backend.position()[0] - centre
    assert doubled_offset == pytest.approx(2.0 * plain_offset, abs=2.0)


def test_a_head_at_the_calibration_pose_gets_no_correction():
    """The property that makes the compensation safe to ship.

    The eye position is recovered by solving against a canonical face model, so
    it carries an unknown offset.  Measuring the shift from the calibration
    pose cancels that offset exactly -- which is the whole reason the
    correction is a difference and not an absolute position.
    """
    controller, backend = make_controller()
    model = make_model()
    model.reference_eye = None
    controller.set_model(model)
    for index in range(60):
        controller.update(sample(8.0, 0.0), index / 30.0)
    without = backend.position()[0]

    controller.set_model(make_model())
    for index in range(60, 180):
        controller.update(sample(8.0, 0.0), index / 30.0)

    assert backend.position()[0] == pytest.approx(without, abs=1.0)


def test_a_shifted_head_moves_the_cursor_with_it():
    """The compensation the user asked for: a head that slides sideways while
    the gaze direction stays put must drag the cursor along.

    The angle is unchanged, so the calibration alone says "stay put".  Only the
    eye's displacement knows the look-at point has moved.
    """
    controller, backend = make_controller()
    for index in range(60):
        controller.update(sample(8.0, 0.0), index / 30.0)
    before = backend.position()[0]

    for index in range(60, 180):
        controller.update(sample(8.0, 0.0, head_shift=(100.0, 0.0, 0.0)), index / 30.0)
    after = backend.position()[0]

    # The test model maps 40 deg to the screen width, so at 8 deg the local
    # scale is SCREEN[0]/40 pixels per degree-equivalent of shift.
    expected = 100.0 * SCREEN[0] / 40.0 / REFERENCE_EYE[2] * (180.0 / math.pi)
    assert (after - before) == pytest.approx(expected, rel=0.05)


def test_leaning_back_widens_the_response():
    """Further from the screen, the same angle sweeps further -- analytically.

    This is what the removed ``compensate_distance`` factor approximated with a
    fitted correction.
    """
    controller, backend = make_controller()
    for index in range(60):
        controller.update(sample(8.0, 0.0), index / 30.0)
    near = backend.position()[0]

    for index in range(60, 180):
        controller.update(sample(8.0, 0.0, head_shift=(0.0, 0.0, 500.0)), index / 30.0)
    far = backend.position()[0]

    centre = SCREEN[0] / 2.0
    assert (far - centre) > (near - centre) * 1.05


# --------------------------------------------------------------------------
# Robustness
# --------------------------------------------------------------------------
def test_a_single_glitch_frame_does_not_move_the_cursor():
    controller, backend = make_controller()
    for index in range(60):
        controller.update(sample(0.0, 0.0), index / 30.0)
    before = backend.position()

    # One frame of teleportation, exactly what a landmark failure looks like.
    controller.update(sample(0.0, 0.0), 60 / 30.0)
    controller.update(sample(55.0, 40.0), 61 / 30.0)
    after_glitch = backend.position()

    assert abs(after_glitch[0] - before[0]) < 250, "a glitch frame jumped the cursor"

    for index in range(62, 120):
        controller.update(sample(0.0, 0.0), index / 30.0)
    assert abs(backend.position()[0] - before[0]) <= 1


def test_cursor_is_released_when_tracking_is_lost():
    controller, _ = make_controller()
    for index in range(60):
        controller.update(sample(5.0, 5.0), index / 30.0)
    assert not controller.is_holding

    for index in range(60, 90):
        assert controller.update(sample(0, 0, valid=False), index / 30.0) is None
    assert controller.is_holding


def test_tracking_returns_without_a_flight_from_the_old_position():
    """After a loss the cursor must appear at the new gaze, not swoop to it."""
    controller, backend = make_controller()
    for index in range(60):
        controller.update(sample(-15.0, 0.0), index / 30.0)
    for index in range(60, 90):
        controller.update(sample(0, 0, valid=False), index / 30.0)

    controller.update(sample(15.0, 0.0), 90 / 30.0)
    target = make_model().predict_one(15.0, 0.0)
    assert backend.position()[0] == pytest.approx(target[0], abs=3.0)


def test_reset_clears_the_filter_history():
    controller, backend = make_controller()
    for index in range(60):
        controller.update(sample(10.0, 0.0), index / 30.0)
    controller.reset()
    controller.update(sample(-10.0, 0.0), 60 / 30.0)
    target = make_model().predict_one(-10.0, 0.0)
    assert backend.position()[0] == pytest.approx(target[0], abs=3.0)


def test_recalibrating_resets_the_smoothing():
    controller, backend = make_controller()
    for index in range(60):
        controller.update(sample(10.0, 0.0), index / 30.0)
    controller.set_model(make_model())
    controller.update(sample(-10.0, 0.0), 60 / 30.0)
    target = make_model().predict_one(-10.0, 0.0)
    assert backend.position()[0] == pytest.approx(target[0], abs=3.0)


def test_apply_settings_reaches_the_filters():
    controller, _ = make_controller()
    controller.apply_settings(CursorSettings(min_cutoff=0.4, beta=0.2, max_speed=1234.0))
    assert controller._filter_x.min_cutoff == 0.4  # pylint: disable=protected-access
    assert controller._gate.max_speed == 1234.0  # pylint: disable=protected-access
    # beta is entered per screen height and handed to the filter per pixel.
    expected_beta = 0.2 / controller.mouse.screen[1]  # pylint: disable=protected-access
    assert controller._filter_y.beta == pytest.approx(expected_beta)  # pylint: disable=protected-access
