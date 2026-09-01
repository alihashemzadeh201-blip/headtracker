"""Cursor control: absolute positioning, distance compensation, tracking loss."""

from __future__ import annotations

import numpy as np
import pytest

from headtracker.calibration import CalibrationModel
from headtracker.controller import CursorSettings, GazeCursorController
from headtracker.geometry import GazeSample
from headtracker.mouse import AbsoluteMouse, NullBackend

SCREEN = (1920.0, 1080.0)


def make_model() -> CalibrationModel:
    """A simple linear map from screen-plane units to pixels."""
    model = CalibrationModel.default(SCREEN)
    model.coefficients = np.array(
        [
            [SCREEN[0] / 2.0, SCREEN[0] / 40.0, 0.0],
            [SCREEN[1] / 2.0, 0.0, SCREEN[1] / 24.0],
        ],
        dtype=np.float64,
    ).T
    model.reference = (0.0, 0.0)
    return model


def make_controller(**settings) -> tuple:
    backend = NullBackend(SCREEN)  # type: ignore[arg-type]
    mouse = AbsoluteMouse(backend)
    controller = GazeCursorController(mouse, make_model(), CursorSettings(**settings))
    return controller, backend


def sample(yaw: float, pitch: float, valid: bool = True, screen_dx: float = 0.0) -> GazeSample:
    """The controller works in screen-plane units, so yaw/pitch double as those."""
    return GazeSample(
        yaw=yaw, pitch=pitch, screen_x=yaw + screen_dx, screen_y=pitch, valid=valid,
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


def test_the_controller_applies_no_distance_term():
    """Distance is compensated upstream, in the geometry, not by a fitted factor.

    The gaze ray is intersected with the screen plane before it reaches this
    layer, so the sample it receives already encodes how far the head is from
    the screen.  Scaling it again here would double-count.
    """
    controller, backend = make_controller()
    for index in range(60):
        controller.update(sample(8.0, 0.0), index / 30.0)
    settled = backend.position()[0]

    moved = sample(8.0, 0.0)
    moved.distance = 5200.0
    for index in range(60, 120):
        controller.update(moved, index / 30.0)

    assert backend.position()[0] == pytest.approx(settled, abs=2.0)


def test_a_shifted_head_moves_the_cursor_with_it():
    """The compensation the user asked for: a head that slides sideways while
    the gaze direction stays put must drag the cursor along.

    ``screen_x`` already carries the head's lateral position, so a shift in it
    has to reach the screen.  This is the behaviour an angle-only mapping
    cannot express -- the angle is unchanged, so the angle says "stay put".
    """
    controller, backend = make_controller()
    for index in range(60):
        controller.update(sample(8.0, 0.0), index / 30.0)
    before = backend.position()[0]

    for index in range(60, 180):
        controller.update(sample(8.0, 0.0, screen_dx=10.0), index / 30.0)
    after = backend.position()[0]

    # The linear test model maps one screen-plane unit to SCREEN[0]/40 pixels.
    expected = 10.0 * SCREEN[0] / 40.0
    assert (after - before) == pytest.approx(expected, rel=0.02)


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
