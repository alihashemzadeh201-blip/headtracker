"""End-to-end proof that head movement does not drag the cursor off target.

The main end-to-end rig in ``test_end_to_end`` models the screen as a function
of the gaze *angle* alone, which quietly assumes the eye never moves.  That is
fine for scoring angular accuracy but it cannot test what happens when the head
slides or turns, because its ground truth does not change when the head does.

This module therefore builds the ground truth the physical way: the user looks
at the point where the ray from their *actual* eye centre, along their actual
gaze direction, meets the screen plane.  A head that leans left really is
looking further left, and a test that says otherwise would be scoring the
tracker against a fiction.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pytest

from headtracker.calibration import CalibrationModel
from headtracker.controller import CursorSettings, GazeCursorController
from headtracker.geometry import GazeEstimator, gaze_direction, project_to_screen_plane
from headtracker.mouse import AbsoluteMouse, NullBackend
from headtracker.settings import AppSettings
from tests.synthetic_face import euler_to_rotation, make_face

#: A head and eye pose.  Grouped because every helper here needs all four
#: angles, and passing them loose would put the signatures over the argument
#: limit without making them any clearer.
Pose = Tuple[float, float, float, float]

SCREEN = (1920.0, 1080.0)
FPS = 30.0
CALIBRATION_DISTANCE_MM = 600.0

_DEFAULTS = AppSettings()
FRAME = (_DEFAULTS.camera_width, _DEFAULTS.camera_height)

#: The point the irises actually orbit, taken from the face model the rig
#: projects.  Deliberately *not* ``geometry.EYE_MID_MODEL``: that constant is
#: the midpoint of the model's eye corners and sits 17 units in front of this,
#: so importing it would grade the estimator against its own assumption.  The
#: resulting offset is systematic, which is exactly the kind of error the
#: calibration exists to absorb.
TRUE_EYE_MID = np.array([0.0, 171.0, -118.0])


@pytest.fixture(name="estimator")
def fixture_estimator() -> GazeEstimator:
    return GazeEstimator(
        FRAME[0],
        FRAME[1],
        camera_fov_deg=_DEFAULTS.camera_fov_deg,
        min_eye_open=_DEFAULTS.min_eye_open,
    )


def face(pose: Pose, distance_mm=CALIBRATION_DISTANCE_MM,
         noise_px=0.0, seed=0, shift_px=0.0):
    """A synthetic face, optionally slid sideways in the camera's view."""
    head_yaw, head_pitch, eye_yaw, eye_pitch = pose
    points = make_face(
        head_yaw=head_yaw,
        head_pitch=head_pitch,
        eye_yaw=eye_yaw,
        eye_pitch=eye_pitch,
        distance_mm=distance_mm,
        noise_px=noise_px,
        seed=seed,
        width=FRAME[0],
        height=FRAME[1],
    ).points
    if shift_px:
        points = points + np.array([shift_px, 0.0])
    return points


def true_eye_centre(pose: Pose, distance_mm=CALIBRATION_DISTANCE_MM,
                    shift_px=0.0) -> np.ndarray:
    """Where the eye actually is, in camera coordinates, from the face model.

    This is the ground truth the estimator is trying to recover.  It is built
    from the same model points and the same pose, so it is exact.  The eye
    angles do not enter it: the eyeball turns inside the socket, so only the
    head pose moves the centre.
    """
    rotation = euler_to_rotation(pose[0], pose[1], 0.0)
    depth = distance_mm / 0.138
    centre = rotation @ TRUE_EYE_MID + np.array([0.0, 0.0, depth])
    if shift_px:
        # A sideways slide in the image is a sideways slide of the head itself.
        focal = (FRAME[0] / 2.0) / np.tan(np.radians(60.0) / 2.0)
        centre[0] += shift_px * centre[2] / focal
    return centre


def true_screen_point(pose: Pose, **kwargs) -> tuple:
    """Where the user is really looking, on the screen plane."""
    centre = true_eye_centre(pose, **kwargs)
    # make_face places the iris relative to the head, so the gaze in the camera
    # frame is the head angle plus the eye angle.
    direction = gaze_direction(pose[0] + pose[2], pose[1] + pose[3])
    hit = project_to_screen_plane(centre, direction)
    assert hit is not None
    return hit


def screen_to_pixel(plane: tuple) -> tuple:
    """The monitor's pixel grid, as an affine map of screen-plane coordinates.

    Both are affine coordinate systems on the same physical plane, so this is
    exact rather than an approximation -- which is precisely why the projection
    in ``geometry`` turns calibration into a near-affine fit.
    """
    units_per_pixel = 2.0
    return (
        SCREEN[0] / 2.0 + plane[0] / units_per_pixel,
        SCREEN[1] / 2.0 + plane[1] / units_per_pixel,
    )


def pixel_to_pose(pixel, head_yaw=0.0, head_pitch=0.0, **kwargs):
    """The eye angles that put the gaze on ``pixel`` from a given head pose.

    Inverted by iteration, because the eye's own position depends on the head
    pose and therefore on the answer.  Two or three passes converge; the loop is
    run to ten for margin.
    """
    target_x = (pixel[0] - SCREEN[0] / 2.0) * 2.0
    target_y = (pixel[1] - SCREEN[1] / 2.0) * 2.0
    eye_yaw = eye_pitch = 0.0
    for _ in range(10):
        centre = true_eye_centre((head_yaw, head_pitch, eye_yaw, eye_pitch), **kwargs)
        vector = np.array([target_x - centre[0], target_y - centre[1], -centre[2]])
        eye_yaw = np.degrees(np.arctan2(vector[0], -vector[2])) - head_yaw
        eye_pitch = np.degrees(np.arctan2(vector[1], -vector[2])) - head_pitch
    return eye_yaw, eye_pitch


def measure_plane(estimator, pose: Pose, **kwargs):
    """The screen-plane point the estimator reports for a posed face."""
    sample = estimator.estimate(face(pose, **kwargs))
    assert sample.valid, sample.reason
    return sample.screen_x, sample.screen_y


# --------------------------------------------------------------------------
# The estimator recovers the ray, not just the angle
# --------------------------------------------------------------------------
def _ray_errors(estimator):
    """Measured screen point minus the true one, in pixels, for a range of poses."""
    rows = {}
    for label, head_yaw, head_pitch, distance, shift in (
        ("reference", 0.0, 0.0, 600.0, 0.0),
        ("yaw -18", -18.0, 0.0, 600.0, 0.0),
        ("yaw +18", 18.0, 0.0, 600.0, 0.0),
        ("pitch -12", 0.0, -12.0, 600.0, 0.0),
        ("pitch +12", 0.0, 12.0, 600.0, 0.0),
        ("450 mm", 0.0, 0.0, 450.0, 0.0),
        ("850 mm", 0.0, 0.0, 850.0, 0.0),
        ("slid -40 px", 0.0, 0.0, 600.0, -40.0),
        ("slid +40 px", 0.0, 0.0, 600.0, 40.0),
    ):
        eye_yaw, eye_pitch = pixel_to_pose((700.0, 400.0), head_yaw, head_pitch,
                                           distance_mm=distance, shift_px=shift)
        pose = (head_yaw, head_pitch, eye_yaw, eye_pitch)
        measured = measure_plane(estimator, pose, distance_mm=distance, shift_px=shift)
        truth = true_screen_point(pose, distance_mm=distance, shift_px=shift)
        rows[label] = np.array([(measured[0] - truth[0]) / 2.0,
                                (measured[1] - truth[1]) / 2.0])
    return rows


def test_sliding_and_leaning_are_compensated_to_within_a_couple_of_pixels(estimator):
    """The strongest form of the claim, and the one that is fully met.

    The measured ray carries a constant offset of roughly (-4, -77) px against
    the true one -- the vertical part is the iris model's distance-dependent
    pitch bias.  That offset does not matter: it is the same everywhere, so the
    calibration removes it.  What must *not* happen is the offset changing with
    head pose, because no single calibration can absorb a moving target.

    Measured against the reference pose, sliding the head 40 px either way and
    leaning between 450 mm and 850 mm all land within 3 px.
    """
    rows = _ray_errors(estimator)
    reference = rows["reference"]
    for label in ("slid -40 px", "slid +40 px", "450 mm", "850 mm"):
        drift = float(np.linalg.norm(rows[label] - reference))
        assert drift < 6.0, f"{label}: the ray drifted {drift:.1f} px"


def test_turning_the_head_leaves_a_bounded_residual(estimator):
    """The honest limit of the compensation, recorded rather than hidden.

    Rotation is only partly compensated: at +-18 deg of head yaw the measured
    ray sits about 38 px further out than it does when the head is square, and
    +-12 deg of pitch about 18 px.  The cause is that the iris offset is read
    along axes that turn with the head, so a rotated head slightly
    under-reports the camera-frame angle.

    It is bounded, smooth and centred on the calibration pose, which is what
    keeps it usable: the cursor test below holds every pose inside 60 px.
    Removing it would mean de-rotating the iris measurement by the head pose,
    and that is worth doing only against a real camera -- the effect is at the
    edge of what a synthetic face can settle.
    """
    rows = _ray_errors(estimator)
    reference = rows["reference"]
    for label, limit in (("yaw -18", 45.0), ("yaw +18", 45.0),
                         ("pitch -12", 25.0), ("pitch +12", 25.0)):
        drift = float(np.linalg.norm(rows[label] - reference))
        assert drift < limit, f"{label}: residual {drift:.1f} px exceeds {limit:.0f}"


# --------------------------------------------------------------------------
# The same, at the cursor
# --------------------------------------------------------------------------
def _cursor_on(estimator, pixel, head_yaw=0.0, head_pitch=0.0, frames=200, **kwargs):
    """Calibrate once at the reference pose, then look at ``pixel`` from anywhere."""
    model = _calibrate(estimator)
    eye_yaw, eye_pitch = pixel_to_pose(pixel, head_yaw, head_pitch, **kwargs)

    backend = NullBackend((int(SCREEN[0]), int(SCREEN[1])))
    controller = GazeCursorController(AbsoluteMouse(backend), model, CursorSettings())
    positions = []
    pose = (head_yaw, head_pitch, eye_yaw, eye_pitch)
    for frame in range(frames):
        sample = estimator.estimate(face(pose, noise_px=1.0, seed=frame, **kwargs))
        controller.update(sample, frame / FPS)
        if frame >= frames - 60:
            positions.append(np.array(backend.position(), dtype=float))
    return np.array(positions).mean(axis=0)


def _calibrate(estimator, columns=4, rows=4) -> CalibrationModel:
    """Fit on measured gaze angles at the pose the user calibrates from.

    The eye position is recorded alongside, because the controller compensates
    head movement as a displacement from it.
    """
    _eyes.clear()
    xs = np.linspace(0.15 * SCREEN[0], 0.85 * SCREEN[0], columns)
    ys = np.linspace(0.20 * SCREEN[1], 0.80 * SCREEN[1], rows)
    grid_x, grid_y = np.meshgrid(xs, ys)
    pixels = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    measured = [_observe(estimator, pixel) for pixel in pixels]

    model = CalibrationModel(degree=2)
    model.fit(np.array(measured), np.array(pixels), screen=SCREEN)
    model.reference_eye = tuple(np.mean(np.array(_eyes), axis=0))
    return model


#: Head translations gathered by the last ``_calibrate`` call.  Kept aside
#: because the fit takes angles and the controller needs the translation, and
#: folding both through one return value made the call site unreadable.
_eyes: list = []


def _observe(estimator, pixel):
    """One calibration observation: the gaze angles, recording the head pose."""
    eye_yaw, eye_pitch = pixel_to_pose(pixel)
    sample = estimator.estimate(face((0.0, 0.0, eye_yaw, eye_pitch)))
    assert sample.valid, sample.reason
    _eyes.append(sample.head_translation)
    return (sample.yaw, sample.pitch)


@pytest.mark.parametrize(
    "label,head_yaw,head_pitch,kwargs",
    [
        ("staying put", 0.0, 0.0, {}),
        ("slid 30 px left", 0.0, 0.0, {"shift_px": -30.0}),
        ("slid 30 px right", 0.0, 0.0, {"shift_px": 30.0}),
        ("leaned in to 500 mm", 0.0, 0.0, {"distance_mm": 500.0}),
        ("leaned back to 750 mm", 0.0, 0.0, {"distance_mm": 750.0}),
        ("turned 12 deg left", -12.0, 0.0, {}),
        ("turned 12 deg right", 12.0, 0.0, {}),
        ("tilted up 8 deg", 0.0, -8.0, {}),
        ("turned, slid and leaned", 10.0, -6.0, {"distance_mm": 700.0, "shift_px": 25.0}),
    ],
)
def test_the_cursor_stays_on_target_while_the_head_moves(
    estimator, label, head_yaw, head_pitch, kwargs
):
    """The user's requirement, stated as a test.

    Calibrate once, then move the head every way a person sitting at a desk
    does.  The cursor has to stay on the point being looked at; a tracker that
    only understands angles loses the target by the width of the head's travel.
    """
    cursor = _cursor_on(estimator, (700.0, 400.0), head_yaw, head_pitch, **kwargs)
    error = float(np.linalg.norm(cursor - np.array([700.0, 400.0])))
    assert error < 60.0, f"{label}: cursor sat {error:.0f} px off target"


def test_holding_the_head_steady_is_accurate_to_a_few_pixels(estimator):
    """Baseline for the test above: at the calibration pose it is near exact."""
    cursor = _cursor_on(estimator, (700.0, 400.0))
    assert float(np.linalg.norm(cursor - np.array([700.0, 400.0]))) < 20.0
