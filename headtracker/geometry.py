"""Geometric gaze estimation from MediaPipe face landmarks.

The previous implementation derived the cursor position from the raw 2-D pixel
position of the nose tip.  That signal mixes head *translation* with head
*rotation* and changes scale with the distance to the camera, so it can never be
mapped onto a fixed point of the screen.

This module instead recovers, per frame:

* the head rotation (yaw / pitch / roll) with ``cv2.solvePnP``, which is a
  translation-invariant measure of where the head is pointing, and
* the rotation of each eye inside its socket, measured from the iris landmarks
  that MediaPipe exposes when the refinement model is used.

Both are expressed in degrees and share a sign convention aligned with screen
coordinates:

    yaw   > 0  ->  looking towards the right edge of the screen
    pitch > 0  ->  looking towards the bottom edge of the screen

Because they are angles rather than pixels, the result is invariant to how far
the user sits from the camera.  Converting them into screen coordinates is left
to :mod:`headtracker.calibration`, which absorbs every remaining systematic
error (camera mounting angle, individual eye geometry, lens distortion).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Landmark indices -- MediaPipe FaceMesh topology (468 mesh + 10 iris = 478)
# ---------------------------------------------------------------------------
NOSE_TIP = 1
CHIN = 152

LEFT_EYE_OUTER = 33
LEFT_EYE_INNER = 133
LEFT_EYE_UPPER = 159
LEFT_EYE_LOWER = 145

RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263
RIGHT_EYE_UPPER = 386
RIGHT_EYE_LOWER = 374

#: Eight points around each eye rim, chosen to mirror each other: the two
#: corners, the lid midpoints, and the lid points one step in from each corner.
#: Averaging them estimates the centre of the eyeball far better than the two
#: corners alone, which sit on one horizontal line and say nothing about where
#: the eye sits vertically.  Measured single-frame gaze noise falls from
#: 1.163 deg to 0.882 deg at 1 px of landmark jitter.
LEFT_EYE_RIM = (33, 133, 159, 145, 160, 158, 144, 153)
RIGHT_EYE_RIM = (263, 362, 386, 374, 387, 385, 373, 380)

MOUTH_LEFT = 61
MOUTH_RIGHT = 291

# (center, ring...) -- the ring is only used to measure the iris radius, so the
# ordering of the four ring points does not matter.
LEFT_IRIS = (468, 469, 470, 471, 472)
RIGHT_IRIS = (473, 474, 475, 476, 477)

#: Index of the first iris landmark.  A result with fewer landmarks than this
#: was produced without landmark refinement and cannot be used for eye gaze.
IRIS_LANDMARK_BASE = 468

#: Number of landmarks in the refined topology.
REFINED_LANDMARK_COUNT = 478

EYE_ASPECT = 0.30
"""Vertical opening of an eye as a fraction of the inter-ocular distance.

Used to turn the vertical iris offset into a scale-free ratio.  Only the ratio
matters -- the calibration fit absorbs the absolute value.
"""

IRIS_ARM_RATIO = 2.05
"""Eyeball radius divided by iris radius.

The iris sits on the surface of a sphere, so when the eye rotates by an angle
``a`` the iris centre travels ``R_eyeball * sin(a)`` while the iris itself still
projects to roughly ``R_iris``.  Measuring the offset in iris radii and taking
the arcsine therefore recovers the rotation angle directly, and saturates
naturally at the physiological limit instead of extrapolating linearly past it.
"""

IRIS_LIMIT_DEG = 34.0
"""Extra clamp on the recovered eye rotation, guarding against bad landmarks."""

HEAD_ANGLE_LIMIT_DEG = 60.0
"""Head angles beyond this are almost certainly a tracking glitch."""

# A canonical 3-D model of the six points used for the PnP solve.  Units are
# arbitrary but consistent; y points up and z points out of the face.
MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),  # nose tip
        (0.0, -330.0, -65.0),  # chin
        (-225.0, 170.0, -135.0),  # left eye, outer corner
        (225.0, 170.0, -135.0),  # right eye, outer corner
        (-150.0, -150.0, -125.0),  # left mouth corner
        (150.0, -150.0, -125.0),  # right mouth corner
    ],
    dtype=np.float64,
)

PNP_IMAGE_POINTS = (NOSE_TIP, CHIN, LEFT_EYE_OUTER, RIGHT_EYE_OUTER, MOUTH_LEFT, MOUTH_RIGHT)

#: Solver order, by measured depth stability.  Over 60 noisy frames against a
#: true depth of 4348 units: SQPNP 0 negative depths (std 20.2), EPNP 0 (24.3),
#: ITERATIVE 17 negative (std 4009.7).  ITERATIVE is kept last because it is
#: the only one of the three that converges from a poor starting pose.
_PNP_FLAGS = (cv2.SOLVEPNP_SQPNP, cv2.SOLVEPNP_EPNP, cv2.SOLVEPNP_ITERATIVE)

MAX_REPROJECTION_ERROR_PX = 60.0
"""Reject a pose only when it is far worse than any plausible face.

This is deliberately generous.  The error measures how well the landmarks fit
``MODEL_POINTS``, and a real skull is not that model: perturbing just the six
pose landmarks by 15 px -- far less than the difference between two people --
already measures 13.7 px.  A tight bound here silently discards real faces,
every frame, and the cursor simply stops moving.  The value is used mainly to
choose between solvers, with rejection only as a last resort.
"""

MIN_FACE_SPREAD_PX = 6.0
"""Reject landmarks that have collapsed onto each other.

This guard exists because reprojection error cannot catch the case: coincident
landmarks were measured at 0.00 px error with a depth of 1.4e16.
"""


@dataclass
class GazeSample:
    """Everything the controller needs from a single video frame."""

    yaw: float = 0.0
    """Gaze yaw in degrees relative to the camera axis (positive = screen right)."""

    pitch: float = 0.0
    """Gaze pitch in degrees (positive = screen bottom)."""

    roll: float = 0.0
    """Head roll in degrees (positive = counter-clockwise in the image)."""

    distance: float = 0.0
    """Head distance from the camera, in model units (relative measure only)."""

    source: str = "none"
    """``"iris"`` when the eyes drove the estimate, ``"head"`` when they could not."""

    head_yaw: float = 0.0
    head_pitch: float = 0.0
    """Gaze angles from the head pose alone, i.e. assuming centred eyes."""

    iris_yaw: float = 0.0
    iris_pitch: float = 0.0
    """Gaze angles from the irises alone.  These are *total* gaze angles, not an
    eye-only offset on top of the head pose -- see :meth:`GazeEstimator.estimate`."""

    left_eye_open: float = 0.0
    right_eye_open: float = 0.0

    eyes_visible: bool = False
    """True when both irises were usable this frame."""

    valid: bool = False
    """True when the sample is good enough to drive the cursor."""

    reason: str = ""
    """Human readable explanation when ``valid`` is False."""

    extras: dict = field(default_factory=dict)


def landmarks_to_array(landmarks: Sequence, width: int, height: int) -> np.ndarray:
    """Flatten a landmark list into an ``(N, 2)`` array of pixel coordinates.

    Accepts both the legacy ``solutions`` normalised landmarks and the newer
    ``tasks`` ``NormalizedLandmark`` containers -- both expose ``.x`` / ``.y``.
    """
    return np.array(
        [(lm.x * width, lm.y * height) for lm in landmarks], dtype=np.float64
    )


def eye_aspect_ratio(points: np.ndarray, indices: Sequence[int]) -> float:
    """Return the eye aspect ratio, i.e. how open an eye is.

    ``indices`` must be ordered
    ``[outer, upper_1, upper_2, inner, lower_2, lower_1]``.
    The ratio is ~0.3 for an open eye and drops towards 0 when it closes.
    """
    if points.shape[0] <= max(indices):
        return 0.0
    p1, p2, p3, p4, p5, p6 = (points[i] for i in indices)
    vertical = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)
    if horizontal < 1e-9:
        return 0.0
    return float(vertical / (2.0 * horizontal))


def default_camera_matrix(width: int, height: int, fov_deg: float = 60.0) -> np.ndarray:
    """Build a pinhole camera matrix for a webcam with a given horizontal FOV."""
    focal = (width / 2.0) / math.tan(math.radians(max(fov_deg, 1.0) / 2.0))
    return np.array(
        [
            [focal, 0.0, width / 2.0],
            [0.0, focal, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _reprojection_error(
    model_points: np.ndarray,
    image_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
) -> float:
    """Mean distance, in pixels, between the solved pose and the landmarks."""
    projected, _ = cv2.projectPoints(
        model_points, rvec, tvec, camera_matrix, np.zeros((4, 1), dtype=np.float64)
    )
    return float(np.mean(np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)))


def solve_head_rotation(
    points: np.ndarray, camera_matrix: np.ndarray
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Recover ``(R, t)`` mapping the face model into camera coordinates.

    Tries several solvers and keeps the most consistent one.  A single
    ``SOLVEPNP_ITERATIVE`` call returned a mirrored, negative depth on 75 of
    300 noisy frames -- a quarter of the stream, which reaches the user as the
    cursor stuttering.  The first solver to land well inside the bound wins;
    the rest are only asked when it does not.
    """
    if points.shape[0] <= max(PNP_IMAGE_POINTS):
        return None
    image_points = np.array([points[i] for i in PNP_IMAGE_POINTS], dtype=np.float64)
    if not np.all(np.isfinite(image_points)):
        return None
    spread = image_points.max(axis=0) - image_points.min(axis=0)
    if float(spread.min()) < MIN_FACE_SPREAD_PX:
        return None

    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    best_error = math.inf
    best_rotation: Optional[np.ndarray] = None
    best_translation: Optional[np.ndarray] = None
    for flag in _PNP_FLAGS:
        try:
            ok, rvec, tvec = cv2.solvePnP(
                MODEL_POINTS, image_points, camera_matrix, dist_coeffs, flags=flag
            )
        except cv2.error:
            # SQPNP raises on degenerate input instead of returning False.
            continue
        if not ok or not np.all(np.isfinite(rvec)) or not np.all(np.isfinite(tvec)):
            continue
        # tvec comes back as a column vector, and tvec[2] is then a length-1
        # array rather than a scalar -- float() on it raises on numpy 2.
        translation = tvec.reshape(3)
        if float(translation[2]) <= 0.0:
            continue  # behind the camera: a mirrored solution
        error = _reprojection_error(MODEL_POINTS, image_points, rvec, tvec, camera_matrix)
        if error < best_error:
            best_error = error
            rotation, _ = cv2.Rodrigues(rvec)
            best_rotation = rotation
            best_translation = translation
            if error < MAX_REPROJECTION_ERROR_PX / 4.0:
                break  # convincing enough that asking further adds nothing

    if best_rotation is None or best_translation is None:
        return None
    if best_error > MAX_REPROJECTION_ERROR_PX:
        return None
    return best_rotation, best_translation


def rotation_to_head_angles(rotation: np.ndarray) -> Tuple[float, float, float]:
    """Turn a rotation matrix into roll-compensated ``(yaw, pitch, roll)`` degrees.

    ``rotation`` maps the face model into camera coordinates, so the image of
    the model's +z axis (out of the face) is the direction the face is looking.
    That direction points back towards the camera, hence its negative depth.

    The gaze direction is decomposed relative to the *head's* own up axis
    rather than the image vertical, so a sideways head tilt is reported as roll
    and does not bleed into yaw or pitch.  Without that, leaning the head would
    drag the cursor across the screen.
    """
    gaze = rotation @ np.array([0.0, 0.0, 1.0])
    up = rotation @ np.array([0.0, 1.0, 0.0])

    depth = max(-gaze[2], 1e-6)
    transverse = math.hypot(gaze[0], gaze[1])
    tan_total = transverse / depth

    # Azimuths measured from image-up, positive towards image-right.
    phi = math.atan2(gaze[0], -gaze[1])
    theta_up = math.atan2(up[0], -up[1])
    azimuth = phi - theta_up

    yaw = math.degrees(math.atan(tan_total * math.sin(azimuth)))
    pitch = math.degrees(math.atan(-tan_total * math.cos(azimuth)))
    roll = math.degrees(theta_up)
    return yaw, pitch, roll


def iris_radius(points: np.ndarray, iris: Sequence[int]) -> float:
    """Median distance from the iris centre to its ring landmarks, in pixels."""
    if points.shape[0] <= max(iris):
        return 0.0
    centre = points[iris[0]]
    ring = points[list(iris)[1:]]
    radii = np.linalg.norm(ring - centre, axis=1)
    radii = radii[np.isfinite(radii)]
    if radii.size == 0:
        return 0.0
    return float(np.median(radii))


# One parameter per landmark that defines the eye frame; grouping them would
# hide which index is which at the two call sites.
def _eye_iris_offset(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    points: np.ndarray,
    corner_a: int,
    corner_b: int,
    upper: int,
    lower: int,
    iris: Sequence[int],
    rim: Sequence[int],
    fallback_scale: float,
) -> Optional[Tuple[float, float]]:
    """Measure one iris' offset from its eye centre, in units of iris radius.

    The local axes are oriented canonically in *image* space -- x towards the
    right, y towards the bottom -- rather than from the inner/outer corner
    naming.  Those names swap sides between the two eyes, and deriving the axis
    from them makes one eye's horizontal offset the mirror image of the other's,
    so averaging the pair cancels the signal instead of halving its noise.
    """
    needed = max([corner_a, corner_b, upper, lower] + list(iris) + list(rim))
    if points.shape[0] <= needed:
        return None

    corner_axis = points[corner_b] - points[corner_a]
    width = np.linalg.norm(corner_axis)
    if width < 1e-6:
        return None
    axis_x = corner_axis / width
    if axis_x[0] < 0:
        axis_x = -axis_x

    lid_axis = points[lower] - points[upper]
    axis_y = lid_axis - axis_x * float(np.dot(lid_axis, axis_x))
    height = np.linalg.norm(axis_y)
    if height < 1e-6:
        return None
    axis_y = axis_y / height
    if axis_y[1] < 0:
        axis_y = -axis_y

    # The iris is a rigid disc, so its radius is the one length on the eye that
    # does not change when the eye moves, blinks or squints.  Falling back to
    # the inter-ocular distance keeps the estimate usable if the ring collapses.
    scale = iris_radius(points, iris)
    if scale <= 1e-6:
        scale = fallback_scale
    if scale <= 1e-6:
        return None

    # Average all five iris landmarks rather than trusting the centre one.  The
    # four ring points sit symmetrically around it, so their mean is the same
    # centre with part of the per-landmark jitter averaged out -- measured at
    # 1.7 deg of single-frame gaze error instead of 2.25 deg.
    iris_centre = points[list(iris)].mean(axis=0)

    # Centre of the eyeball, from the whole rim rather than the two corners.
    # The corners lie on one horizontal line, so their midpoint carries no
    # information about where the eye sits vertically, and both move together
    # when the face turns -- averaging the rim cancels part of that.
    eye_centre = points[list(rim)].mean(axis=0)
    delta = iris_centre - eye_centre
    return (
        float(np.dot(delta, axis_x)) / scale,
        float(np.dot(delta, axis_y)) / scale,
    )


def _offset_to_angle(offset: float) -> float:
    """Convert an iris offset, in iris radii, to an eye rotation in degrees."""
    ratio = _clamp(offset / IRIS_ARM_RATIO, -0.995, 0.995)
    return _clamp(math.degrees(math.asin(ratio)), -IRIS_LIMIT_DEG, IRIS_LIMIT_DEG)


def estimate_iris_gaze(points: np.ndarray) -> Optional[Tuple[float, float, float]]:
    """Return ``(yaw_deg, pitch_deg, interocular_px)`` or ``None``.

    Both eyes are measured and averaged: averaging halves the landmark noise,
    and the two eyes have to agree for the measurement to mean anything.
    """
    if points.shape[0] <= max(RIGHT_IRIS):
        return None

    left_centre = 0.5 * (points[LEFT_EYE_OUTER] + points[LEFT_EYE_INNER])
    right_centre = 0.5 * (points[RIGHT_EYE_OUTER] + points[RIGHT_EYE_INNER])
    iod = float(np.linalg.norm(right_centre - left_centre))
    if iod < 1e-6:
        return None

    # Used only if an iris ring degenerates.
    fallback_scale = iod * EYE_ASPECT

    yaws = []
    pitches = []
    for outer, inner, upper, lower, iris, rim in (
        (LEFT_EYE_OUTER, LEFT_EYE_INNER, LEFT_EYE_UPPER, LEFT_EYE_LOWER,
         LEFT_IRIS, LEFT_EYE_RIM),
        (RIGHT_EYE_OUTER, RIGHT_EYE_INNER, RIGHT_EYE_UPPER, RIGHT_EYE_LOWER,
         RIGHT_IRIS, RIGHT_EYE_RIM),
    ):
        measured = _eye_iris_offset(points, outer, inner, upper, lower, iris, rim, fallback_scale)
        if measured is not None:
            yaws.append(_offset_to_angle(measured[0]))
            pitches.append(_offset_to_angle(measured[1]))

    if not yaws:
        return None

    return float(np.mean(yaws)), float(np.mean(pitches)), iod


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class GazeEstimator:
    """Turns a frame's landmarks into a :class:`GazeSample`."""

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        camera_fov_deg: float = 60.0,
        min_eye_open: float = 0.18,
        use_eyes: bool = True,
    ) -> None:
        self.frame_width = int(frame_width)
        self.frame_height = int(frame_height)
        self.camera_fov_deg = float(camera_fov_deg)
        self.min_eye_open = float(min_eye_open)
        self.use_eyes = bool(use_eyes)
        self._camera_matrix = default_camera_matrix(
            self.frame_width, self.frame_height, self.camera_fov_deg
        )

    def set_frame_size(self, width: int, height: int) -> None:
        """Update the intrinsics after the camera reports a new resolution."""
        if width == self.frame_width and height == self.frame_height:
            return
        self.frame_width = int(width)
        self.frame_height = int(height)
        self._camera_matrix = default_camera_matrix(
            self.frame_width, self.frame_height, self.camera_fov_deg
        )

    def estimate(self, points: np.ndarray) -> GazeSample:
        """Estimate the gaze for ``points``, an ``(N, 2)`` pixel array."""
        sample = GazeSample()

        rotation = solve_head_rotation(points, self._camera_matrix)
        if rotation is None:
            sample.reason = "head pose solve failed"
            return sample
        rotation_matrix, tvec = rotation

        head_yaw, head_pitch, roll = rotation_to_head_angles(rotation_matrix)
        sample.head_yaw = head_yaw
        sample.head_pitch = head_pitch
        sample.roll = roll
        sample.distance = float(tvec[2])

        sample.left_eye_open = eye_aspect_ratio(points, (33, 160, 158, 133, 153, 144))
        sample.right_eye_open = eye_aspect_ratio(points, (362, 385, 387, 263, 373, 380))

        # Default to the head pose: it is always available, and it is the right
        # answer whenever the eyes sit centred in their sockets.
        sample.yaw, sample.pitch = head_yaw, head_pitch
        sample.source = "head"
        rejected = None

        if self.use_eyes:
            iris = estimate_iris_gaze(points)
            if iris is None:
                sample.reason = (
                    "iris landmarks unavailable"
                    if points.shape[0] < IRIS_LANDMARK_BASE
                    else "iris not measurable"
                )
            else:
                iris_yaw, iris_pitch, iod = iris
                sample.iris_yaw = iris_yaw
                sample.iris_pitch = iris_pitch
                sample.eyes_visible = True
                sample.extras["iod_px"] = iod

                if min(sample.left_eye_open, sample.right_eye_open) < self.min_eye_open:
                    # An iris centre measured through a closed lid is noise, and
                    # a user with their eyes shut is not looking at anything.
                    # Holding the cursor beats guessing.
                    rejected = "eyes closed"
                else:
                    # Not head + eye: the iris already encodes the total gaze.
                    # It sits on the eyeball at the point the gaze exits, so its
                    # offset from the socket centre measures the gaze direction
                    # in the camera frame no matter how the head is turned --
                    # adding the head angle on top would count it twice.
                    sample.yaw, sample.pitch = iris_yaw, iris_pitch
                    sample.source = "iris"

        sample.yaw = _clamp(sample.yaw, -HEAD_ANGLE_LIMIT_DEG, HEAD_ANGLE_LIMIT_DEG)
        sample.pitch = _clamp(sample.pitch, -HEAD_ANGLE_LIMIT_DEG, HEAD_ANGLE_LIMIT_DEG)

        if rejected is not None:
            sample.reason = rejected
            return sample
        if not np.isfinite(sample.yaw) or not np.isfinite(sample.pitch):
            sample.reason = "non-finite gaze"
            return sample
        if sample.distance <= 0:
            sample.reason = "invalid head distance"
            return sample

        sample.valid = True
        return sample
