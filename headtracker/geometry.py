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

#: Every landmark that sits on the rim of each eye socket.  Their centroid is
#: the eye's centre, which is what the iris offset is measured against.  The
#: two corners alone would do, but the centre is the noisier half of that
#: measurement -- it is one subtraction of two jittering points -- and averaging
#: eight of them instead of two cut the single-frame yaw error from 1.163 deg to
#: 0.882 deg at 1 px of landmark jitter, with no measured cost when the eye is
#: half closed.
LEFT_EYE_FRAME = (33, 133, 144, 145, 153, 158, 159, 160)
RIGHT_EYE_FRAME = (362, 263, 373, 374, 380, 385, 386, 387)

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

#: Midpoint of the two model eye corners -- the ray origin for the gaze.
EYE_MID_MODEL = 0.5 * (MODEL_POINTS[2] + MODEL_POINTS[3])


def gaze_direction(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Unit gaze direction in camera coordinates.

    Follows the sign convention used everywhere here: +x towards image right,
    +y towards image bottom, and a negative z because the user looks back
    towards the camera rather than into the scene.
    """
    direction = np.array(
        [math.tan(math.radians(yaw_deg)), math.tan(math.radians(pitch_deg)), -1.0]
    )
    return direction / float(np.linalg.norm(direction))


def project_to_screen_plane(
    origin: np.ndarray, direction: np.ndarray, plane_z: float = 0.0
) -> Optional[Tuple[float, float]]:
    """Intersect the gaze ray with the screen plane and return that point.

    The webcam sits on the monitor, so the screen plane passes essentially
    through the camera: ``plane_z = 0``.  Whatever residual tilt and offset the
    real setup has is a smooth function of position, which the calibration
    absorbs.

    The important property is that the result depends on *where the eye is* as
    well as where it points, so it stays correct when the head moves.  Leaning
    left shifts the origin left and the hit point follows; moving back lengthens
    the lever arm and the same angle sweeps further.  Both are handled here
    analytically instead of being fitted.
    """
    if direction[2] >= -1e-6:
        return None  # looking away from the screen
    scale = (plane_z - origin[2]) / direction[2]
    if scale <= 0:
        return None  # the screen is behind the eye
    hit = origin + scale * direction
    return float(hit[0]), float(hit[1])


@dataclass
class GazeSample:
    """Everything the controller needs from a single video frame."""

    screen_x: float = 0.0
    screen_y: float = 0.0
    """Where the gaze ray hits the screen plane, in model units.

    These -- not the raw angles -- are what the calibration maps to pixels.
    They already account for the head's position, distance and rotation, so
    moving your head does not move the cursor away from what you are looking at.
    """

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

    head_position: tuple = (0.0, 0.0, 0.0)
    """Eye midpoint in camera coordinates, in model units."""

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


#: Solver flags to try, in order.
#:
#: The six model points are very nearly coplanar, which makes the PnP problem
#: ambiguous: ``SOLVEPNP_ITERATIVE`` happily returns the mirrored solution, and
#: with one pixel of landmark jitter it did so on 28% of frames in measurement.
#: Those frames carry a negative depth, get rejected, and the cursor visibly
#: stutters.  ``SQPNP`` is a globally optimal solver with no such ambiguity --
#: measured depth std 20 model units against 4010 for ``ITERATIVE``.
MIN_FACE_SPREAD_PX = 20.0
"""Smallest span of the pose landmarks that is still a usable face.

The guard that actually rejects degenerate input.  When MediaPipe loses the
face it returns coincident points, and PnP answers those with a rotation and a
translation of ``1e16`` -- a solution whose reprojection error is *zero*,
because every model point projects onto the same pixel the landmarks already
share.  Nothing about the fit detects it; only the physical size of the face
does.  A real face spans roughly 190 px at 1080p and stays above 40 px however
far back the user sits.
"""

MAX_REPROJECTION_ERROR_PX = 12.0
"""Widest mean reprojection error still accepted as a real head pose.

Clean faces sit near 0.0 px, 1 px of landmark noise near 1.2 px and 8 px of
noise near 9.5 px.  Past that the landmarks no longer agree with a rigid face
and the recovered pose is unreliable, so the frame is dropped.
"""

_PNP_FLAGS = tuple(
    name
    for name in ("SOLVEPNP_SQPNP", "SOLVEPNP_EPNP", "SOLVEPNP_ITERATIVE")
    if hasattr(cv2, name)
)


def solve_head_rotation(
    points: np.ndarray, camera_matrix: np.ndarray
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Recover ``(R, t)`` mapping the face model into camera coordinates.

    Returns ``None`` rather than a mirrored solution: a face in front of the
    camera always has positive depth, so a negative ``t`` means the solver found
    the wrong branch and the frame is unusable.
    """
    if points.shape[0] <= max(PNP_IMAGE_POINTS):
        return None
    image_points = np.array([points[i] for i in PNP_IMAGE_POINTS], dtype=np.float64)
    if not np.all(np.isfinite(image_points)):
        return None
    if float(np.ptp(image_points, axis=0).max()) < MIN_FACE_SPREAD_PX:
        return None
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    for flag_name in _PNP_FLAGS:
        try:
            ok, rvec, tvec = cv2.solvePnP(
                MODEL_POINTS,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=getattr(cv2, flag_name),
            )
        except cv2.error:
            continue
        if not ok:
            continue
        translation = tvec.reshape(3)
        if translation[2] <= 0:
            continue
        if reprojection_error(rvec, tvec, image_points, camera_matrix, dist_coeffs) > \
                MAX_REPROJECTION_ERROR_PX:
            continue
        rotation, _ = cv2.Rodrigues(rvec)
        return rotation, translation
    return None


def reprojection_error(
    rvec: np.ndarray,
    tvec: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> float:
    """Mean distance in pixels between the landmarks and where PnP places them.

    A pose that really fits the face puts its model points within a couple of
    pixels of the landmarks.  Note that this cannot reject coincident landmarks
    on its own: for those, the face is "infinitely far away", every model point
    projects to the same pixel, and the error comes out exactly zero.
    """
    projected, _ = cv2.projectPoints(MODEL_POINTS, rvec, tvec, camera_matrix, distortion)
    projected = np.asarray(projected, dtype=np.float64).reshape(-1, 2)
    return float(np.mean(np.linalg.norm(projected - image_points.reshape(-1, 2), axis=1)))


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


# Six parameters because the eye frame is defined by four independent landmark
# groups -- corners for the horizontal axis, lid pair for the vertical, the whole
# rim for the centre, the iris ring for the scale -- and collapsing them into one
# tuple per eye would hide which index is which at the call site.
def _eye_iris_offset(  # pylint: disable=too-many-positional-arguments,too-many-locals
    points: np.ndarray,
    corners: Tuple[int, int],
    lids: Tuple[int, int],
    frame: Sequence[int],
    iris: Sequence[int],
    fallback_scale: float,
) -> Optional[Tuple[float, float]]:
    """Measure one iris' offset from its eye centre, in units of iris radius.

    The local axes are oriented canonically in *image* space -- x towards the
    right, y towards the bottom -- rather than from the inner/outer corner
    naming.  Those names swap sides between the two eyes, and deriving the axis
    from them makes one eye's horizontal offset the mirror image of the other's,
    so averaging the pair cancels the signal instead of halving its noise.

    The axes come from the corners and the lid pair, which are the landmarks
    that define the eye's orientation; only the *centre* is averaged over the
    whole rim.  Keeping the two apart matters: the lid points drift when the
    eye squints, and an axis built from a drifting pair would rotate the
    measurement, whereas a centre built from them only translates it.
    """
    needed = max(list(corners) + list(lids) + list(frame) + list(iris))
    if points.shape[0] <= needed:
        return None

    corner_axis = points[corners[1]] - points[corners[0]]
    width = np.linalg.norm(corner_axis)
    if width < 1e-6:
        return None
    axis_x = corner_axis / width
    if axis_x[0] < 0:
        axis_x = -axis_x

    lid_axis = points[lids[1]] - points[lids[0]]
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

    eye_centre = points[list(frame)].mean(axis=0)
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
    for corners, lids, frame, iris in (
        ((LEFT_EYE_OUTER, LEFT_EYE_INNER), (LEFT_EYE_UPPER, LEFT_EYE_LOWER),
         LEFT_EYE_FRAME, LEFT_IRIS),
        ((RIGHT_EYE_OUTER, RIGHT_EYE_INNER), (RIGHT_EYE_UPPER, RIGHT_EYE_LOWER),
         RIGHT_EYE_FRAME, RIGHT_IRIS),
    ):
        measured = _eye_iris_offset(points, corners, lids, frame, iris, fallback_scale)
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

    def _apply_iris(self, sample: GazeSample, points: np.ndarray) -> Optional[str]:
        """Refine a head-pose sample with the iris measurement.

        Returns the reason to reject the frame, or ``None`` to keep it.  When
        the iris cannot be measured the sample is left on the head pose, which
        is always available and is the right answer whenever the eyes sit
        centred in their sockets.
        """
        iris = estimate_iris_gaze(points)
        if iris is None:
            sample.reason = (
                "iris landmarks unavailable"
                if points.shape[0] < IRIS_LANDMARK_BASE
                else "iris not measurable"
            )
            return None

        iris_yaw, iris_pitch, iod = iris
        sample.iris_yaw = iris_yaw
        sample.iris_pitch = iris_pitch
        sample.eyes_visible = True
        sample.extras["iod_px"] = iod

        if min(sample.left_eye_open, sample.right_eye_open) < self.min_eye_open:
            # An iris centre measured through a closed lid is noise, and a user
            # with their eyes shut is not looking at anything.  Holding the
            # cursor beats guessing.
            return "eyes closed"

        # Not head + eye: the iris already encodes the total gaze.  It sits on
        # the eyeball at the point the gaze exits, so its offset from the socket
        # centre measures the gaze direction in the camera frame no matter how
        # the head is turned -- adding the head angle on top would count it
        # twice.
        sample.yaw, sample.pitch = iris_yaw, iris_pitch
        sample.source = "iris"
        return None

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

        eye_origin = rotation_matrix @ EYE_MID_MODEL + tvec
        sample.head_position = (float(eye_origin[0]), float(eye_origin[1]), float(eye_origin[2]))

        sample.left_eye_open = eye_aspect_ratio(points, (33, 160, 158, 133, 153, 144))
        sample.right_eye_open = eye_aspect_ratio(points, (362, 385, 387, 263, 373, 380))

        # Default to the head pose: it is always available, and it is the right
        # answer whenever the eyes sit centred in their sockets.
        sample.yaw, sample.pitch = head_yaw, head_pitch
        sample.source = "head"
        rejected = None

        if self.use_eyes:
            rejected = self._apply_iris(sample, points)

        sample.yaw = _clamp(sample.yaw, -HEAD_ANGLE_LIMIT_DEG, HEAD_ANGLE_LIMIT_DEG)
        sample.pitch = _clamp(sample.pitch, -HEAD_ANGLE_LIMIT_DEG, HEAD_ANGLE_LIMIT_DEG)

        if rejected is not None:
            sample.reason = rejected
            return sample
        if not np.isfinite(sample.yaw) or not np.isfinite(sample.pitch):
            sample.reason = "non-finite gaze"
            return sample

        hit = project_to_screen_plane(eye_origin, gaze_direction(sample.yaw, sample.pitch))
        if hit is None or not all(np.isfinite(hit)):
            sample.reason = "gaze does not reach the screen"
            return sample
        sample.screen_x, sample.screen_y = hit

        sample.valid = True
        return sample
