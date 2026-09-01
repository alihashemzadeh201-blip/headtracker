"""Synthetic face generator used to verify the gaze geometry numerically.

Instead of stubbing the geometry, this builds a real 3-D face, poses it at a
known yaw / pitch / roll and eye rotation, projects it through a pinhole camera
and hands the resulting pixel landmarks to the code under test.  Whatever the
estimator reports can then be compared against the pose that was asked for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from headtracker.geometry import (
    IRIS_ARM_RATIO,
    LEFT_IRIS,
    RIGHT_IRIS,
    REFINED_LANDMARK_COUNT,
)

#: Landmarks the estimator reads, in face-model coordinates.
#: Consistent with ``geometry.MODEL_POINTS``: y up, z out of the face.
FACE_POINTS: Dict[int, Tuple[float, float, float]] = {
    1: (0.0, 0.0, 0.0),  # nose tip
    152: (0.0, -330.0, -65.0),  # chin
    33: (-225.0, 170.0, -135.0),  # left eye outer corner
    263: (225.0, 170.0, -135.0),  # right eye outer corner
    61: (-150.0, -150.0, -125.0),  # left mouth corner
    291: (150.0, -150.0, -125.0),  # right mouth corner
    133: (-78.0, 172.0, -108.0),  # left eye inner corner
    362: (78.0, 172.0, -108.0),  # right eye inner corner
    159: (-150.0, 222.0, -112.0),  # left upper lid
    145: (-150.0, 122.0, -112.0),  # left lower lid
    386: (150.0, 222.0, -112.0),  # right upper lid
    374: (150.0, 122.0, -112.0),  # right lower lid
    # Eye aspect ratio landmarks (33,160,158,133,153,144 / 362,385,387,263,373,380)
    160: (-170.0, 214.0, -115.0),
    158: (-128.0, 214.0, -115.0),
    153: (-128.0, 128.0, -115.0),
    144: (-170.0, 128.0, -115.0),
    385: (170.0, 214.0, -115.0),
    387: (128.0, 214.0, -115.0),
    373: (128.0, 128.0, -115.0),
    380: (170.0, 128.0, -115.0),
}

#: Inter-ocular distance of the model face is 450 units ~ 62 mm, so one model
#: unit is about 0.138 mm.  A 12 mm eyeball and a 5.85 mm iris give the
#: IRIS_ARM_RATIO the estimator assumes.
EYEBALL_RADIUS = 12.0 / 0.138
IRIS_RADIUS = EYEBALL_RADIUS / IRIS_ARM_RATIO

EYE_CENTRES = {
    "left": np.array([-151.5, 171.0, -118.0]),
    "right": np.array([151.5, 171.0, -118.0]),
}


@dataclass
class SyntheticFace:
    """A posed face plus the ground truth that produced it."""

    points: np.ndarray
    head_yaw: float
    head_pitch: float
    head_roll: float
    eye_yaw: float
    eye_pitch: float
    distance_mm: float


def euler_to_rotation(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Model -> camera rotation.

    The base pose looks straight down the optical axis, which requires flipping
    the model's y and z axes (the model points up and out of the face, the
    camera points down and into the scene).

    Signs follow the estimator's convention, which matches screen pixels:
    positive yaw turns the gaze towards the right of the image and positive
    pitch turns it towards the bottom.  Yaw is negated because the model's x
    axis points at the subject's left, which is the camera's right.
    """
    base = np.diag([1.0, -1.0, -1.0])

    yaw, pitch, roll = np.radians(-yaw_deg), np.radians(pitch_deg), np.radians(roll_deg)
    r_roll = np.array(
        [
            [np.cos(roll), -np.sin(roll), 0.0],
            [np.sin(roll), np.cos(roll), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    r_pitch = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(pitch), -np.sin(pitch)],
            [0.0, np.sin(pitch), np.cos(pitch)],
        ]
    )
    r_yaw = np.array(
        [
            [np.cos(yaw), 0.0, np.sin(yaw)],
            [0.0, 1.0, 0.0],
            [-np.sin(yaw), 0.0, np.cos(yaw)],
        ]
    )
    return r_roll @ r_pitch @ r_yaw @ base


def _iris_block(centre: np.ndarray, eye_yaw_deg: float, eye_pitch_deg: float) -> np.ndarray:
    """Return the 5 landmarks of one iris for a given eye rotation."""
    yaw, pitch = np.radians(eye_yaw_deg), np.radians(eye_pitch_deg)
    direction = np.array([np.tan(yaw), -np.tan(pitch), 1.0])
    direction /= np.linalg.norm(direction)

    helper = np.array([0.0, 1.0, 0.0])
    if abs(float(np.dot(direction, helper))) > 0.9:
        helper = np.array([1.0, 0.0, 0.0])
    axis_a = np.cross(direction, helper)
    axis_a /= np.linalg.norm(axis_a)
    axis_b = np.cross(direction, axis_a)

    iris_centre = centre + EYEBALL_RADIUS * direction
    ring = [
        iris_centre + IRIS_RADIUS * (np.cos(a) * axis_a + np.sin(a) * axis_b)
        for a in (0.0, np.pi / 2, np.pi, 3 * np.pi / 2)
    ]
    return np.array([iris_centre] + ring, dtype=np.float64)


# Every parameter is one independently interesting knob on the synthetic face;
# tests set two or three at a time and rely on defaults for the rest.
def make_face(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    head_yaw: float = 0.0,
    head_pitch: float = 0.0,
    head_roll: float = 0.0,
    eye_yaw: float = 0.0,
    eye_pitch: float = 0.0,
    distance_mm: float = 600.0,
    width: int = 1280,
    height: int = 720,
    fov_deg: float = 60.0,
    eye_open: float = 1.0,
    noise_px: float = 0.0,
    seed: int = 0,
) -> SyntheticFace:
    """Build and project a posed face, returning its landmarks and ground truth."""
    model = {index: np.array(value, dtype=np.float64) for index, value in FACE_POINTS.items()}

    if eye_open != 1.0:
        # Pull the lids towards the eye centre line to simulate closing.
        lid_pairs = ((159, 145), (386, 374), (160, 144), (158, 153), (385, 380), (373, 387))
        for upper, lower in lid_pairs:
            centre_y = 0.5 * (model[upper][1] + model[lower][1])
            model[upper][1] = centre_y + (model[upper][1] - centre_y) * eye_open
            model[lower][1] = centre_y + (model[lower][1] - centre_y) * eye_open

    for offset, iris in enumerate((LEFT_IRIS, RIGHT_IRIS)):
        side = "left" if offset == 0 else "right"
        block = _iris_block(EYE_CENTRES[side], eye_yaw, eye_pitch)
        for slot, point in enumerate(block):
            model[iris[slot]] = point

    rotation = euler_to_rotation(head_yaw, head_pitch, head_roll)
    depth = distance_mm / 0.138
    translation = np.array([0.0, 0.0, depth])

    focal = (width / 2.0) / np.tan(np.radians(fov_deg) / 2.0)
    points = np.zeros((REFINED_LANDMARK_COUNT, 2), dtype=np.float64)
    for index, point in model.items():
        camera = rotation @ point + translation
        points[index, 0] = focal * camera[0] / camera[2] + width / 2.0
        points[index, 1] = focal * camera[1] / camera[2] + height / 2.0

    if noise_px > 0:
        rng = np.random.default_rng(seed)
        points += rng.normal(0.0, noise_px, size=points.shape)

    return SyntheticFace(
        points=points,
        head_yaw=head_yaw,
        head_pitch=head_pitch,
        head_roll=head_roll,
        eye_yaw=eye_yaw,
        eye_pitch=eye_pitch,
        distance_mm=distance_mm,
    )
