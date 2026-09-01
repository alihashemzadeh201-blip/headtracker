"""Numerical verification of the gaze geometry against projected synthetic faces."""

from __future__ import annotations

import math

import numpy as np
import pytest

from headtracker.filters import OneEuroFilter
from headtracker.geometry import (
    GazeEstimator,
    default_camera_matrix,
    estimate_iris_gaze,
    eye_aspect_ratio,
    rotation_to_head_angles,
    solve_head_rotation,
)
from tests.synthetic_face import make_face

WIDTH, HEIGHT, FOV = 1280, 720, 60.0


@pytest.fixture(name="estimator")
def fixture_estimator() -> GazeEstimator:
    return GazeEstimator(WIDTH, HEIGHT, camera_fov_deg=FOV)


# --------------------------------------------------------------------------
# Head pose
# --------------------------------------------------------------------------
def test_straight_ahead_face_yields_zero_angles():
    face = make_face(distance_mm=600)
    camera = default_camera_matrix(WIDTH, HEIGHT, FOV)
    rotation, tvec = solve_head_rotation(face.points, camera)
    yaw, pitch, roll = rotation_to_head_angles(rotation)
    assert yaw == pytest.approx(0.0, abs=0.5)
    assert pitch == pytest.approx(0.0, abs=0.5)
    assert roll == pytest.approx(0.0, abs=0.5)
    assert tvec[2] == pytest.approx(600 / 0.138, rel=0.05)


@pytest.mark.parametrize("angle", [-30.0, -15.0, 0.0, 15.0, 30.0])
def test_head_yaw_is_recovered(angle):
    face = make_face(head_yaw=angle, distance_mm=600)
    camera = default_camera_matrix(WIDTH, HEIGHT, FOV)
    rotation, _ = solve_head_rotation(face.points, camera)
    yaw, _, _ = rotation_to_head_angles(rotation)
    assert yaw == pytest.approx(angle, abs=1.5)


@pytest.mark.parametrize("angle", [-20.0, -8.0, 0.0, 8.0, 20.0])
def test_head_pitch_is_recovered(angle):
    face = make_face(head_pitch=angle, distance_mm=600)
    camera = default_camera_matrix(WIDTH, HEIGHT, FOV)
    rotation, _ = solve_head_rotation(face.points, camera)
    _, pitch, _ = rotation_to_head_angles(rotation)
    assert pitch == pytest.approx(angle, abs=1.5)


def test_roll_does_not_leak_into_yaw():
    """A sideways head tilt must not be read as a change of gaze direction.

    This is the failure the roll de-rotation exists to prevent: without it the
    cursor slides across the screen whenever the user leans their head.
    """
    camera = default_camera_matrix(WIDTH, HEIGHT, FOV)
    upright = make_face(head_yaw=10.0, head_roll=0.0)
    tilted = make_face(head_yaw=10.0, head_roll=25.0)

    yaw_upright, pitch_upright, roll_upright = rotation_to_head_angles(
        solve_head_rotation(upright.points, camera)[0]
    )
    yaw_tilted, pitch_tilted, roll_tilted = rotation_to_head_angles(
        solve_head_rotation(tilted.points, camera)[0]
    )

    assert roll_tilted - roll_upright == pytest.approx(25.0, abs=2.0)
    assert yaw_tilted == pytest.approx(yaw_upright, abs=2.5)
    assert pitch_tilted == pytest.approx(pitch_upright, abs=2.5)


# --------------------------------------------------------------------------
# Iris gaze
# --------------------------------------------------------------------------
@pytest.mark.parametrize("angle", [-25.0, -12.0, 0.0, 12.0, 25.0])
def test_eye_yaw_is_recovered(angle):
    face = make_face(eye_yaw=angle)
    measured = estimate_iris_gaze(face.points)
    assert measured is not None
    yaw, _, _ = measured
    assert yaw == pytest.approx(angle, abs=3.5)


@pytest.mark.parametrize("angle", [-18.0, -7.0, 0.0, 7.0, 18.0])
def test_eye_pitch_is_recovered(angle):
    face = make_face(eye_pitch=angle)
    measured = estimate_iris_gaze(face.points)
    assert measured is not None
    _, pitch, _ = measured
    assert pitch == pytest.approx(angle, abs=3.5)


def test_iris_estimate_is_distance_invariant():
    """Moving towards or away from the camera must not change the gaze reading.

    This is the core advantage over reading a raw landmark pixel position, which
    changes scale with distance by construction.
    """
    near = estimate_iris_gaze(make_face(eye_yaw=15.0, distance_mm=400).points)
    far = estimate_iris_gaze(make_face(eye_yaw=15.0, distance_mm=900).points)

    assert near[0] == pytest.approx(far[0], abs=1.0)
    # The vertical axis carries a small perspective bias -- the iris sits in
    # front of the eye-corner plane, so it projects slightly further from the
    # image centre -- which makes it drift a couple of degrees with distance.
    # It is a smooth function of head position, so calibration absorbs it.
    assert near[1] == pytest.approx(far[1], abs=2.5)
    # ...while the raw inter-ocular distance obviously does change.
    assert near[2] > far[2] * 1.5


def test_distance_invariance_beats_raw_pixel_tracking():
    """Quantifies what replacing pixel tracking with angles actually buys."""
    near = make_face(eye_yaw=15.0, distance_mm=400)
    far = make_face(eye_yaw=15.0, distance_mm=900)

    # The iris centre is the signal the old implementation effectively tracked.
    pixel_shift = abs(float(near.points[468][0]) - float(far.points[468][0]))
    assert pixel_shift > 20.0, "expected the pixel signal to move with distance"

    # The same gaze, expressed as an angle, does not move at all.
    assert abs(estimate_iris_gaze(near.points)[0] - estimate_iris_gaze(far.points)[0]) < 1.0


# --------------------------------------------------------------------------
# Combined estimate
# --------------------------------------------------------------------------
def test_iris_reports_total_gaze_and_is_not_double_counted(estimator):
    """Head rotation and eye rotation must compose to the total gaze, once.

    The iris sits on the eyeball where the gaze exits, so its offset already
    measures the gaze direction in the camera frame whatever the head does.
    Adding the head angle on top -- which the naive formulation does -- would
    double the cursor's response to a head turn.
    """
    head_only = estimator.estimate(make_face(head_yaw=20.0, eye_yaw=0.0).points)
    eyes_only = estimator.estimate(make_face(head_yaw=0.0, eye_yaw=15.0).points)
    combined = estimator.estimate(make_face(head_yaw=20.0, eye_yaw=15.0).points)

    assert head_only.valid and eyes_only.valid and combined.valid
    assert head_only.yaw == pytest.approx(20.0, abs=2.5)
    assert eyes_only.yaw == pytest.approx(15.0, abs=3.5)
    assert combined.yaw == pytest.approx(35.0, abs=4.0)
    assert combined.source == "iris"


def test_head_pose_is_used_when_the_iris_is_unavailable(estimator):
    """Without iris refinement the head pose still has to drive the cursor."""
    face = make_face(head_yaw=18.0, head_pitch=-6.0)
    sample = estimator.estimate(face.points[:468])
    assert sample.valid, sample.reason
    assert sample.source == "head"
    assert sample.yaw == pytest.approx(18.0, abs=2.5)
    assert sample.pitch == pytest.approx(-6.0, abs=2.5)


def test_use_eyes_off_falls_back_to_head_pose():
    estimator = GazeEstimator(WIDTH, HEIGHT, camera_fov_deg=FOV, use_eyes=False)
    sample = estimator.estimate(make_face(head_yaw=12.0, eye_yaw=-25.0).points)
    assert sample.valid and sample.source == "head"
    assert sample.yaw == pytest.approx(12.0, abs=2.5)


def test_closed_eyes_are_rejected(estimator):
    sample = estimator.estimate(make_face(eye_yaw=10.0, eye_open=0.05).points)
    assert not sample.valid
    assert "eyes" in sample.reason


def test_open_eyes_pass_the_openness_gate(estimator):
    sample = estimator.estimate(make_face(eye_open=1.0).points)
    assert sample.valid, sample.reason
    assert min(sample.left_eye_open, sample.right_eye_open) > 0.2


def test_ear_drops_as_the_eye_closes():
    open_face = make_face(eye_open=1.0)
    closed_face = make_face(eye_open=0.1)
    left = (33, 160, 158, 133, 153, 144)
    assert eye_aspect_ratio(open_face.points, left) > 0.2
    assert eye_aspect_ratio(closed_face.points, left) < 0.08


def test_missing_face_gives_an_invalid_sample(estimator):
    sample = estimator.estimate(np.zeros((478, 2)))
    assert not sample.valid
    assert sample.reason


def test_short_landmark_set_does_not_raise(estimator):
    """A result without iris refinement must degrade, not crash."""
    face = make_face(head_yaw=10.0)
    sample = estimator.estimate(face.points[:468])
    assert "iris" in sample.reason
    assert not sample.eyes_visible


def test_landmark_noise_degrades_gracefully(estimator):
    """Landmark jitter must stay bounded, and smoothing must reduce it.

    The iris is only ~10 px across at 720p, so a couple of pixels of landmark
    jitter really is several degrees of single-frame gaze error -- that is the
    resolution floor of webcam gaze tracking, not a bug.  What has to hold is
    that the error is unbiased and that the One Euro filter takes it down to a
    usable level without adding lag to an actual glance.
    """
    truth = 18.0  # head 10 deg + eye 8 deg
    samples = np.array(
        [
            estimator.estimate(
                make_face(head_yaw=10.0, eye_yaw=8.0, noise_px=2.0, seed=seed).points
            ).yaw
            for seed in range(40)
        ]
    )

    assert abs(float(samples.mean()) - truth) < 2.0, "gaze estimate is biased"
    assert float(samples.std()) < 6.0, "single-frame gaze is too unstable"

    filtered = OneEuroFilter(0.8, 0.05)
    smoothed = np.array(
        [filtered.filter(value, index / 30.0) for index, value in enumerate(samples)]
    )[10:]
    assert float(smoothed.std()) < float(samples.std()), "filter did not reduce jitter"


def test_glance_is_not_lagged_into_uselessness(estimator):
    """Heavy smoothing would steady the cursor but make it feel like molasses."""
    frames = [
        estimator.estimate(
            make_face(head_yaw=10.0, eye_yaw=8.0 if i < 30 else 22.0, noise_px=1.0, seed=i).points
        ).yaw
        for i in range(90)
    ]
    filtered = OneEuroFilter(0.8, 0.05)
    smoothed = [filtered.filter(value, i / 30.0) for i, value in enumerate(frames)]

    start = float(np.mean(frames[:25]))
    end = float(np.mean(frames[60:]))
    threshold = start + 0.9 * (end - start)
    reached = next(i for i in range(30, 90) if smoothed[i] >= threshold)

    latency_ms = (reached - 30) / 30.0 * 1000.0
    assert latency_ms < 250.0, f"a 14 deg glance took {latency_ms:.0f} ms to land"


def test_angle_sign_convention_matches_screen_coordinates(estimator):
    """Positive yaw/pitch must mean right/down, matching screen pixel axes."""
    right = estimator.estimate(make_face(eye_yaw=20.0).points)
    left = estimator.estimate(make_face(eye_yaw=-20.0).points)
    down = estimator.estimate(make_face(eye_pitch=15.0).points)
    up = estimator.estimate(make_face(eye_pitch=-15.0).points)

    assert right.yaw > 0 > left.yaw
    assert down.pitch > 0 > up.pitch


def test_distance_grows_as_the_face_moves_away(estimator):
    near = estimator.estimate(make_face(distance_mm=400).points)
    far = estimator.estimate(make_face(distance_mm=900).points)
    assert far.distance > near.distance * 1.8
    assert math.isfinite(near.distance)
