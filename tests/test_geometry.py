"""Numerical verification of the gaze geometry against projected synthetic faces."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from headtracker.filters import OneEuroFilter
from headtracker.geometry import (
    MODEL_POINTS,
    PNP_IMAGE_POINTS,
    GazeEstimator,
    default_camera_matrix,
    estimate_iris_gaze,
    eye_aspect_ratio,
    gaze_direction,
    project_to_screen_plane,
    reprojection_error,
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


# --------------------------------------------------------------------------
# Pose solver robustness
# --------------------------------------------------------------------------
def test_coincident_landmarks_are_rejected_rather_than_solved():
    """The guard that keeps a lost face from producing a cursor position.

    MediaPipe answers with coincident points when it loses the face, and PnP
    will happily fit those: it reports the face at a depth of 1e16 with a
    reprojection error of *exactly zero*, because every model point projects
    onto the pixel the landmarks already share.  Only the physical size of the
    face distinguishes that from a real pose.
    """
    camera = default_camera_matrix(WIDTH, HEIGHT, FOV)
    assert solve_head_rotation(np.zeros((478, 2)), camera) is None


def test_a_nearly_coincident_face_is_rejected_too():
    camera = default_camera_matrix(WIDTH, HEIGHT, FOV)
    points = np.zeros((478, 2))
    points[list(PNP_IMAGE_POINTS)] = np.array([[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5]])
    assert solve_head_rotation(points, camera) is None


def test_a_real_face_survives_heavy_landmark_noise():
    """The rejection must not be so eager that it drops usable frames.

    This is the failure the solver change was made to fix: ``ITERATIVE`` alone
    returned a mirrored, negative-depth solution on 17 of 60 jittered frames,
    and every one of those became a dropped frame and a stuttering cursor.
    """
    camera = default_camera_matrix(WIDTH, HEIGHT, FOV)
    depths = []
    for index in range(60):
        points = make_face(head_yaw=10.0, eye_yaw=8.0, noise_px=3.0, seed=index,
                           width=WIDTH, height=HEIGHT).points
        solved = solve_head_rotation(points, camera)
        assert solved is not None, f"frame {index} was rejected"
        depths.append(solved[1][2])

    depths = np.array(depths)
    assert np.all(depths > 0)
    # A stable depth means the solver is not flipping between branches.
    assert depths.std() < 200.0, f"depth std {depths.std():.0f}"


def test_reprojection_error_separates_a_real_pose_from_a_fabricated_one():
    """The error has to distinguish a fit from a fabrication, by a wide margin.

    It is used to choose between solvers, and as a last-resort bound.  The
    bound is generous on purpose -- 60 px -- because a real face does not match
    the canonical model and reprojects imperfectly even when the pose is right.
    A 15 px mismatch in the pose landmarks alone measures 13.7 px, so a tight
    bound tuned on synthetic faces would reject real ones.
    """
    camera = default_camera_matrix(WIDTH, HEIGHT, FOV)
    points = make_face(head_yaw=10.0, eye_yaw=8.0, width=WIDTH, height=HEIGHT).points
    face_points = np.array([points[i] for i in PNP_IMAGE_POINTS], dtype=np.float64)
    distortion = np.zeros((4, 1))

    ok, rvec, tvec = cv2.solvePnP(
        MODEL_POINTS, face_points, camera, distortion, flags=cv2.SOLVEPNP_SQPNP
    )
    assert ok
    good = reprojection_error(rvec, tvec, face_points, camera, distortion)

    wrong, _ = cv2.Rodrigues(np.array([1.7, -1.2, 0.9]))
    bad = reprojection_error(wrong, tvec, face_points, camera, distortion)

    assert good < 1.0, f"a clean synthetic face should fit to 0.0 px, got {good:.1f}"
    assert bad > good * 20, f"fabricated pose ({bad:.1f}) not separated from real ({good:.1f})"


def test_a_face_that_does_not_match_the_model_is_still_accepted():
    """The guard must not depend on the face matching the canonical model.

    MediaPipe's mesh is one particular skull.  Perturbing the six pose
    landmarks by 15 px -- far less than the difference between two real
    people -- pushes the reprojection error past 13 px.  The earlier bound
    rejected those frames outright, which on a real camera meant rejecting all
    of them and freezing the cursor.
    """
    camera = default_camera_matrix(WIDTH, HEIGHT, FOV)
    points = make_face(head_yaw=10.0, eye_yaw=8.0, width=WIDTH, height=HEIGHT).points
    rng = np.random.default_rng(0)
    points[list(PNP_IMAGE_POINTS)] += rng.normal(0.0, 15.0, size=(6, 2))

    solved = solve_head_rotation(points, camera)
    assert solved is not None, "a real face with an ordinary skull was rejected"
    assert solved[1][2] > 0


# --------------------------------------------------------------------------
# Screen-plane projection
# --------------------------------------------------------------------------
def test_a_downward_gaze_hits_the_plane_in_front_of_the_eye():
    """Sign convention: pitch > 0 looks down, so it must land below the eye."""
    hit = project_to_screen_plane(np.array([0.0, 0.0, 4000.0]), gaze_direction(0.0, 10.0))
    assert hit is not None
    assert hit[1] > 0.0


def test_a_gaze_parallel_to_the_screen_never_arrives():
    """Looking along the horizon meets the plane at infinity, not at a pixel."""
    assert project_to_screen_plane(np.array([0.0, 0.0, 4000.0]),
                                  gaze_direction(0.0, 90.0)) is None


def test_the_hit_point_moves_with_the_eye_at_a_fixed_angle():
    """The whole point of projecting instead of integrating an angle.

    Sliding the eye sideways while holding the gaze direction still has to move
    the hit point by exactly the same amount: the ray translates with its
    origin.  An angle-only mapping reports "unchanged" here, which is why a
    head that drifts off centre takes the cursor with it in the wrong direction.
    """
    direction = gaze_direction(5.0, -3.0)
    near = project_to_screen_plane(np.array([0.0, 0.0, 4000.0]), direction)
    shifted = project_to_screen_plane(np.array([120.0, 0.0, 4000.0]), direction)
    assert near is not None and shifted is not None
    assert shifted[0] - near[0] == pytest.approx(120.0, abs=1e-9)
    assert shifted[1] == pytest.approx(near[1], abs=1e-9)


def test_the_lever_arm_grows_with_distance():
    """Leaning back makes the same angle sweep more screen, automatically.

    This is what the removed ``compensate_distance`` factor approximated with a
    fitted correction; the projection gets it from the geometry instead.
    """
    near = project_to_screen_plane(np.array([0.0, 0.0, 3000.0]), gaze_direction(10.0, 0.0))
    far = project_to_screen_plane(np.array([0.0, 0.0, 4500.0]), gaze_direction(10.0, 0.0))
    assert near is not None and far is not None
    assert far[0] / near[0] == pytest.approx(1.5, rel=1e-6)


def test_moving_the_head_across_the_frame_moves_the_screen_point(estimator):
    """End to end: translate the landmarks, hold the gaze, watch the ray follow.

    The shift is applied to every landmark at once, which is exactly what a
    head sliding sideways in the camera's view looks like.
    """
    base = make_face(head_yaw=5.0, eye_yaw=10.0, width=WIDTH, height=HEIGHT).points
    still = estimator.estimate(base)
    moved = estimator.estimate(base + np.array([-30.0, 0.0]))

    assert still.valid and moved.valid
    # The gaze angle is unchanged -- the eyes are doing the same thing.
    assert moved.yaw == pytest.approx(still.yaw, abs=0.5)
    # And the ray now starts 30 px further left, so it lands further left too.
    # Measured at this resolution: -120.8 screen units for a -30 px shift.
    assert moved.screen_x < still.screen_x - 20.0
    assert moved.screen_y == pytest.approx(still.screen_y, abs=30.0)


def test_the_wider_eye_frame_is_quieter_than_the_two_corners():
    """Guards the noise measurement behind ``LEFT_EYE_FRAME``.

    The eye centre is one subtraction of two jittering points, so it carries
    most of the measurement noise.  Averaging the eight landmarks on the rim
    instead of the two corners measured 0.882 deg against 1.163 deg; if that
    ever regresses, the cursor jitter regresses with it.
    """
    estimator = GazeEstimator(1920, 1080, camera_fov_deg=60.0)
    yaws = []
    for index in range(120):
        sample = estimator.estimate(
            make_face(head_yaw=5.0, eye_yaw=12.0, noise_px=1.0, seed=index,
                      width=1920, height=1080).points
        )
        yaws.append(sample.yaw)
    assert float(np.std(yaws)) < 1.0, f"yaw std {np.std(yaws):.3f} deg"
