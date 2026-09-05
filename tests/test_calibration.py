"""Calibration accuracy, robustness and the collection state machine."""

from __future__ import annotations

import json

import numpy as np
import pytest

from headtracker.calibration import (
    FORMAT_VERSION,
    MIN_SAMPLES_PER_POINT,
    CalibrationModel,
    CalibrationSession,
    coefficient_count,
    design_matrix,
    grid_points,
)

SCREEN = (1920.0, 1080.0)


def true_mapping(yaw: np.ndarray, pitch: np.ndarray) -> np.ndarray:
    """A realistic angle -> pixel mapping: pinhole projection on a tilted camera.

    Deliberately nonlinear and off-centre, so a fit that only gets the linear
    part right will not pass.
    """
    mount_yaw, mount_pitch = 3.0, -1.5
    gain_x = SCREEN[0] / 2.0 / np.tan(np.radians(26.0))
    gain_y = SCREEN[1] / 2.0 / np.tan(np.radians(18.0))
    x = SCREEN[0] * 0.47 + gain_x * np.tan(np.radians(yaw - mount_yaw))
    y = SCREEN[1] * 0.52 + gain_y * np.tan(np.radians(pitch - mount_pitch))
    return np.column_stack([x, y])


def angles_for_points(points, jitter: float = 0.0, seed: int = 0):
    """Invert ``true_mapping`` numerically to get the gaze angle for each point."""
    mount_yaw, mount_pitch = 3.0, -1.5
    gain_x = SCREEN[0] / 2.0 / np.tan(np.radians(26.0))
    gain_y = SCREEN[1] / 2.0 / np.tan(np.radians(18.0))
    angles = []
    for normalised in points:
        px, py = normalised[0] * SCREEN[0], normalised[1] * SCREEN[1]
        yaw = np.degrees(np.arctan((px - SCREEN[0] * 0.47) / gain_x)) + mount_yaw
        pitch = np.degrees(np.arctan((py - SCREEN[1] * 0.52) / gain_y)) + mount_pitch
        angles.append((yaw, pitch))
    angles = np.array(angles)
    if jitter:
        angles = angles + np.random.default_rng(seed).normal(0.0, jitter, angles.shape)
    return angles


# --------------------------------------------------------------------------
# Fit quality
# --------------------------------------------------------------------------
def test_fit_recovers_an_exact_quadratic():
    """Guards the least-squares solver itself: an exact fit must be exact."""
    angles = np.array(grid_points(4, 4))

    def quadratic(features):
        x, y = features[:, 0], features[:, 1]
        return np.column_stack(
            [
                900 + 40 * x + 3 * y + 0.8 * x * x - 0.2 * x * y + 0.5 * y * y,
                500 - 25 * x + 2 * y + 0.1 * x * y + 0.3 * y * y,
            ]
        )

    model = CalibrationModel(degree=2)
    model.fit(angles, quadratic(angles), screen=SCREEN)
    assert float(np.max(np.abs(model.predict(angles) - quadratic(angles)))) < 0.01


def test_degree_two_captures_curvature_that_degree_one_cannot():
    # grid_points returns normalised screen positions; the fit works on gaze
    # angles in degrees, so scale them to a realistic range first.
    angles = (np.array(grid_points(4, 4)) - 0.5) * np.array([44.0, 28.0])
    curved = np.column_stack(
        [900 + 40 * angles[:, 0] + 0.9 * angles[:, 0] ** 2, 500 + 30 * angles[:, 1]]
    )

    linear = CalibrationModel(degree=1).fit(angles, curved, screen=SCREEN)
    quadratic = CalibrationModel(degree=2).fit(angles, curved, screen=SCREEN)

    assert linear.rms_error > 20.0, "degree 1 should visibly miss the curvature"
    assert quadratic.rms_error < 0.05


def test_pinhole_mapping_is_fitted_well_below_the_gaze_noise_floor():
    """The calibration must not be the limiting factor in end-to-end accuracy.

    A real angle -> pixel mapping is a pinhole projection, i.e. a tangent, which
    a degree-2 surface cannot represent exactly -- the cubic term is left over.
    Over a realistic +-25 deg range that leaves single-digit pixel error, which
    is far smaller than the ~1 deg (~40 px) gaze noise a webcam produces, so
    calibration is not what limits accuracy.
    """
    points = grid_points(4, 4)
    angles = angles_for_points(points)
    targets = true_mapping(angles[:, 0], angles[:, 1])

    model = CalibrationModel(degree=2)
    report = model.fit(angles, targets, screen=SCREEN)

    assert report.usable
    assert report.rms_error < 15.0, f"fit is not accurate enough: {report.describe()}"
    assert report.rms_error / SCREEN[0] < 0.01


def test_degree_two_beats_degree_one_on_a_curved_mapping():
    points = grid_points(4, 4)
    angles = angles_for_points(points)
    targets = true_mapping(angles[:, 0], angles[:, 1])

    linear = CalibrationModel(degree=1).fit(angles, targets, screen=SCREEN)
    quadratic = CalibrationModel(degree=2).fit(angles, targets, screen=SCREEN)

    assert quadratic.rms_error < linear.rms_error
    assert quadratic.max_error < linear.max_error


def test_fit_extrapolates_sensibly_between_calibration_points():
    """Accuracy must hold *between* the dots, not just on them."""
    points = grid_points(4, 4)
    angles = angles_for_points(points)
    targets = true_mapping(angles[:, 0], angles[:, 1])
    model = CalibrationModel(degree=2)
    model.fit(angles, targets, screen=SCREEN)

    held_out_yaw = np.linspace(-14.0, 14.0, 9)
    held_out_pitch = np.linspace(-8.0, 8.0, 7)
    yaw_grid, pitch_grid = np.meshgrid(held_out_yaw, held_out_pitch)
    probes = np.column_stack([yaw_grid.ravel(), pitch_grid.ravel()])

    predicted = model.predict(probes)
    errors = np.linalg.norm(predicted - true_mapping(probes[:, 0], probes[:, 1]), axis=1)
    # Measured: ~9 px mean and ~14 px worst case for a 16-point calibration of
    # a tangent mapping.  The surface is exact on the dots and drifts between
    # them; more points shrink it only slowly, which is fine because the gaze
    # signal itself is noisier than this.
    assert float(np.mean(errors)) < 20.0
    assert float(np.max(errors)) < 40.0


def test_outliers_are_rejected():
    """One bad glance during calibration must not warp the whole surface."""
    points = grid_points(4, 4)
    angles = angles_for_points(points)
    targets = true_mapping(angles[:, 0], angles[:, 1])

    corrupted_targets = targets.copy()
    corrupted_targets[5] += np.array([400.0, -350.0])  # the user blinked here

    model = CalibrationModel(degree=2)
    report = model.fit(angles, corrupted_targets, screen=SCREEN)

    assert report.points_rejected >= 1
    assert report.rms_error < 15.0, f"outlier survived: {report.describe()}"


def test_jittery_calibration_samples_still_fit_well():
    """Each point is captured over ~1 s of noisy gaze, so the fit must cope."""
    points = grid_points(4, 4)
    angles = angles_for_points(points, jitter=1.2, seed=7)
    targets = true_mapping(angles[:, 0], angles[:, 1])

    model = CalibrationModel(degree=2)
    report = model.fit(angles, targets, screen=SCREEN)
    assert report.usable
    assert report.rms_error < 30.0


def test_too_few_points_is_an_error_not_a_silent_bad_fit():
    model = CalibrationModel(degree=2)
    angles = angles_for_points(grid_points(2, 2))
    with pytest.raises(ValueError, match="at least"):
        model.fit(angles, true_mapping(angles[:, 0], angles[:, 1]), screen=SCREEN)


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        CalibrationModel().predict_one(0.0, 0.0)


def test_default_model_is_roughly_centred():
    model = CalibrationModel.default(SCREEN)
    x, y = model.predict_one(0.0, 0.0)
    assert x == pytest.approx(SCREEN[0] / 2.0, abs=1.0)
    assert y == pytest.approx(SCREEN[1] / 2.0, abs=1.0)
    right, _ = model.predict_one(10.0, 0.0)
    _, down = model.predict_one(0.0, 10.0)
    assert right > x and down > y


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------
def test_round_trip_through_json(tmp_path):
    points = grid_points(3, 3)
    angles = angles_for_points(points)
    targets = true_mapping(angles[:, 0], angles[:, 1])
    original = CalibrationModel(degree=2)
    original.reference = (1.0, 2.0)
    original.reference_distance = 4300.0
    original.fit(angles, targets, screen=SCREEN, reference_distance=4300.0)

    path = tmp_path / "cal.json"
    original.save(path)
    assert json.loads(path.read_text(encoding="utf-8"))["degree"] == 2

    restored = CalibrationModel.load(path)
    assert restored is not None
    probe = np.array([[5.0, -3.0], [-9.0, 6.0]])
    np.testing.assert_allclose(restored.predict(probe), original.predict(probe), rtol=1e-9)
    assert restored.reference_distance == pytest.approx(4300.0)


def test_loading_a_corrupt_file_returns_none(tmp_path):
    path = tmp_path / "cal.json"
    path.write_text("{not json", encoding="utf-8")
    assert CalibrationModel.load(path) is None
    assert CalibrationModel.load(tmp_path / "missing.json") is None


def _fitted_model():
    points = grid_points(3, 3)
    angles = angles_for_points(points)
    model = CalibrationModel(degree=2)
    model.fit(angles, true_mapping(angles[:, 0], angles[:, 1]), screen=SCREEN)
    return model


def test_a_saved_file_carries_the_format_version(tmp_path):
    path = tmp_path / "cal.json"
    _fitted_model().save(path)
    assert json.loads(path.read_text(encoding="utf-8"))["version"] == FORMAT_VERSION


@pytest.mark.parametrize("version", [1, 2, FORMAT_VERSION + 1, 99])
def test_a_calibration_from_another_format_is_refused(tmp_path, version):
    """Coefficients mean nothing without knowing what the features were.

    Reading a file whose features meant something else does not raise -- it
    returns a confident, wrong pixel, which the user experiences as "the
    accuracy is low" with nothing on screen to explain it.  Refusing is the only
    safe answer.  Version 2 is in the list deliberately: see the next test.
    """
    model = _fitted_model()
    path = tmp_path / "cal.json"
    model.save(path)

    data = json.loads(path.read_text(encoding="utf-8"))
    data["version"] = version
    path.write_text(json.dumps(data), encoding="utf-8")

    assert CalibrationModel.load(path) is None, "a stale file must not be used"
    with pytest.raises(ValueError, match="recalibrate"):
        CalibrationModel.from_dict(data)


def test_a_screen_plane_era_file_is_refused(tmp_path):
    """The case that made the version number have to move off 2.

    Three different feature spaces in this project's history all wrote
    ``"version": 2``: ``(yaw, pitch)`` angles in ``c3d1a3a``, screen-plane
    coordinates in ``733ded0``, and head-translation features in ``3135c66``.
    So a version-2 file on disk is ambiguous, and leaving ``FORMAT_VERSION`` at
    2 would have kept accepting all three.

    This is the exact key set ``733ded0`` wrote -- no ``reference_distance`` --
    with a correctly shaped degree-2 coefficient block.  Under a build whose
    ``FORMAT_VERSION`` was still 2 this loaded, reported itself fitted, and
    predicted ``(0.0, 0.0)``.
    """
    stale = {
        "version": 2,
        "degree": 2,
        "mean": [960.0, 540.0],
        "scale": [940.0, 520.0],
        "coefficients": np.zeros((6, 2)).tolist(),
        "reference": None,
        "screen": [1920.0, 1080.0],
        "rms_error": 14.0,
    }
    path = tmp_path / "cal.json"
    path.write_text(json.dumps(stale), encoding="utf-8")

    assert CalibrationModel.load(path) is None, "an ambiguous old file must not be used"
    with pytest.raises(ValueError, match="recalibrate"):
        CalibrationModel.from_dict(stale)


def test_a_file_from_before_versioning_is_refused(tmp_path):
    """No ``version`` key means an older build wrote it -- same answer."""
    model = _fitted_model()
    path = tmp_path / "cal.json"
    model.save(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    del data["version"]
    path.write_text(json.dumps(data), encoding="utf-8")

    assert CalibrationModel.load(path) is None


def test_refusing_a_stale_file_leaves_the_app_asking_for_a_recalibration(tmp_path):
    """The refusal must degrade to the normal "not calibrated" path."""
    path = tmp_path / "cal.json"
    _fitted_model().save(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["version"] = 1
    path.write_text(json.dumps(data), encoding="utf-8")

    model = CalibrationModel.load(path)
    assert model is None or not model.is_fitted


# --------------------------------------------------------------------------
# Session state machine
# --------------------------------------------------------------------------
def test_session_walks_every_point_and_fits():
    points = grid_points(3, 3)
    session = CalibrationSession(points, screen=SCREEN, countdown_s=0.2)
    angles = angles_for_points(points)

    clock = 0.0
    session.start(clock)
    guard = 0
    while not session.finished and guard < 5000:
        guard += 1
        clock += 1.0 / 60.0
        current = session.index
        if session.is_collecting():
            # Symmetric jitter around the true gaze for the point being shown.
            yaw, pitch = angles[current]
            wobble = 0.6 if (guard % 2) else -0.6
            session.add_sample(yaw + wobble, pitch - wobble, distance=4200.0)
        session.update(clock)
        if session.is_ready():
            session.advance(clock)  # the user, not the timer, moves it on

    assert session.finished
    assert len(session.features) == len(points)

    model = session.build(degree=2)
    assert model.is_fitted
    assert model.reference_distance == pytest.approx(4200.0)

    # The jitter is symmetric, so its median is the true gaze and the fitted
    # surface should land back on the calibration target.
    probe = angles[0]
    predicted = model.predict_one(probe[0], probe[1])
    expected = points[0][0] * SCREEN[0], points[0][1] * SCREEN[1]
    assert predicted[0] == pytest.approx(expected[0], abs=15.0)
    assert predicted[1] == pytest.approx(expected[1], abs=15.0)


def test_session_uses_the_median_so_a_blink_cannot_move_a_point():
    session = CalibrationSession(grid_points(3, 3), screen=SCREEN, countdown_s=0.0)
    session.start(0.0)
    session.update(0.0)  # leave the countdown

    for _ in range(20):
        session.add_sample(5.0, 2.0, distance=4000.0)
    session.add_sample(90.0, 90.0, distance=4000.0)  # a single wild sample

    session.advance(1.5)  # commits the point
    assert session.features[0] == (5.0, 2.0)
    assert session.reference_distance == pytest.approx(4000.0)


def test_update_never_moves_to_the_next_point_on_its_own():
    """The whole point of the manual session: no timer may advance it.

    A calibration that walks itself forward yanks the dot away from under a
    user who is still settling on it, and the samples captured in that last
    fraction of a second are the worst ones in the whole fit.  Ten minutes of
    ticking with a full sample buffer must leave the point exactly where it was.
    """
    session = CalibrationSession(grid_points(3, 3), screen=SCREEN, countdown_s=0.0)
    session.start(0.0)
    session.update(0.0)  # leave the countdown first, or samples are discarded

    clock = 0.0
    for _ in range(36000):  # ten minutes at 60 Hz, buffer kept full the whole way
        clock += 1.0 / 60.0
        session.add_sample(1.0, 1.0, distance=4000.0)
        session.update(clock)

    assert session.is_ready(), "the buffer was full; a click would have worked"
    assert session.index == 0, "update() must never commit a point"
    assert not session.features
    assert not session.finished


def test_advance_is_the_only_thing_that_moves_the_session_on():
    session = CalibrationSession(grid_points(3, 3), screen=SCREEN, countdown_s=0.0)
    session.start(0.0)
    session.update(0.0)  # leave the countdown, so samples are actually kept
    for _ in range(MIN_SAMPLES_PER_POINT):
        session.add_sample(1.0, 1.0, distance=4000.0)

    session.update(600.0)
    assert session.index == 0, "ten minutes of ticking must not move it"

    session.advance(600.0)
    assert session.index == 1, "a click must"


def test_is_ready_reports_when_a_click_will_actually_do_something():
    """Without this the only feedback for clicking too early is nothing."""
    session = CalibrationSession(grid_points(3, 3), screen=SCREEN, countdown_s=0.5)
    session.start(0.0)
    assert not session.is_ready(), "still counting down"

    session.update(0.6)  # countdown over, collecting
    assert not session.is_ready(), "no samples yet"

    for _ in range(MIN_SAMPLES_PER_POINT - 1):
        session.add_sample(1.0, 1.0, distance=4000.0)
    assert not session.is_ready(), "one sample short"

    session.add_sample(1.0, 1.0, distance=4000.0)
    assert session.is_ready()


def test_session_will_not_commit_without_enough_samples():
    session = CalibrationSession(grid_points(3, 3), screen=SCREEN, countdown_s=0.0)
    session.start(0.0)
    session.update(0.0)
    session.add_sample(1.0, 1.0)
    session.update(5.0)
    session.advance(5.0)  # a click, with only one sample behind it
    assert session.index == 0
    assert not session.features


def test_advance_commits_the_point_and_moves_on():
    """A click commits the current point.  Nothing else does.

    The user, not a timer, decides when they have held the dot long enough.
    """
    session = CalibrationSession(grid_points(3, 3), screen=SCREEN, countdown_s=0.0)
    session.start(0.0)
    session.update(0.0)  # leave the countdown
    for _ in range(MIN_SAMPLES_PER_POINT):
        session.add_sample(3.0, 4.0, distance=4000.0)

    assert session.advance(0.1) is False  # moved on, calibration not finished
    assert session.index == 1
    assert session.features[0] == (3.0, 4.0)


def test_advance_starts_collecting_when_pressed_during_the_countdown():
    """Pressing early must not throw the point away."""
    session = CalibrationSession(grid_points(3, 3), screen=SCREEN, countdown_s=5.0)
    session.start(0.0)
    assert not session.is_collecting()

    session.advance(0.1)
    assert session.is_collecting()
    assert session.index == 0


def test_advance_refuses_to_commit_a_thinly_sampled_point():
    """A point averaged from one frame is the outlier the median exists to drop.

    Letting a keypress commit it would trade a second of waiting for a badly
    placed control point, and a degree-2 surface is only six coefficients --
    one bad point out of nine moves the whole fit.
    """
    session = CalibrationSession(grid_points(3, 3), screen=SCREEN, countdown_s=0.0)
    session.start(0.0)
    session.update(0.0)
    session.add_sample(1.0, 1.0)

    session.advance(0.1)
    assert session.index == 0
    assert not session.features


def test_advance_finishes_the_session_on_the_last_point():
    session = CalibrationSession(
        [(0.5, 0.5), (0.2, 0.2), (0.8, 0.8)], screen=SCREEN, countdown_s=0.0
    )
    session.start(0.0)
    for index in range(3):
        session.update(index * 10.0)  # leave each countdown
        for _ in range(MIN_SAMPLES_PER_POINT):
            session.add_sample(float(index), float(index), distance=4000.0)
        done = session.advance(index * 10.0 + 0.1)
        assert done is (index == 2)
    assert session.finished
    assert len(session.features) == 3


def test_advance_is_inert_once_finished():
    session = CalibrationSession(
        [(0.5, 0.5), (0.2, 0.2), (0.8, 0.8)], screen=SCREEN, countdown_s=0.0
    )
    clock = 0.0
    session.start(clock)
    while not session.finished and clock < 50.0:
        clock += 0.2
        for _ in range(MIN_SAMPLES_PER_POINT):
            session.add_sample(1.0, 1.0, distance=4000.0)
        session.update(clock)
        session.advance(clock)
    assert session.finished
    before = session.index
    session.advance(clock + 1.0)
    assert session.index == before


def test_session_rejects_a_too_short_point_list():
    with pytest.raises(ValueError, match="at least 3"):
        CalibrationSession([(0.5, 0.5)], screen=SCREEN)


def test_min_samples_constant_is_consistent():
    assert MIN_SAMPLES_PER_POINT >= 3
    assert coefficient_count(1) == 3
    assert coefficient_count(2) == 6


def test_grid_points_stay_inside_the_screen_and_start_centre_out():
    points = grid_points(3, 3)
    assert len(points) == 9
    for x, y in points:
        assert 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0
    first = np.array(points[0])
    assert np.linalg.norm(first - 0.5) < 0.1, "the centre point should come first"


def test_grid_reaches_the_screen_edges():
    """The outermost points must sit near the border, not 8% in from it.

    The margin sets the domain the fitted polynomial is valid over, and a
    degree-2 surface extrapolates badly outside it.  With the margin at 0.08
    the screen edges were extrapolation: 35.5 px of filtered cursor error in
    the outer band against 10.9 px in the centre, and the corners predicted
    pixels off the display entirely.  Measured with the margin at 0.02 and a
    5x4 grid, the outer band falls to 17.2 px and the whole screen from 23.8
    to 13.1 px -- which is the difference between the cursor covering the
    display and it giving up near the edges.
    """
    points = grid_points(5, 4)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    assert min(xs) <= 0.03, "the left edge must be calibrated, not extrapolated"
    assert max(xs) >= 0.97, "the right edge must be calibrated, not extrapolated"
    assert min(ys) <= 0.03
    assert max(ys) >= 0.97


def test_design_matrix_shapes():
    features = np.zeros((5, 2))
    assert design_matrix(features, 1).shape == (5, 3)
    assert design_matrix(features, 2).shape == (5, 6)
    with pytest.raises(ValueError):
        design_matrix(features, 3)
    with pytest.raises(ValueError):
        design_matrix(np.zeros((5, 3)), 1)
