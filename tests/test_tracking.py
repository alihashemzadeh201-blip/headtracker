"""Face landmark tracking wrapper.

MediaPipe's model file cannot be downloaded in a sandbox without network
access, so the landmarker itself is stubbed here.  Everything around it --
frame plumbing, timestamp handling, estimator wiring and model discovery -- is
exercised for real.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from headtracker import tracking
from headtracker.geometry import REFINED_LANDMARK_COUNT
from headtracker.tracking import (
    FaceGazeTracker,
    _bgr_to_rgb,
    ensure_model,
    has_iris_landmarks,
)
from tests.synthetic_face import make_face

WIDTH, HEIGHT = 1280, 720


def as_landmarks(points: np.ndarray, width: int, height: int):
    """Wrap pixel coordinates in the normalised containers MediaPipe returns."""
    return [
        SimpleNamespace(x=float(x) / width, y=float(y) / height, z=0.0)
        for x, y in points
    ]


class StubBackend:
    """Returns a pre-baked landmark set for every frame."""

    def __init__(self, points):
        self.points = points
        self.calls = 0
        self.closed = False
        self.last_timestamp = None

    def track(self, bgr_frame, timestamp_ms):
        self.calls += 1
        self.last_timestamp = timestamp_ms
        height, width = bgr_frame.shape[:2]
        return as_landmarks(self.points, width, height)

    def close(self):
        self.closed = True


class EmptyBackend(StubBackend):
    """A landmarker that never finds a face."""

    def track(self, bgr_frame, timestamp_ms):
        self.calls += 1
        return []


@pytest.fixture(name="tracker")
def fixture_tracker(monkeypatch) -> FaceGazeTracker:
    points = make_face(head_yaw=8.0, eye_yaw=6.0).points
    stub = StubBackend(points)
    monkeypatch.setattr(FaceGazeTracker, "_create_backend", staticmethod(lambda *a, **k: stub))
    instance = FaceGazeTracker(model_path=None)
    instance.stub = stub  # type: ignore[attr-defined]
    return instance


def test_process_produces_a_valid_sample(tracker):
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    sample = tracker.process(frame, timestamp_ms=0)

    assert sample.valid, sample.reason
    assert sample.source == "iris"
    assert sample.yaw == pytest.approx(14.0, abs=4.0)
    assert sample.distance > 0
    assert sample.extras["landmark_count"] == REFINED_LANDMARK_COUNT


def test_process_reports_a_missing_face(tracker, monkeypatch):
    monkeypatch.setattr(tracker, "_backend", EmptyBackend([]))
    sample = tracker.process(np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8), timestamp_ms=0)
    assert not sample.valid
    assert sample.reason == "no face detected"


def test_timestamps_are_strictly_increasing(tracker):
    """MediaPipe's VIDEO mode rejects a non-monotonic timestamp."""
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    seen = []

    class Recorder(StubBackend):
        """Records the timestamps the tracker hands to MediaPipe."""

        def track(self, bgr_frame, timestamp_ms):
            seen.append(timestamp_ms)
            return super().track(bgr_frame, timestamp_ms)

    tracker._backend = Recorder(tracker.stub.points)  # pylint: disable=protected-access

    for value in (500, 500, 499, 600, 600):
        tracker.process(frame, timestamp_ms=value)

    assert seen == sorted(seen)
    assert len(set(seen)) == len(seen), f"timestamps not unique: {seen}"


def test_timestamps_are_generated_when_not_supplied(tracker):
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    first = tracker.process(frame)
    second = tracker.process(frame)
    assert first.valid and second.valid
    assert tracker._last_timestamp_ms > 0  # pylint: disable=protected-access


def test_estimator_is_rebuilt_when_the_resolution_changes(tracker):
    tracker.process(np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8), timestamp_ms=0)
    assert tracker._estimator.frame_width == WIDTH  # pylint: disable=protected-access

    tracker.process(np.zeros((1080, 1920, 3), dtype=np.uint8), timestamp_ms=1)
    assert tracker._estimator.frame_width == 1920  # pylint: disable=protected-access
    assert tracker._estimator.frame_height == 1080  # pylint: disable=protected-access


def test_context_manager_closes_the_backend(tracker):
    with tracker:
        pass
    assert tracker.stub.closed


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def test_bgr_to_rgb_swaps_channels_and_stays_contiguous():
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    frame[:, :, 0] = 1  # blue
    frame[:, :, 2] = 2  # red
    rgb = _bgr_to_rgb(frame)
    assert rgb[0, 0, 0] == 2 and rgb[0, 0, 2] == 1
    assert rgb.flags["C_CONTIGUOUS"]


@pytest.mark.parametrize(
    "count, expected", [(478, True), (468, False), (0, False), (500, True)]
)
def test_has_iris_landmarks(count, expected):
    assert has_iris_landmarks(count) is expected


def test_ensure_model_finds_an_existing_file(tmp_path):
    model = tmp_path / "face_landmarker.task"
    model.write_bytes(b"weights")
    assert ensure_model(model) == model


def test_ensure_model_explains_how_to_fix_a_failed_download(tmp_path, monkeypatch):
    def explode(*args, **kwargs):
        raise OSError("network unreachable")

    monkeypatch.setattr(tracking, "_download", explode)
    monkeypatch.setattr(tracking, "default_model_dir", lambda: tmp_path)
    monkeypatch.setattr("pathlib.Path.cwd", lambda: tmp_path / "nowhere")

    with pytest.raises(FileNotFoundError) as excinfo:
        ensure_model(tmp_path / "absent" / "face_landmarker.task")

    message = str(excinfo.value)
    assert "mediapipe-models" in message, "should point at the download URL"
    assert "face_landmarker.task" in message


def test_ensure_model_prefers_the_repo_assets_directory(tmp_path, monkeypatch):
    assets = tmp_path / "assets"
    assets.mkdir()
    bundled = assets / tracking.MODEL_FILENAME
    bundled.write_bytes(b"weights")

    monkeypatch.setattr("pathlib.Path.cwd", lambda: tmp_path)
    assert ensure_model() == bundled


def test_legacy_backend_rejects_a_mediapipe_without_solutions(monkeypatch):
    import sys  # pylint: disable=import-outside-toplevel
    import types  # pylint: disable=import-outside-toplevel

    fake = types.ModuleType("mediapipe")
    monkeypatch.setitem(sys.modules, "mediapipe", fake)

    with pytest.raises(RuntimeError, match="Tasks API"):
        tracking._LegacyBackend(0.5, 0.5)  # pylint: disable=protected-access
