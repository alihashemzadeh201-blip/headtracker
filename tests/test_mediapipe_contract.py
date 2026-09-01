"""Contract with the installed MediaPipe.

The whole reason this project needed rewriting is that MediaPipe 1.0 deleted
``mp.solutions``, which broke the import outright.  These tests pin the API
surface :mod:`headtracker.tracking` actually relies on, so the next upstream
change shows up here instead of at runtime on the user's machine.

Skipped when MediaPipe is not installed or its native libraries are missing --
the rest of the suite never needs MediaPipe, because the landmarker is stubbed
in ``test_tracking.py``.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect

import pytest

mediapipe = importlib.util.find_spec("mediapipe") is not None
requires_mediapipe = pytest.mark.skipif(not mediapipe, reason="mediapipe not installed")


def _load(*names):
    from importlib import import_module  # pylint: disable=import-outside-toplevel

    return [import_module(name) for name in names]


@requires_mediapipe
def test_the_tasks_api_is_available():
    """The replacement for the removed ``mp.solutions.face_mesh``."""
    (vision,) = _load("mediapipe.tasks.python.vision")
    assert hasattr(vision, "FaceLandmarker")
    assert hasattr(vision, "FaceLandmarkerOptions")


@requires_mediapipe
def test_options_accept_every_argument_we_pass():
    """A renamed or removed field here is exactly the breakage that hit 1.0."""
    from mediapipe.tasks.python.core.base_options import (  # pylint: disable=import-outside-toplevel
        BaseOptions,
    )
    from mediapipe.tasks.python.vision import (  # pylint: disable=import-outside-toplevel
        FaceLandmarkerOptions,
    )
    from mediapipe.tasks.python.vision.core.vision_task_running_mode import (  # pylint: disable=import-outside-toplevel
        VisionTaskRunningMode,
    )

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="/nonexistent.task"),
        running_mode=VisionTaskRunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.6,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    assert options.running_mode == VisionTaskRunningMode.VIDEO
    assert options.num_faces == 1
    assert options.min_tracking_confidence == pytest.approx(0.6)


@requires_mediapipe
def test_video_mode_is_supported():
    from mediapipe.tasks.python.vision.core.vision_task_running_mode import (  # pylint: disable=import-outside-toplevel
        VisionTaskRunningMode,
    )

    assert hasattr(VisionTaskRunningMode, "VIDEO")


@requires_mediapipe
def test_detect_for_video_takes_a_timestamp():
    from mediapipe.tasks.python.vision import (  # pylint: disable=import-outside-toplevel
        FaceLandmarker,
    )

    parameters = list(inspect.signature(FaceLandmarker.detect_for_video).parameters)
    assert parameters[:3] == ["self", "image", "timestamp_ms"]


@requires_mediapipe
def test_result_exposes_landmarks_with_x_and_y():
    """``landmarks_to_array`` reads ``.x`` / ``.y`` off every landmark."""
    from mediapipe.tasks.python.components.containers.landmark import (  # pylint: disable=import-outside-toplevel
        NormalizedLandmark,
    )
    from mediapipe.tasks.python.vision.face_landmarker import (  # pylint: disable=import-outside-toplevel
        FaceLandmarkerResult,
    )

    names = {f.name for f in dataclasses.fields(NormalizedLandmark)}
    assert {"x", "y"} <= names
    assert "face_landmarks" in {f.name for f in dataclasses.fields(FaceLandmarkerResult)}


@requires_mediapipe
def test_image_constructor_matches_our_call():
    from mediapipe.tasks.python.vision.core.image import (  # pylint: disable=import-outside-toplevel
        Image,
    )

    assert list(inspect.signature(Image.__init__).parameters) == [
        "self",
        "image_format",
        "data",
    ]


@requires_mediapipe
def test_legacy_solutions_is_only_used_when_it_exists():
    """The fallback must not be reached on a MediaPipe that dropped it."""
    import mediapipe as mp  # pylint: disable=import-outside-toplevel

    from headtracker.tracking import (  # pylint: disable=import-outside-toplevel
        _LegacyBackend,
    )

    if not hasattr(mp, "solutions"):
        with pytest.raises(RuntimeError, match="Tasks API"):
            _LegacyBackend(0.5, 0.5)
