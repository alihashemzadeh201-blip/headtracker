"""Face landmark tracking and gaze extraction from a video frame.

MediaPipe 1.0 removed the legacy ``mp.solutions`` namespace that this project
originally used; a fresh ``pip install mediapipe`` therefore crashes the old
``head.py`` with ``AttributeError: module 'mediapipe' has no attribute
'solutions'``.  This module targets the replacement Tasks API
(:class:`FaceLandmarker`) and falls back to the legacy solver when an older
MediaPipe is installed.
"""

from __future__ import annotations

import time
import urllib.request
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from .geometry import (
    IRIS_LANDMARK_BASE,
    GazeEstimator,
    GazeSample,
    landmarks_to_array,
)

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_FILENAME = "face_landmarker.task"


def default_model_dir() -> Path:
    """Where the landmarker model is cached between runs."""
    base = Path.home() / ".cache" / "headtracker"
    base.mkdir(parents=True, exist_ok=True)
    return base


def ensure_model(
    path: Optional[Path] = None, progress: Optional[Callable[[int], None]] = None
) -> Path:
    """Return a path to the model file, downloading it on first use.

    Raises :class:`FileNotFoundError` with actionable instructions when the
    model is missing and cannot be fetched -- a bare exception from MediaPipe
    about a missing asset is much harder to act on.
    """
    candidates = []
    if path is not None:
        candidates.append(Path(path))
    candidates += [
        Path.cwd() / "assets" / MODEL_FILENAME,
        Path(__file__).resolve().parent.parent / "assets" / MODEL_FILENAME,
        default_model_dir() / MODEL_FILENAME,
    ]

    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate

    target = candidates[-1]
    try:
        _download(MODEL_URL, target, progress)
    except Exception as exc:  # pylint: disable=broad-except
        raise FileNotFoundError(
            f"The MediaPipe face landmarker model is required but could not be "
            f"downloaded ({exc}).\n"
            f"Download it manually from:\n  {MODEL_URL}\n"
            f"and save it as:\n  {target}"
        ) from exc
    return target


def _download(url: str, target: Path, progress: Optional[Callable[[int], None]]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    received = 0
    with urllib.request.urlopen(url, timeout=60) as response, open(  # noqa: S310
        temporary, "wb"
    ) as handle:
        while True:
            chunk = response.read(1 << 16)
            if not chunk:
                break
            handle.write(chunk)
            received += len(chunk)
            if progress is not None:
                progress(received)
    temporary.replace(target)


class FaceGazeTracker:
    """Runs the face landmarker over BGR frames and yields gaze samples."""

    def __init__(  # pylint: disable=too-many-positional-arguments
        self,
        model_path: Path,
        camera_fov_deg: float = 60.0,
        min_face_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.6,
        min_eye_open: float = 0.18,
        use_eyes: bool = True,
    ) -> None:
        self._estimator: Optional[GazeEstimator] = None
        self._camera_fov_deg = camera_fov_deg
        self._min_eye_open = min_eye_open
        self._use_eyes = use_eyes
        self._last_timestamp_ms = -1
        self.landmark_count = 0
        self._backend = self._create_backend(
            model_path,
            min_face_detection_confidence,
            min_tracking_confidence,
        )

    # -- backend selection --------------------------------------------------
    @staticmethod
    def _create_backend(model_path: Path, detection: float, tracking: float):
        """Build a Tasks-API landmarker, or a legacy one on old MediaPipe."""
        try:
            # pylint: disable-next=import-outside-toplevel
            from mediapipe.tasks.python.core.base_options import BaseOptions
            # pylint: disable-next=import-outside-toplevel
            from mediapipe.tasks.python.vision import (
                FaceLandmarker,
                FaceLandmarkerOptions,
            )
            # pylint: disable-next=import-outside-toplevel
            from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
                VisionTaskRunningMode,
            )
        except ImportError:
            return _LegacyBackend(detection, tracking)

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=VisionTaskRunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=detection,
            min_face_presence_confidence=detection,
            min_tracking_confidence=tracking,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        return _TasksBackend(FaceLandmarker.create_from_options(options))

    def set_use_eyes(self, use_eyes: bool) -> None:
        """Enable or disable the iris contribution, discarding cached state."""
        self._use_eyes = bool(use_eyes)
        if self._estimator is not None:
            self._estimator.use_eyes = self._use_eyes

    def close(self) -> None:
        self._backend.close()

    def __enter__(self) -> "FaceGazeTracker":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # -- per-frame work -----------------------------------------------------
    def process(self, bgr_frame: np.ndarray, timestamp_ms: Optional[int] = None) -> GazeSample:
        """Track one BGR frame and return its gaze sample."""
        height, width = bgr_frame.shape[:2]
        if self._estimator is None or (
            self._estimator.frame_width != width or self._estimator.frame_height != height
        ):
            self._estimator = GazeEstimator(
                width,
                height,
                camera_fov_deg=self._camera_fov_deg,
                min_eye_open=self._min_eye_open,
                use_eyes=self._use_eyes,
            )

        landmarks = self._backend.track(bgr_frame, self._next_timestamp(timestamp_ms))
        if not landmarks:
            return GazeSample(reason="no face detected")

        self.landmark_count = len(landmarks)
        points = landmarks_to_array(landmarks, width, height)
        sample = self._estimator.estimate(points)
        sample.extras["landmark_count"] = self.landmark_count
        return sample

    def _next_timestamp(self, timestamp_ms: Optional[int]) -> int:
        """VIDEO mode requires strictly increasing timestamps."""
        if timestamp_ms is None:
            timestamp_ms = int(time.monotonic() * 1000)
        timestamp_ms = max(int(timestamp_ms), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms
        return timestamp_ms


class _TasksBackend:
    """MediaPipe >= 1.0 Tasks API."""

    def __init__(self, landmarker) -> None:
        # pylint: disable-next=import-outside-toplevel
        import mediapipe as mp

        self._mp = mp
        self._landmarker = landmarker

    def track(self, bgr_frame: np.ndarray, timestamp_ms: int) -> Sequence:
        rgb = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=_bgr_to_rgb(bgr_frame),
        )
        result = self._landmarker.detect_for_video(rgb, timestamp_ms)
        if not result.face_landmarks:
            return []
        return result.face_landmarks[0]

    def close(self) -> None:
        self._landmarker.close()


class _LegacyBackend:
    """MediaPipe < 1.0 ``solutions.face_mesh``."""

    def __init__(self, detection: float, tracking: float) -> None:
        # pylint: disable-next=import-outside-toplevel
        import mediapipe as mp

        if not hasattr(mp, "solutions"):
            raise RuntimeError(
                "This MediaPipe build exposes neither the Tasks API nor the "
                "legacy solutions API. Install mediapipe>=0.10 with "
                "'pip install -U mediapipe'."
            )
        self._mp = mp
        self._mesh = mp.solutions.face_mesh.FaceMesh(  # pylint: disable=no-member
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=detection,
            min_tracking_confidence=tracking,
        )

    def track(self, bgr_frame: np.ndarray, timestamp_ms: int) -> Sequence:
        del timestamp_ms  # the legacy graph is stateful and takes no timestamp
        result = self._mesh.process(_bgr_to_rgb(bgr_frame))
        if not result.multi_face_landmarks:
            return []
        return result.multi_face_landmarks[0].landmark

    def close(self) -> None:
        self._mesh.close()


def _bgr_to_rgb(bgr_frame: np.ndarray) -> np.ndarray:
    """Swap BGR -> RGB without pulling in OpenCV for a single call."""
    return np.ascontiguousarray(bgr_frame[:, :, ::-1])


def has_iris_landmarks(count: int) -> bool:
    """True when the landmark set includes the iris refinement."""
    return count >= IRIS_LANDMARK_BASE + 1
