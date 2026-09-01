"""HeadTracker -- gaze-driven mouse control.

Accuracy comes from four pieces working together:

``geometry``
    Recovers head rotation with ``cv2.solvePnP`` and eye rotation from the iris
    landmarks, giving a translation-invariant gaze direction in degrees.
``calibration``
    Fits a polynomial from gaze angles to screen pixels, absorbing the camera
    mounting angle, the monitor distance and individual eye geometry.
``filters``
    One Euro smoothing keeps the cursor steady at rest without lagging a glance.
``controller`` / ``mouse``
    Drives the cursor to an **absolute** position every frame, so unlike the
    relative moves it replaces, no error can accumulate over time.
"""

from __future__ import annotations

from .calibration import CalibrationModel, CalibrationSession, grid_points
from .controller import CursorSettings, GazeCursorController
from .filters import GlitchGate, LowPassFilter, OneEuroFilter
from .geometry import GazeEstimator, GazeSample
from .mouse import AbsoluteMouse, create_backend
from .settings import AppSettings
from .tracking import FaceGazeTracker, ensure_model

__all__ = [
    "AbsoluteMouse",
    "AppSettings",
    "CalibrationModel",
    "CalibrationSession",
    "CursorSettings",
    "FaceGazeTracker",
    "GazeCursorController",
    "GazeEstimator",
    "GazeSample",
    "GlitchGate",
    "LowPassFilter",
    "OneEuroFilter",
    "create_backend",
    "ensure_model",
    "grid_points",
]

__version__ = "2.0.0"
