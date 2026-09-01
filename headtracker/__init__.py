"""HeadTracker -- gaze-driven mouse control.

Accuracy comes from four pieces working together:

``geometry``
    Recovers where the eye is with ``cv2.solvePnP`` and which way it points from
    the iris landmarks, then intersects that ray with the screen plane. Because
    the head's position and distance enter the arithmetic, moving your head does
    not move the point you are looking at.
``calibration``
    Fits a polynomial from screen-plane points to screen pixels, absorbing the
    camera mounting angle, the screen's tilt and individual eye geometry.
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
