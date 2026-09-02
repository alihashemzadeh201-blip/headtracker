"""Camera -> landmarks -> gaze -> cursor, with no GUI involved.

Both the terminal runner in :mod:`headtracker.app` and the window in
:mod:`headtracker.gui` drive this, which keeps the whole control path testable
without a display server and stops the two front ends from importing each other.
"""

from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np

from .calibration import CalibrationModel, CalibrationSession, grid_points
from .controller import CursorSettings, GazeCursorController
from .geometry import GazeSample
from .mouse import AbsoluteMouse, create_backend
from .settings import CALIBRATION_PATH, AppSettings
from .tracking import FaceGazeTracker, ensure_model

class TrackingEngine:
    """Camera -> landmarks -> gaze -> cursor, with no GUI involved.

    Kept separate from the window code so the whole control path can be driven
    and tested without a display server.
    """

    def __init__(self, settings: AppSettings, model: Optional[CalibrationModel] = None) -> None:
        self.settings = settings
        self.mouse = AbsoluteMouse(create_backend())
        self.controller = GazeCursorController(
            self.mouse,
            model or CalibrationModel.default(self.mouse.screen),
            self.cursor_settings(),
        )
        self.tracker = FaceGazeTracker(
            ensure_model(),
            camera_fov_deg=settings.camera_fov_deg,
            min_eye_open=settings.min_eye_open,
            use_eyes=settings.use_eyes,
        )
        self.capture = self._open_camera()
        self.estimator_size = (0, 0)
        self.last_sample: Optional[GazeSample] = None
        self.fps = 0.0
        self._frames = 0
        self._dwell_since: Optional[float] = None
        self._dwell_anchor: tuple = (0.0, 0.0)
        self._dwell_armed = True
        self._fps_clock: Optional[float] = None

    def _open_camera(self) -> cv2.VideoCapture:
        capture = cv2.VideoCapture(self.settings.camera_index)
        # These are requests; many webcams only offer a fixed set of modes and
        # silently hand back something else, so read the real values back.
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.settings.camera_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.settings.camera_height)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return capture

    @property
    def camera_resolution(self) -> tuple:
        if not self.capture.isOpened():
            return (0, 0)
        return (
            int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def resolution_shortfall(self) -> tuple:
        """``(requested, actual)`` if the webcam ignored the request, else ``None``.

        Resolution is the single biggest lever on accuracy, and ``CAP_PROP_*``
        is a request that many webcams quietly refuse.  Measured single-frame
        gaze noise at 1 px of landmark jitter: 1.18 deg at 720p, 0.79 deg at
        1080p, 0.60 deg at 1440p -- and the rig amplifies that by ~37 px per
        degree, so a camera stuck at 640x480 costs tens of pixels of accuracy
        with nothing on screen to explain it.
        """
        got = self.camera_resolution
        want = (self.settings.camera_width, self.settings.camera_height)
        if got[0] <= 0 or got[1] <= 0:
            return None
        if got[0] >= want[0] and got[1] >= want[1]:
            return None
        return (want, got)

    def cursor_settings(self) -> CursorSettings:
        settings = self.settings
        return CursorSettings(
            gain=settings.gain,
            min_cutoff=settings.min_cutoff,
            beta=settings.beta,
            max_speed=settings.max_speed,
            compensate_distance=settings.compensate_distance,
        )

    def apply_settings(self) -> None:
        self.controller.apply_settings(self.cursor_settings())

    # -- calibration --------------------------------------------------------
    def start_calibration(self, columns: int = 5, rows: int = 4) -> CalibrationSession:
        return CalibrationSession(
            grid_points(columns, rows),
            screen=self.mouse.screen,
            dwell_s=1.1,
            countdown_s=0.6,
        )

    def install_calibration(self, model: CalibrationModel) -> None:
        self.controller.set_model(model)
        model.save(CALIBRATION_PATH)

    # -- per-frame ----------------------------------------------------------
    def read_frame(self) -> Optional[np.ndarray]:
        if not self.capture.isOpened():
            return None
        ok, frame = self.capture.read()
        return frame if ok else None

    def step(self, frame: np.ndarray, enabled: bool, now: Optional[float] = None) -> GazeSample:
        """Process one frame and move the cursor if tracking is enabled."""
        now = time.monotonic() if now is None else now
        sample = self.tracker.process(frame, int(now * 1000))
        self.last_sample = sample
        self._tick_fps(now)

        if enabled:
            self.controller.update(sample, now)
            self._handle_dwell_click(now)
        return sample

    def _tick_fps(self, now: float) -> None:
        """Rolling frame rate, measured over ~0.5 s windows."""
        if self._fps_clock is None:
            self._fps_clock = now
        self._frames += 1
        elapsed = now - self._fps_clock
        if elapsed >= 0.5:
            self.fps = self._frames / elapsed
            self._frames = 0
            self._fps_clock = now

    #: How far the cursor must leave a spot before dwell can trigger there again.
    DWELL_RADIUS_PX = 40.0

    def _handle_dwell_click(self, now: float) -> None:
        """Click after the cursor has held still on a spot for ``dwell_s``.

        Fires once per visit: staring at the same place must not produce a
        stream of clicks, so the dwell only re-arms once the cursor has left
        the area it just clicked in.
        """
        if not self.settings.dwell_click:
            return

        x, y = self.mouse.target()
        anchor = self._dwell_anchor
        moved = (x - anchor[0]) ** 2 + (y - anchor[1]) ** 2 > self.DWELL_RADIUS_PX ** 2

        if moved:
            self._dwell_anchor = (x, y)
            self._dwell_since = now
            self._dwell_armed = True
            return

        if not self._dwell_armed:
            return
        if self._dwell_since is None:
            self._dwell_since = now
            return
        if now - self._dwell_since >= self.settings.dwell_s:
            self.mouse.click("left")
            self._dwell_armed = False
            self._dwell_since = None

    def close(self) -> None:
        self.capture.release()
        self.tracker.close()


def check_wink(sample: GazeSample, settings: AppSettings) -> bool:
    """True when exactly one eye is closed and the other is open."""
    if not settings.wink_click or sample.left_eye_open <= 0 or sample.right_eye_open <= 0:
        return False
    closed, opened = settings.wink_close, settings.wink_open
    left_wink = sample.left_eye_open < closed and sample.right_eye_open > opened
    right_wink = sample.right_eye_open < closed and sample.left_eye_open > opened
    return left_wink or right_wink
