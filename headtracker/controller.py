"""Turns gaze samples into cursor movement.

This is the layer where the two failure modes of the original implementation
are fixed.  It positions the cursor **absolutely** from the calibrated gaze, so
nothing integrates over time, and it smooths with a One Euro filter so that the
cursor is steady when the user is still yet still follows a quick glance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from .calibration import CalibrationModel
from .filters import GlitchGate, OneEuroFilter
from .geometry import GazeSample
from .mouse import AbsoluteMouse

Point = Tuple[float, float]


@dataclass
class CursorSettings:
    """Tunables exposed in the GUI."""

    gain: float = 1.0
    """Multiplier on the offset from the calibration centre."""

    min_cutoff: float = 0.3
    """One Euro cut-off in Hz while the gaze is nearly still -- lower is steadier.

    Measured on synthetic faces with 1 px of landmark jitter at 30 fps on a
    1920x1080 screen: 0.3 Hz holds the resting cursor to about 17 px of wobble
    against 46 px unfiltered, and reaches 90% of a 1200 px glance in 133 ms.
    Dropping to 0.15 Hz buys another 4 px of steadiness for 34 ms more lag.
    """

    beta: float = 1.1
    """How fast the filter opens up with gaze speed, in Hz per screen height/s.

    Expressed as a fraction of the screen rather than in pixels so that the
    same number means the same thing on any monitor: the controller divides it
    by the screen height when it reaches the filter.  In pixels the term is
    large -- a 1 px landmark wobble at 30 fps reads as roughly 900 px/s -- so a
    beta carried over from an angle-space tuning silently opens the filter
    right up, which is what made 0.05 behave as if no filter were running.
    """

    max_speed: float = 9000.0
    """Cursor speed above which a frame is treated as a tracking glitch."""

    hold_on_invalid_s: float = 0.3
    """Keep the cursor still for this long after tracking is lost."""


class GazeCursorController:
    """Maps :class:`GazeSample` objects onto the physical cursor."""

    def __init__(
        self,
        mouse: AbsoluteMouse,
        model: Optional[CalibrationModel] = None,
        settings: Optional[CursorSettings] = None,
    ) -> None:
        self.mouse = mouse
        self.settings = settings or CursorSettings()
        self.model = model or CalibrationModel.default(mouse.screen)
        self._filter_x = OneEuroFilter()
        self._filter_y = OneEuroFilter()
        self._gate = GlitchGate()
        self._invalid_since: Optional[float] = None
        self._holding = False
        # Constructing the filters above and pushing the settings in here, rather
        # than passing them to the constructors, keeps the attributes the only
        # place the tunables live.  Skipping this step silently ran the filters
        # at their library defaults no matter what the caller asked for.
        self.apply_settings(self.settings)

    # -- configuration ------------------------------------------------------
    def set_model(self, model: CalibrationModel) -> None:
        """Install a freshly fitted calibration and discard the filter history.

        Without the reset the cursor would glide from its old position to the
        new mapping over several frames, which reads as a lurch.
        """
        self.model = model
        self.reset()

    def apply_settings(self, settings: CursorSettings) -> None:
        self.settings = settings
        beta = settings.beta / self._beta_scale()
        self._filter_x.min_cutoff = settings.min_cutoff
        self._filter_x.beta = beta
        self._filter_y.min_cutoff = settings.min_cutoff
        self._filter_y.beta = beta
        self._gate.max_speed = settings.max_speed

    def _beta_scale(self) -> float:
        """Screen height, the unit ``CursorSettings.beta`` is expressed in.

        The filter's speed term is measured in pixels per second, so a beta
        written per pixel would mean something different on every monitor.
        Dividing by the screen height converts the setting into that space and
        leaves the value the user sees resolution independent.
        """
        _, height = self.mouse.screen
        return float(height) if height > 0 else 1.0

    def reset(self) -> None:
        """Forget all smoothing state, e.g. after tracking was lost."""
        self._filter_x.reset()
        self._filter_y.reset()
        self._gate.reset()
        self._invalid_since = None
        self._holding = False

    @property
    def is_holding(self) -> bool:
        """True while the cursor is deliberately frozen."""
        return self._holding

    # -- main path ----------------------------------------------------------
    def gaze_to_screen(self, sample: GazeSample) -> Point:
        """Project a gaze sample through the calibration, before smoothing."""
        raw_x, raw_y = self.model.predict_one(sample.yaw, sample.pitch)
        shift_x, shift_y = self._head_shift(sample)
        raw_x += shift_x
        raw_y += shift_y

        gain = self.settings.gain
        reference = self.model.reference
        if reference is not None and abs(gain - 1.0) > 1e-9:
            centre_x, centre_y = self.model.predict_one(reference[0], reference[1])
            raw_x = centre_x + (raw_x - centre_x) * gain
            raw_y = centre_y + (raw_y - centre_y) * gain

        return self.model.clamp_to_screen((raw_x, raw_y))

    def _head_shift(self, sample: GazeSample) -> Point:
        """How far the look-at point moved because the *head* moved, in pixels.

        The calibration maps a gaze angle to a pixel, which is only true for
        the head position the user calibrated from.  Slide sideways and the
        same angle now lands somewhere else; lean back and the same angle
        sweeps further, because the lever arm from eye to screen is longer.

        Both follow from intersecting the gaze ray with the screen plane.  With
        the eye at ``(ex, ey, ez)`` the hit point is ``ex + ez*tan(yaw)``, so
        against the calibration pose ``(ex0, ey0, ez0)`` it moves by

            (ex - ex0) + (ez - ez0) * tan(yaw)

        Measuring the displacement from the calibration pose rather than from
        the origin is what makes this safe.  The translation comes from solving
        the pose against a canonical face model that is not the user's face, so
        it carries an unknown offset -- but that offset is the same at
        calibration time and at run time, and subtracting cancels it.  Only the
        *change* survives, which is the part that actually matters.

        This replaced an earlier version that fitted the calibration directly on
        screen-plane coordinates.  That was mathematically tidier and measurably
        worse in use: it made the whole mapping depend on the absolute accuracy
        of the pose solve, which the test rig cannot check because it generates
        faces from the same model the solver uses.
        """
        reference = self.model.reference_eye
        if reference is None or not sample.head_translation:
            return 0.0, 0.0

        # The solve's translation, not the eye position.  The eye also swings
        # sideways when the head turns, but the gaze angle already reports that
        # -- adding it again double-counts.  Measured at 12 deg of head yaw,
        # compensating the eye position made the cursor 14 px *worse*, while
        # compensating the translation alone is what fixes sliding and leaning.
        ex, ey, ez = sample.head_translation
        ex0, ey0, ez0 = reference
        if ez <= 0 or ez0 <= 0:
            return 0.0, 0.0

        yaw = math.radians(sample.yaw)
        pitch = math.radians(sample.pitch)
        shift_x = (ex - ex0) + (ez - ez0) * math.tan(yaw)
        shift_y = (ey - ey0) + (ez - ez0) * math.tan(pitch)

        return (
            shift_x * self._pixels_per_unit(sample.yaw, sample.pitch, axis=0, depth=ez),
            shift_y * self._pixels_per_unit(sample.yaw, sample.pitch, axis=1, depth=ez),
        )

    def _pixels_per_unit(
        self, yaw: float, pitch: float, axis: int, depth: float
    ) -> float:
        """Screen pixels per face-model unit, at this gaze direction.

        The shift above is in model units and the cursor needs pixels.  Rather
        than assume a scale, read it off the calibration: moving the gaze by a
        small angle moves the hit point by ``ez * sec^2(angle)`` model units and
        the cursor by whatever the fitted mapping says, so the ratio of the two
        is the local scale.  That keeps the correction consistent with the
        calibration however the camera is mounted.
        """
        step = 0.25
        if axis == 0:
            before = self.model.predict_one(yaw - step, pitch)[0]
            after = self.model.predict_one(yaw + step, pitch)[0]
            angle = math.radians(yaw)
        else:
            before = self.model.predict_one(yaw, pitch - step)[1]
            after = self.model.predict_one(yaw, pitch + step)[1]
            angle = math.radians(pitch)

        pixels_per_degree = (after - before) / (2.0 * step)
        units_per_degree = depth * (1.0 / math.cos(angle)) ** 2 * math.pi / 180.0
        if abs(units_per_degree) < 1e-9:
            return 0.0
        return pixels_per_degree / units_per_degree

    def update(self, sample: GazeSample, timestamp: float) -> Optional[Point]:
        """Advance one frame.  Returns the new cursor position, or ``None``."""
        if not sample.valid:
            return self._handle_invalid(timestamp)

        if self._holding:
            # Tracking came back: start from the current gaze with no history so
            # the cursor does not fly in from wherever it was left.
            self.reset()

        target = self.gaze_to_screen(sample)
        accepted = self._gate.check(target[0], target[1], timestamp)
        if accepted is None:
            return None

        smoothed = (
            self._filter_x.filter(accepted[0], timestamp),
            self._filter_y.filter(accepted[1], timestamp),
        )
        smoothed = self.model.clamp_to_screen(smoothed)
        self.mouse.move_to(smoothed[0], smoothed[1])
        return smoothed

    def _handle_invalid(self, timestamp: float) -> None:
        """Freeze, then release, the cursor while the face is not tracked."""
        if self._invalid_since is None:
            self._invalid_since = timestamp
        if timestamp - self._invalid_since > self.settings.hold_on_invalid_s:
            if not self._holding:
                self.reset()
                self._holding = True
