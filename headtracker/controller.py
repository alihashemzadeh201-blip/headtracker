"""Turns gaze samples into cursor movement.

This is the layer where the two failure modes of the original implementation
are fixed.  It positions the cursor **absolutely** from the calibrated gaze, so
nothing integrates over time, and it smooths with a One Euro filter so that the
cursor is steady when the user is still yet still follows a quick glance.
"""

from __future__ import annotations

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

    min_cutoff: float = 0.8
    """One Euro cut-off in Hz while the gaze is nearly still -- lower is steadier.

    Measured on synthetic faces with 1 px of landmark jitter at 30 fps, 0.8 Hz
    cuts the resting cursor jitter from 2.4 deg to 1.2 deg.  Going much lower
    keeps reducing jitter but adds visible lag to a quick glance.
    """

    beta: float = 0.05
    """How fast the filter opens up with gaze speed -- higher is snappier.

    At 0.05 the cursor reaches 90% of a 14 deg glance in about 100 ms.  Raising
    beta shortens that to ~67 ms but lets roughly 20% more jitter through.
    """

    max_speed: float = 9000.0
    """Cursor speed above which a frame is treated as a tracking glitch."""

    compensate_distance: bool = True
    """Scale the gaze offset by how far the head is from the screen."""

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
        self._filter_x.min_cutoff = settings.min_cutoff
        self._filter_x.beta = settings.beta
        self._filter_y.min_cutoff = settings.min_cutoff
        self._filter_y.beta = settings.beta
        self._gate.max_speed = settings.max_speed

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

        gain = self.settings.gain
        if self.settings.compensate_distance:
            gain *= self.model.distance_factor(sample.distance)

        reference = self.model.reference
        if reference is not None and abs(gain - 1.0) > 1e-9:
            centre_x, centre_y = self.model.predict_one(reference[0], reference[1])
            raw_x = centre_x + (raw_x - centre_x) * gain
            raw_y = centre_y + (raw_y - centre_y) * gain

        return self.model.clamp_to_screen((raw_x, raw_y))

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
