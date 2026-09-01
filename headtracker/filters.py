"""Smoothing filters for cursor control.

Raw gaze estimates from a webcam jitter by a pixel or two on every frame even
when the user is perfectly still, while heavy smoothing makes the cursor lag
behind a fast glance.  The One Euro filter (Casiez, Roussel & Vogel, CHI 2012)
resolves exactly that trade-off: it uses a low cut-off while the signal is
still -- killing the jitter -- and raises the cut-off as the signal speeds up,
which removes the lag.

All filters here are time-aware and take an explicit timestamp, because webcam
frame intervals are irregular and a frame-count-based filter would change its
behaviour with the frame rate.

A constant-velocity Kalman filter was implemented and measured against One Euro
before being dropped.  Its steady state matched its own Riccati equation to the
second decimal (18.32 px predicted, 22.95 px measured once the 2-D radial mean
is accounted for), so it was correct -- it simply loses.  Gaze is long
fixations punctuated by saccades, and a model that only expects smooth
acceleration has to build up velocity before it follows a step.  At equal lag
One Euro was measurably steadier: 17.3 px against 27.5 px at 133 ms, and 13.1
px against 23.0 px at 167 ms.
"""

from __future__ import annotations

import math
from typing import Optional


def _alpha(cutoff_hz: float, interval_s: float) -> float:
    """Exponential smoothing factor for a given cut-off frequency."""
    tau = 1.0 / (2.0 * math.pi * max(cutoff_hz, 1e-6))
    return 1.0 / (1.0 + tau / max(interval_s, 1e-6))


class LowPassFilter:
    """First-order exponential smoother with a time-based coefficient."""

    def __init__(self, cutoff_hz: float = 5.0) -> None:
        self.cutoff_hz = float(cutoff_hz)
        self._value: Optional[float] = None
        self._last_t: float = 0.0

    def reset(self) -> None:
        self._value = None

    @property
    def value(self) -> Optional[float]:
        return self._value

    def filter(self, sample: float, timestamp: float) -> float:
        if self._value is None:
            self._value = float(sample)
            self._last_t = timestamp
            return self._value
        interval = timestamp - self._last_t
        self._last_t = timestamp
        coefficient = _alpha(self.cutoff_hz, interval)
        self._value = coefficient * float(sample) + (1.0 - coefficient) * self._value
        return self._value


class OneEuroFilter:
    """Adaptive low-pass filter tuned for pointing devices.

    ``min_cutoff`` sets the amount of smoothing while the cursor is nearly
    stationary (lower = steadier, but laggier on the first movement).
    ``beta`` sets how aggressively the filter opens up with speed
    (higher = snappier, but noisier during fast glances).
    """

    def __init__(
        self,
        min_cutoff: float = 1.4,
        beta: float = 0.06,
        d_cutoff: float = 1.0,
    ) -> None:
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x_prev: Optional[float] = None
        self._dx_prev = 0.0
        self._t_prev: Optional[float] = None

    def reset(self) -> None:
        """Forget the history, e.g. after tracking was lost."""
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None

    def filter(self, sample: float, timestamp: float) -> float:
        sample = float(sample)

        if self._t_prev is None or self._x_prev is None:
            self._x_prev = sample
            self._dx_prev = 0.0
            self._t_prev = timestamp
            return sample

        interval = timestamp - self._t_prev
        if interval <= 0:
            interval = 1e-3
        self._t_prev = timestamp

        derivative = (sample - self._x_prev) / interval
        d_alpha = _alpha(self.d_cutoff, interval)
        derivative_hat = d_alpha * derivative + (1.0 - d_alpha) * self._dx_prev

        cutoff = self.min_cutoff + self.beta * abs(derivative_hat)
        alpha = _alpha(cutoff, interval)
        smoothed = alpha * sample + (1.0 - alpha) * self._x_prev

        self._x_prev = smoothed
        self._dx_prev = derivative_hat
        return smoothed


class GlitchGate:
    """Rejects single-frame teleportation caused by a landmark tracker failure.

    MediaPipe occasionally jumps the mesh to a spurious position for one frame.
    Feeding that straight into the cursor makes it flicker across the screen, so
    any jump beyond ``max_speed`` px/s is dropped and the last good value is
    held instead.
    """

    def __init__(self, max_speed: float = 9000.0, recovery_frames: int = 2) -> None:
        self.max_speed = float(max_speed)
        self.recovery_frames = int(recovery_frames)
        self._last: Optional[tuple] = None
        self._bad_streak = 0

    def reset(self) -> None:
        self._last = None
        self._bad_streak = 0

    def check(self, x: float, y: float, timestamp: float) -> Optional[tuple]:
        """Return an accepted ``(x, y)`` or ``None`` when the sample is a glitch."""
        if self._last is None:
            self._last = (x, y, timestamp)
            self._bad_streak = 0
            return (x, y)

        last_x, last_y, last_t = self._last
        interval = timestamp - last_t
        if interval <= 0:
            return (x, y)

        distance = math.hypot(x - last_x, y - last_y)
        if distance / interval > self.max_speed:
            self._bad_streak += 1
            if self._bad_streak > self.recovery_frames:
                # Several "glitches" in a row mean the user really did move.
                self._last = (x, y, timestamp)
                self._bad_streak = 0
                return (x, y)
            return None

        self._last = (x, y, timestamp)
        self._bad_streak = 0
        return (x, y)
