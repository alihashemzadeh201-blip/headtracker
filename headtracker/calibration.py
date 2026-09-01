"""Mapping calibrated gaze angles onto screen coordinates.

Gaze angles are only a *direction*; where that direction lands on a particular
monitor depends on the camera position, the monitor distance, the lens and the
individual geometry of the user's eyes.  None of those can be assumed, so they
are measured: the user looks at a handful of known points and a least-squares
polynomial is fitted from angles to pixels.

The fit is deliberately low order.  A degree-2 surface already captures the
pincushion/barrel curvature of a webcam view, and higher orders start fitting
the noise of the calibration samples instead of the real mapping.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

Point = Tuple[float, float]

#: Fraction of the screen edge kept away from the border, so the outermost
#: calibration points are comfortably reachable without the user straining.
GRID_MARGIN = 0.08

MIN_SAMPLES_PER_POINT = 6


def grid_points(columns: int = 3, rows: int = 3, margin: float = GRID_MARGIN) -> List[Point]:
    """Return evenly spaced normalised ``(x, y)`` points in ``[margin, 1-margin]``."""
    columns = max(2, int(columns))
    rows = max(2, int(rows))
    low, high = margin, 1.0 - margin
    xs = np.linspace(low, high, columns)
    ys = np.linspace(low, high, rows)
    # Centre-first ordering: the most-used region of the screen is captured
    # first, so an interrupted calibration is still usable.
    points = [(float(x), float(y)) for y in ys for x in xs]
    centre = np.array([0.5, 0.5])
    points.sort(key=lambda p: float(np.linalg.norm(np.array(p) - centre)))
    return points


def design_matrix(features: np.ndarray, degree: int) -> np.ndarray:
    """Build the polynomial design matrix for ``(N, 2)`` angle features."""
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or features.shape[1] != 2:
        raise ValueError("features must have shape (N, 2)")
    if degree not in (1, 2):
        raise ValueError("degree must be 1 or 2")

    x, y = features[:, 0], features[:, 1]
    columns = [np.ones_like(x), x, y]
    if degree == 2:
        columns += [x * x, x * y, y * y]
    return np.column_stack(columns)


def coefficient_count(degree: int) -> int:
    """Number of fitted coefficients per output axis."""
    return 3 if degree == 1 else 6


@dataclass
class CalibrationReport:
    """Quality of a completed calibration, in screen pixels."""

    rms_error: float = float("inf")
    max_error: float = float("inf")
    mean_error: float = float("inf")
    points_used: int = 0
    points_rejected: int = 0
    per_point: List[float] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        """True when the mapping is good enough to drive a cursor."""
        return self.points_used >= 4 and np.isfinite(self.rms_error)

    def describe(self) -> str:
        if not self.usable:
            return "calibration incomplete"
        return (
            f"mean {self.mean_error:.0f}px, worst {self.max_error:.0f}px "
            f"({self.points_used} points)"
        )


class CalibrationModel:
    """Fitted polynomial from gaze angles to screen pixels."""

    def __init__(self, degree: int = 2, ridge: float = 1e-3) -> None:
        self.degree = int(degree)
        self.ridge = float(ridge)
        self.mean: Optional[np.ndarray] = None
        self.scale: Optional[np.ndarray] = None
        self.coefficients: Optional[np.ndarray] = None
        self.reference: Optional[Point] = None
        self.reference_distance: float = 0.0
        self.screen: Point = (0.0, 0.0)
        self.report = CalibrationReport()

    @property
    def is_fitted(self) -> bool:
        return self.coefficients is not None and self.mean is not None

    @classmethod
    def default(cls, screen: Point) -> "CalibrationModel":
        """A rough, uncalibrated map so the tracker is usable on first launch.

        It assumes a straight-ahead gaze hits the middle of the screen and that
        about +-18 deg of yaw and +-12 deg of pitch sweep the full width and
        height.  Real setups deviate from that, which is exactly what
        :meth:`fit` corrects -- but it is close enough to be usable immediately.
        """
        width, height = float(screen[0]), float(screen[1])
        model = cls(degree=1)
        model.mean = np.zeros(2)
        model.scale = np.ones(2)
        model.coefficients = np.array(
            [
                [width / 2.0, width / 36.0, 0.0],
                [height / 2.0, 0.0, height / 24.0],
            ],
            dtype=np.float64,
        ).T
        model.screen = (width, height)
        model.reference = (0.0, 0.0)
        model.reference_distance = 0.0
        model.report = CalibrationReport()
        return model

    # -- fitting ------------------------------------------------------------
    # The fit has to hold the design matrix, the keep-mask and both residual
    # sets at once; splitting it would only scatter one computation.
    def fit(  # pylint: disable=too-many-locals
        self,
        features: np.ndarray,
        targets: np.ndarray,
        screen: Point,
        reference_distance: float = 0.0,
        reject_outliers: bool = True,
    ) -> CalibrationReport:
        """Fit angles -> pixels, reporting the residual error in pixels."""
        features = np.asarray(features, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.float64)
        if features.shape[0] != targets.shape[0]:
            raise ValueError("features and targets must have the same length")

        needed = coefficient_count(self.degree)
        if features.shape[0] < needed:
            raise ValueError(
                f"need at least {needed} calibration points for degree {self.degree}, "
                f"got {features.shape[0]}"
            )
        if not np.all(np.isfinite(features)) or not np.all(np.isfinite(targets)):
            raise ValueError("calibration data contains non-finite values")

        self.mean = features.mean(axis=0)
        spread = features.std(axis=0)
        self.scale = np.where(spread < 1e-6, 1.0, spread)

        normalised = (features - self.mean) / self.scale
        keep = np.ones(features.shape[0], dtype=bool)
        rejected = 0

        for _ in range(3 if reject_outliers else 1):
            matrix = design_matrix(normalised[keep], self.degree)
            self.coefficients = self._solve(matrix, targets[keep])
            if not reject_outliers or keep.sum() <= needed + 1:
                break
            residuals = np.linalg.norm(
                self.predict(features[keep]) - targets[keep], axis=1
            )
            limit = max(2.5 * float(np.sqrt(np.mean(residuals ** 2))), 25.0)
            good = residuals <= limit
            if good.all() or good.sum() < needed:
                break
            keep &= self._index_mask(good, keep)
            rejected = int((~keep).sum())

        all_residuals = np.linalg.norm(self.predict(features) - targets, axis=1)
        kept_residuals = all_residuals[keep]
        self.screen = (float(screen[0]), float(screen[1]))
        self.reference_distance = float(reference_distance)
        self.report = CalibrationReport(
            rms_error=float(np.sqrt(np.mean(kept_residuals ** 2))),
            max_error=float(np.max(all_residuals)),
            mean_error=float(np.mean(kept_residuals)),
            points_used=int(keep.sum()),
            points_rejected=rejected,
            per_point=[float(v) for v in all_residuals],
        )
        return self.report

    @staticmethod
    def _index_mask(mask_on_kept: np.ndarray, keep: np.ndarray) -> np.ndarray:
        """Expand a mask over kept rows back to a mask over all rows."""
        expanded = np.zeros_like(keep)
        expanded[keep] = mask_on_kept
        return expanded

    def _solve(self, matrix: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Ridge least squares, leaving the intercept unpenalised."""
        gram = matrix.T @ matrix
        penalty = self.ridge * np.eye(gram.shape[0])
        penalty[0, 0] = 0.0
        try:
            return np.linalg.solve(gram + penalty, matrix.T @ targets)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(matrix, targets, rcond=None)[0]

    # -- prediction ---------------------------------------------------------
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Map ``(N, 2)`` gaze angles to ``(N, 2)`` screen pixels."""
        if not self.is_fitted:
            raise RuntimeError("calibration model has not been fitted")
        features = np.atleast_2d(np.asarray(features, dtype=np.float64))
        normalised = (features - self.mean) / self.scale
        matrix = design_matrix(normalised, self.degree)
        return matrix @ self.coefficients

    def predict_one(self, yaw: float, pitch: float) -> Point:
        """Map a single gaze angle pair to a screen pixel."""
        x, y = self.predict(np.array([[yaw, pitch]]))[0]
        return float(x), float(y)

    def clamp_to_screen(self, point: Point) -> Point:
        width, height = self.screen
        if width <= 0 or height <= 0:
            return point
        return (
            min(max(point[0], 0.0), width - 1.0),
            min(max(point[1], 0.0), height - 1.0),
        )

    def distance_factor(self, distance: float) -> float:
        """Scale factor compensating for leaning towards or away from the screen.

        A gaze angle sweeps a longer arc across the screen the further the head
        is from it, so the angular offset from the calibration pose is scaled by
        the ratio of the current to the reference head distance.
        """
        if self.reference_distance <= 0 or distance <= 0:
            return 1.0
        return min(max(distance / self.reference_distance, 0.6), 1.8)

    # -- persistence --------------------------------------------------------
    def to_dict(self) -> dict:
        if not self.is_fitted:
            return {}
        return {
            "version": 2,
            "degree": self.degree,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "coefficients": self.coefficients.tolist(),
            "reference": list(self.reference) if self.reference else None,
            "reference_distance": self.reference_distance,
            "screen": list(self.screen),
            "rms_error": self.report.rms_error,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CalibrationModel":
        model = cls(degree=int(data.get("degree", 2)))
        model.mean = np.array(data["mean"], dtype=np.float64)
        model.scale = np.array(data["scale"], dtype=np.float64)
        model.coefficients = np.array(data["coefficients"], dtype=np.float64)
        reference = data.get("reference")
        model.reference = tuple(reference) if reference else None
        model.reference_distance = float(data.get("reference_distance", 0.0))
        model.screen = tuple(data.get("screen", (0.0, 0.0)))
        model.report = CalibrationReport(
            rms_error=float(data.get("rms_error", float("inf"))),
            points_used=len(data.get("per_point", [])) or coefficient_count(model.degree),
        )
        return model

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Optional["CalibrationModel"]:
        if not path.exists():
            return None
        try:
            return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (ValueError, KeyError, OSError, json.JSONDecodeError):
            return None


class CalibrationSession:
    """Collects gaze samples while the user looks at a sequence of points.

    Deliberately free of any GUI code: the caller draws
    :meth:`current_point` and feeds samples in, which keeps the whole
    calibration pipeline unit-testable without a display.
    """

    def __init__(
        self,
        points: Sequence[Point],
        screen: Point,
        dwell_s: float = 1.1,
        countdown_s: float = 0.6,
    ) -> None:
        if len(points) < 3:
            raise ValueError("at least 3 calibration points are required")
        self.points: List[Point] = [tuple(p) for p in points]  # type: ignore[misc]
        self.screen: Point = (float(screen[0]), float(screen[1]))
        self.dwell_s = float(dwell_s)
        self.countdown_s = float(countdown_s)
        self.index = 0
        self._samples: List[Point] = []
        self._distances: List[float] = []
        self._state = "countdown"
        self._state_started: Optional[float] = None
        self.features: List[Point] = []
        self.targets: List[Point] = []
        self.reference_distance = 0.0
        self.finished = False

    @property
    def total_points(self) -> int:
        return len(self.points)

    def current_point(self) -> Optional[Point]:
        """Normalised target for the point currently being shown."""
        if self.finished:
            return None
        return self.points[self.index]

    def current_target(self) -> Point:
        """The current point in screen pixels."""
        point = self.current_point()
        if point is None:
            return (0.0, 0.0)
        return (point[0] * self.screen[0], point[1] * self.screen[1])

    def is_collecting(self) -> bool:
        return not self.finished and self._state == "collect"

    def progress(self) -> float:
        """Fraction of the current point's collection window that has elapsed."""
        if self._state_started is None:
            return 0.0
        return min(1.0, self._samples_progress())

    def _samples_progress(self) -> float:
        return len(self._samples) / max(MIN_SAMPLES_PER_POINT, 1)

    def start(self, timestamp: float) -> None:
        self._state = "countdown"
        self._state_started = timestamp

    def add_sample(self, yaw: float, pitch: float, distance: float = 0.0) -> bool:
        """Record a gaze sample; returns ``True`` while the point is still filling."""
        if not self.is_collecting():
            return False
        if not math.isfinite(yaw) or not math.isfinite(pitch):
            return True
        self._samples.append((float(yaw), float(pitch)))
        if distance > 0 and math.isfinite(distance):
            self._distances.append(float(distance))
        return True

    def update(  # pylint: disable=too-many-return-statements
        self, timestamp: float
    ) -> bool:
        """Advance the state machine; returns ``True`` once calibration is done."""
        if self.finished:
            return True
        if self._state_started is None:
            self.start(timestamp)
            return False

        elapsed = timestamp - self._state_started
        if self._state == "countdown" and elapsed >= self.countdown_s:
            self._state = "collect"
            self._state_started = timestamp
            self._samples = []
            self._distances = []
            return False

        if self._state != "collect":
            return False
        if elapsed < self.dwell_s or len(self._samples) < MIN_SAMPLES_PER_POINT:
            return False

        self._commit_point()
        self.index += 1
        if self.index >= len(self.points):
            self.finished = True
            return True
        self._state = "countdown"
        self._state_started = timestamp
        return False

    def _commit_point(self) -> None:
        """Store the median gaze for the current point.

        The median rather than the mean, so that one blink or tracking glitch
        during the dwell cannot drag the fitted surface away from the truth.
        """
        if not self._samples:
            return
        stacked = np.array(self._samples, dtype=np.float64)
        yaw, pitch = np.median(stacked, axis=0)
        self.features.append((float(yaw), float(pitch)))
        self.targets.append(self.current_target())
        if self._distances:
            self.reference_distance += float(np.median(self._distances))

    def build(self, degree: int = 2, ridge: float = 1e-3) -> CalibrationModel:
        """Fit the collected samples into a usable model."""
        if len(self.features) < coefficient_count(degree):
            raise ValueError(
                f"only {len(self.features)} points were captured, "
                f"degree {degree} needs {coefficient_count(degree)}"
            )
        model = CalibrationModel(degree=degree, ridge=ridge)
        if self.reference_distance > 0:
            model.reference_distance = self.reference_distance / len(self.features)
        model.reference = (
            float(np.mean([f[0] for f in self.features])),
            float(np.mean([f[1] for f in self.features])),
        )
        model.fit(
            np.array(self.features, dtype=np.float64),
            np.array(self.targets, dtype=np.float64),
            screen=self.screen,
            reference_distance=model.reference_distance,
        )
        return model
