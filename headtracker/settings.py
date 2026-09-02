"""Persistent user settings stored as JSON next to the calibration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Optional

#: Where the user's settings and calibration are kept between runs.
CONFIG_DIR = Path.home() / ".config" / "headtracker"
SETTINGS_PATH = CONFIG_DIR / "settings.json"
CALIBRATION_PATH = CONFIG_DIR / "calibration.json"


@dataclass
class AppSettings:
    """Everything the GUI needs to restore between runs."""

    enabled: bool = False
    gain: float = 1.0
    min_cutoff: float = 0.8
    beta: float = 0.05
    max_speed: float = 9000.0
    compensate_distance: bool = True
    use_eyes: bool = True
    camera_index: int = 0
    # Requested, not guaranteed -- see TrackingEngine.resolution_shortfall.
    # Resolution is the single biggest lever on accuracy: measured single-frame
    # gaze noise at 1 px of landmark jitter is 1.18 deg at 720p, 0.79 deg at
    # 1080p and 0.60 deg at 1440p.  The screen amplifies that by roughly 37 px
    # per degree, so the difference between 720p and 1080p is tens of pixels.
    camera_width: int = 1920
    camera_height: int = 1080
    camera_fov_deg: float = 60.0
    min_eye_open: float = 0.18
    wink_click: bool = True
    dwell_click: bool = False
    dwell_s: float = 0.7
    wink_close: float = 0.19
    wink_open: float = 0.24
    cooldown_s: float = 0.6
    mirror_preview: bool = True
    calibration_degree: int = 2

    @classmethod
    def load(cls, path: Path) -> "AppSettings":
        """Read settings, ignoring unknown keys from older versions."""
        if not path.is_file():
            return cls()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            return cls()
        if not isinstance(raw, dict):
            return cls()
        known = {f.name: f.type for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})

    def save(self, path: Optional[Path]) -> None:
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
