"""The non-GUI parts of the application: engine plumbing, clicks, settings."""

from __future__ import annotations

import numpy as np
import pytest

from headtracker import app as app_module
from headtracker import engine as engine_module
from headtracker.app import build_parser
from headtracker.engine import TrackingEngine, check_wink
from headtracker.geometry import GazeSample
from headtracker.mouse import NullBackend
from headtracker.settings import AppSettings


def sample(left: float = 0.3, right: float = 0.3, **kwargs) -> GazeSample:
    return GazeSample(left_eye_open=left, right_eye_open=right, valid=True, **kwargs)


# --------------------------------------------------------------------------
# Wink detection
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "left, right, expected",
    [
        (0.10, 0.30, True),  # left eye shut, right eye open
        (0.30, 0.10, True),  # right eye shut, left eye open
        (0.30, 0.30, False),  # both open
        (0.10, 0.10, False),  # both shut: a blink, not a wink
        (0.10, 0.20, False),  # neither eye clearly open
        (0.22, 0.30, False),  # left eye only half closed
    ],
)
def test_check_wink(left, right, expected):
    settings = AppSettings(wink_close=0.19, wink_open=0.24)
    assert check_wink(sample(left, right), settings) is expected


def test_wink_can_be_disabled():
    settings = AppSettings(wink_click=False)
    assert check_wink(sample(0.05, 0.35), settings) is False


def test_a_sample_without_eye_data_is_not_a_wink():
    settings = AppSettings()
    assert check_wink(GazeSample(valid=False), settings) is False


# --------------------------------------------------------------------------
# Settings persistence
# --------------------------------------------------------------------------
def test_settings_round_trip(tmp_path):
    path = tmp_path / "settings.json"
    original = AppSettings(gain=1.75, min_cutoff=0.6, camera_index=2, dwell_click=True)
    original.save(path)

    restored = AppSettings.load(path)
    assert restored.gain == 1.75
    assert restored.min_cutoff == 0.6
    assert restored.camera_index == 2
    assert restored.dwell_click is True


def test_unknown_keys_from_a_newer_version_are_ignored(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text('{"gain": 2.0, "feature_from_the_future": 1}', encoding="utf-8")
    assert AppSettings.load(path).gain == 2.0


@pytest.mark.parametrize("content", ["", "not json", "[1,2,3]", "null"])
def test_a_broken_settings_file_falls_back_to_defaults(tmp_path, content):
    path = tmp_path / "settings.json"
    path.write_text(content, encoding="utf-8")
    assert AppSettings.load(tmp_path / "settings.json").gain == AppSettings().gain


def test_missing_settings_file_gives_defaults(tmp_path):
    assert AppSettings.load(tmp_path / "absent.json").gain == AppSettings().gain


def test_saving_to_none_is_a_noop():
    AppSettings().save(None)


# --------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------
def test_parser_defaults():
    args = build_parser().parse_args([])
    assert not args.headless and not args.calibrate and args.grid == 4


def test_parser_overrides():
    args = build_parser().parse_args(
        ["--headless", "--calibrate", "--camera", "2", "--width", "1280", "--grid", "5"]
    )
    assert args.headless and args.calibrate
    assert args.camera == 2 and args.width == 1280 and args.grid == 5


def test_main_applies_camera_overrides(monkeypatch):
    captured = {}

    def fake_headless(settings, columns, rows):
        captured.update(settings=settings, columns=columns, rows=rows)
        return 0
    monkeypatch.setattr(app_module, "run_headless", fake_headless)
    monkeypatch.setattr(AppSettings, "load", classmethod(lambda cls, _path: cls()))

    assert app_module.main(["--headless", "--calibrate", "--grid", "3", "--camera", "1"]) == 0
    assert captured["columns"] == 3 and captured["rows"] == 3
    assert captured["settings"].camera_index == 1


def test_headless_without_calibrate_skips_the_grid(monkeypatch):
    captured = {}

    def fake_headless(_settings, columns, rows):
        captured.update(columns=columns, rows=rows)
        return 0

    monkeypatch.setattr(app_module, "run_headless", fake_headless)
    monkeypatch.setattr(AppSettings, "load", classmethod(lambda cls, _path: cls()))

    app_module.main(["--headless"])
    assert captured["columns"] == 0


def test_grid_size_is_clamped_to_a_sane_range(monkeypatch):
    captured = {}

    def fake_headless(_settings, columns, _rows):
        captured.update(columns=columns)
        return 0

    monkeypatch.setattr(app_module, "run_headless", fake_headless)
    monkeypatch.setattr(AppSettings, "load", classmethod(lambda cls, _path: cls()))

    app_module.main(["--headless", "--calibrate", "--grid", "99"])
    assert captured["columns"] == 6


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------
class StubCapture:
    """An OpenCV capture that always returns a black 1080p frame."""

    def __init__(self):
        self.opened = True
        self.released = False

    def isOpened(self):  # pylint: disable=invalid-name  # OpenCV's own name
        return self.opened

    def set(self, *_args):
        return True

    def get(self, prop):
        import cv2  # pylint: disable=import-outside-toplevel

        return {
            cv2.CAP_PROP_FRAME_WIDTH: 1920.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080.0,
        }.get(prop, 0.0)

    def read(self):
        return True, np.zeros((1080, 1920, 3), dtype=np.uint8)

    def release(self):
        self.released = True


@pytest.fixture(name="engine")
def fixture_engine(monkeypatch):
    """A TrackingEngine with the camera, landmarker and mouse all stubbed."""
    monkeypatch.setattr(engine_module, "ensure_model", lambda: None)
    monkeypatch.setattr(engine_module.cv2, "VideoCapture", lambda _index: StubCapture())
    monkeypatch.setattr(engine_module, "create_backend", lambda: NullBackend((1920, 1080)))

    class StubTracker:
        """A face landmarker that always reports the same, valid gaze."""

        def __init__(self, *_args, **_kwargs):
            self.frames = 0
            self.closed = False
            self._estimator = None

        def process(self, _frame, _timestamp_ms):
            self.frames += 1
            return GazeSample(yaw=5.0, pitch=2.0, distance=4200.0, valid=True, source="iris")

        def set_use_eyes(self, _value):
            return None

        def close(self):
            self.closed = True

    monkeypatch.setattr(engine_module, "FaceGazeTracker", StubTracker)
    return TrackingEngine(AppSettings())


def test_engine_reports_the_real_camera_resolution(engine):
    assert engine.camera_resolution == (1920, 1080)


def test_engine_step_moves_the_cursor_when_enabled(engine):
    engine.step(engine.read_frame(), enabled=True, now=100.0)
    assert engine.mouse.target() != (960.0, 540.0)


def test_engine_step_holds_still_when_disabled(engine):
    engine.step(engine.read_frame(), enabled=False, now=100.0)
    assert engine.mouse.backend.moves == []


def test_engine_tracks_fps(engine):
    for index in range(40):
        engine.step(engine.read_frame(), enabled=False, now=100.0 + index / 30.0)
    assert engine.fps > 20.0


def test_dwell_click_fires_once_after_the_cursor_settles(engine):
    """Staring at one spot clicks once, not once every dwell interval."""
    engine.settings.dwell_click = True
    engine.settings.dwell_s = 0.5

    for index in range(150):  # five seconds of a steady gaze
        engine.step(engine.read_frame(), enabled=True, now=100.0 + index / 30.0)

    assert engine.mouse.backend.clicks == ["left"], engine.mouse.backend.clicks


def test_dwell_click_does_not_fire_while_the_cursor_moves(engine):
    engine.settings.dwell_click = True
    engine.settings.dwell_s = 0.3

    # Sweep the gaze across the screen so the cursor is genuinely in motion
    # the whole time -- a target that saturates against the screen edge would
    # stop moving and legitimately trigger a dwell.
    for index in range(120):
        yaw = 12.0 * np.sin(index / 6.0)
        engine.controller.update(
            GazeSample(yaw=yaw, pitch=6.0 * np.cos(index / 6.0), distance=4200.0, valid=True),
            100.0 + index / 30.0,
        )
        engine._handle_dwell_click(100.0 + index / 30.0)  # pylint: disable=protected-access

    assert engine.mouse.backend.clicks == []


def test_dwell_click_is_off_by_default(engine):
    for index in range(60):
        engine.step(engine.read_frame(), enabled=True, now=100.0 + index / 30.0)
    assert engine.mouse.backend.clicks == []


def test_engine_close_releases_the_camera(engine):
    engine.close()
    assert engine.capture.released
    assert engine.tracker.closed


def test_cursor_settings_mirror_the_app_settings(engine):
    engine.settings.gain = 1.5
    engine.settings.min_cutoff = 0.42
    engine.settings.compensate_distance = False
    cursor = engine.cursor_settings()
    assert cursor.gain == 1.5
    assert cursor.min_cutoff == 0.42
    assert cursor.compensate_distance is False


def test_start_calibration_covers_the_screen(engine):
    session = engine.start_calibration(4, 4)
    assert session.total_points == 16
    assert session.screen == (1920.0, 1080.0)


# --------------------------------------------------------------------------
# Startup failures must be explained, not dumped as tracebacks
# --------------------------------------------------------------------------
def test_no_mouse_backend_reports_the_candidates(monkeypatch, capsys):
    def explode(*_args):
        raise RuntimeError("no usable mouse backend: X11Backend: no display")

    monkeypatch.setattr(engine_module, "ensure_model", lambda: None)
    monkeypatch.setattr(engine_module, "create_backend", explode)
    monkeypatch.setattr(AppSettings, "load", classmethod(lambda cls, _path: cls()))

    assert app_module.main(["--headless"]) == 1
    assert "no usable mouse backend" in capsys.readouterr().err


def test_a_missing_model_says_where_to_get_it(monkeypatch, capsys):
    def missing():
        raise FileNotFoundError(
            "download it from https://storage.googleapis.com/mediapipe-models/ "
            "and save it as /tmp/face_landmarker.task"
        )

    monkeypatch.setattr(engine_module, "ensure_model", missing)
    monkeypatch.setattr(engine_module, "create_backend", lambda: NullBackend((1920, 1080)))
    monkeypatch.setattr(AppSettings, "load", classmethod(lambda cls, _path: cls()))

    assert app_module.main(["--headless"]) == 1
    message = capsys.readouterr().err
    assert "mediapipe-models" in message
    assert "face_landmarker.task" in message


def test_no_warning_when_the_camera_delivers_what_was_asked(engine):
    assert engine.resolution_shortfall() is None


def test_a_camera_that_ignores_the_request_is_reported(monkeypatch):
    """A webcam that quietly refuses 1080p must not fail silently.

    Resolution is the biggest lever on accuracy -- measured single-frame gaze
    noise at 1 px of landmark jitter is 1.18 deg at 720p against 0.79 deg at
    1080p, amplified by roughly 37 px per degree on screen -- and CAP_PROP_*
    is only a request.  Without this the user sees a worse cursor and nothing
    to explain it.
    """
    class SmallCapture(StubCapture):
        """A webcam that quietly hands back 640x480 instead of 1080p."""

        def get(self, prop):
            import cv2  # pylint: disable=import-outside-toplevel

            return {
                cv2.CAP_PROP_FRAME_WIDTH: 640.0,
                cv2.CAP_PROP_FRAME_HEIGHT: 480.0,
            }.get(prop, 0.0)

        def read(self):
            return True, np.zeros((480, 640, 3), dtype=np.uint8)

    monkeypatch.setattr(engine_module, "ensure_model", lambda: None)
    monkeypatch.setattr(engine_module.cv2, "VideoCapture", lambda _index: SmallCapture())
    monkeypatch.setattr(engine_module, "create_backend", lambda: NullBackend((1920, 1080)))
    monkeypatch.setattr(engine_module, "FaceGazeTracker", type("T", (), {
        "__init__": lambda self, *a, **k: None,
        "process": lambda self, _f, _t: GazeSample(valid=False, reason="stub"),
        "set_use_eyes": lambda self, _v: None,
        "close": lambda self: None,
    }))
    engine = TrackingEngine(AppSettings())
    assert engine.resolution_shortfall() == ((1920, 1080), (640, 480))


# --------------------------------------------------------------------------
# Lighting diagnostic
# --------------------------------------------------------------------------
def test_well_lit_frame_has_no_complaint():
    frame = np.full((480, 640, 3), 120, dtype=np.uint8)
    frame[:, 320:] = 190  # give it contrast
    report = TrackingEngine.lighting(frame)
    assert report is not None
    assert report.problem is None


def test_a_dark_frame_is_reported():
    frame = np.full((480, 640, 3), 20, dtype=np.uint8)
    report = TrackingEngine.lighting(frame)
    assert report.problem is not None
    assert "too dark" in report.problem


def test_a_blown_out_frame_is_reported():
    frame = np.full((480, 640, 3), 240, dtype=np.uint8)
    report = TrackingEngine.lighting(frame)
    assert report.problem is not None
    assert "too bright" in report.problem


def test_a_flat_backlit_frame_is_reported():
    """Correct mean brightness but no contrast -- a face silhouetted by a window.

    This is the case that is hardest to notice: the preview looks fine, the
    brightness number looks fine, and the iris landmarks are still unreliable
    because the eyelid boundary has no edge to fit against.
    """
    frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    report = TrackingEngine.lighting(frame)
    assert report.problem is not None
    assert "too flat" in report.problem


def test_lighting_returns_none_for_an_empty_frame():
    assert TrackingEngine.lighting(np.zeros((0, 0, 3), dtype=np.uint8)) is None
