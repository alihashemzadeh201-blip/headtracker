"""Graphical front end and headless runner for HeadTracker."""

from __future__ import annotations

import argparse
import sys
import time
from typing import Optional

from .calibration import CalibrationModel
from .engine import TrackingEngine, check_wink
from .settings import CALIBRATION_PATH, SETTINGS_PATH, AppSettings

# ---------------------------------------------------------------------------
# Headless mode
# ---------------------------------------------------------------------------
def run_headless(settings: AppSettings, columns: int, rows: int) -> int:
    """Drive the cursor from the terminal, with no window at all."""
    model = CalibrationModel.load(CALIBRATION_PATH)
    try:
        engine = TrackingEngine(settings, model)
    except (FileNotFoundError, RuntimeError) as exc:
        # Missing model file, no camera, or no usable mouse backend.  All three
        # are setup problems the user can act on, so say so instead of dumping
        # a traceback.
        print(str(exc), file=sys.stderr)
        return 1
    print(f"camera: {engine.camera_resolution[0]}x{engine.camera_resolution[1]}")
    print(f"screen: {engine.mouse.screen}")
    state = "loaded" if model and model.is_fitted else "default (run with --calibrate)"
    print(f"calibration: {state}")

    try:
        session = None
        if columns:
            session = engine.start_calibration(columns, rows)
            session.start(time.monotonic())
            print(f"calibrating: look at each of {session.total_points} points")

        enabled = model is not None and model.is_fitted
        last_click = 0.0
        while True:
            frame = engine.read_frame()
            if frame is None:
                time.sleep(0.01)
                continue
            now = time.monotonic()
            sample = engine.step(frame, enabled and session is None, now)

            if session is not None:
                if sample.valid:
                    session.add_sample(sample.yaw, sample.pitch, sample.distance)
                if session.update(now):
                    engine.install_calibration(session.build(settings.calibration_degree))
                    print(f"calibrated: {engine.controller.model.report.describe()}")
                    session = None
                    enabled = True
                elif session.index:
                    print(f"  point {session.index + 1}/{session.total_points}")

            if enabled and sample.valid and check_wink(sample, settings):
                if now - last_click > settings.cooldown_s:
                    engine.mouse.click("left")
                    last_click = now

            print(
                f"\rfps {engine.fps:5.1f}  yaw {sample.yaw:+6.1f}  pitch {sample.pitch:+6.1f}  "
                f"src {sample.source:5s}  {'OK ' if sample.valid else sample.reason[:24]:24s}",
                end="",
                flush=True,
            )
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        engine.close()
    return 0


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------
def run_gui(settings: AppSettings) -> int:
    """Launch the graphical front end.

    The toolkit is imported here rather than at module scope so that the
    headless path -- and the test suite -- keep working on a machine with no
    display server and no customtkinter installed.
    """
    # pylint: disable-next=import-outside-toplevel
    from .gui import HeadTrackerApp

    app = HeadTrackerApp(settings)
    app.mainloop()
    return 0


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="headtracker",
        description="Move the mouse to the point on screen you are looking at.",
    )
    parser.add_argument("--headless", action="store_true", help="run without a window")
    parser.add_argument("--calibrate", action="store_true", help="run the calibration routine")
    parser.add_argument("--camera", type=int, default=None, help="camera index")
    parser.add_argument("--width", type=int, default=None, help="requested camera width")
    parser.add_argument("--height", type=int, default=None, help="requested camera height")
    parser.add_argument("--grid", type=int, default=4, help="calibration grid size (2-6)")
    return parser


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    settings = AppSettings.load(SETTINGS_PATH)
    if args.camera is not None:
        settings.camera_index = args.camera
    if args.width:
        settings.camera_width = args.width
    if args.height:
        settings.camera_height = args.height

    if args.headless or args.calibrate:
        grid = max(2, min(args.grid, 6)) if args.calibrate else 0
        return run_headless(settings, grid, grid)
    return run_gui(settings)


if __name__ == "__main__":
    sys.exit(main())
