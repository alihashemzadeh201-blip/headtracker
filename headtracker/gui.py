"""customtkinter front end: camera preview, calibration overlay and tuning.

Imported lazily by :func:`headtracker.app.run_gui` so the rest of the package
stays usable without a display server.
"""

from __future__ import annotations

import time
from typing import Optional

import cv2
import customtkinter as ctk
import numpy as np
from PIL import Image

from .calibration import CalibrationModel, CalibrationSession
from .engine import TrackingEngine, check_wink
from .geometry import GazeSample
from .settings import CALIBRATION_PATH, SETTINGS_PATH, AppSettings

PREVIEW_MAX_WIDTH = 760


class CalibrationOverlay(  # pylint: disable=too-few-public-methods
    ctk.CTkToplevel
):
    """Full-screen dot the user looks at while samples are collected."""

    def __init__(self, master, session, on_done) -> None:
        super().__init__(master)
        self.session = session
        self.on_done = on_done
        self.attributes("-fullscreen", True)
        self.attributes("-topmost", True)
        self.configure(fg_color="black")
        self.configure(cursor="none")
        self.canvas = ctk.CTkCanvas(self, bg="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.protocol("WM_DELETE_WINDOW", self.on_done)
        self.bind("<Escape>", lambda _event: self.on_done())

    def render(self, cancelled: bool = False) -> None:
        self.canvas.delete("all")
        if cancelled:
            return
        width = self.winfo_width()
        height = self.winfo_height()
        point = self.session.current_point()
        if point is None:
            return
        x, y = point[0] * width, point[1] * height
        progress = self.session.progress() if self.session.is_collecting() else 0.0

        self.canvas.create_oval(x - 34, y - 34, x + 34, y + 34, outline="#30363d", width=2)
        if progress > 0:
            self.canvas.create_arc(
                x - 34, y - 34, x + 34, y + 34, start=90, extent=-360 * progress,
                outline="#3fb950", width=4,
            )
        self.canvas.create_oval(x - 9, y - 9, x + 9, y + 9, fill="#e53935", outline="")
        self.canvas.create_text(
            width / 2, 30, fill="#8b949e",
            text=f"Look at the dot   {self.session.index + 1}/{self.session.total_points}"
                 "        (Esc to cancel)",
            font=("Helvetica", 16),
        )

class HeadTrackerApp(ctk.CTk):
    """Main window: camera preview, calibration and tuning."""

    def __init__(self, settings: AppSettings) -> None:
        super().__init__()
        self.settings = settings
        self.title("HeadTracker")
        self.geometry("1180x640")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.enabled = False
        self.session: Optional[CalibrationSession] = None
        self.overlay: Optional[CalibrationOverlay] = None
        self.last_click = 0.0
        self.is_winking = False

        self._build_ui()
        self._start_engine()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        for key in ("e", "E"):
            self.bind(f"<KeyPress-{key}>", lambda _event: self.toggle_enabled())
        self.after(10, self.loop)

    # -- construction ---------------------------------------------------
    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0, minsize=360)
        self.grid_rowconfigure(0, weight=1)

        self.preview_box = ctk.CTkFrame(self, corner_radius=10)
        self.preview_box.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.preview_label = ctk.CTkLabel(self.preview_box, text="Starting camera...")
        self.preview_label.pack(expand=True, fill="both", padx=5, pady=5)

        panel = ctk.CTkFrame(self, corner_radius=10)
        panel.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        self.toggle_button = ctk.CTkButton(
            panel, text="ENABLE MOUSE  (E)", height=42,
            fg_color="#C2185B", hover_color="#9C1448", command=self.toggle_enabled,
        )
        self.toggle_button.pack(fill="x", padx=16, pady=(16, 8))

        self.calibrate_button = ctk.CTkButton(
            panel, text="CALIBRATE  (16 points)", height=36,
            fg_color="#1565C0", hover_color="#0D47A1", command=self.begin_calibration,
        )
        self.calibrate_button.pack(fill="x", padx=16, pady=(0, 8))

        self.status_label = ctk.CTkLabel(panel, text="", justify="left", anchor="w")
        self.status_label.pack(fill="x", padx=16, pady=(0, 10))

        scroll = ctk.CTkScrollableFrame(panel, fg_color="transparent")
        scroll.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.scroll = scroll

        self.vars = {}
        self._slider(scroll, "gain", "Gain", 0.3, 3.0)
        self._slider(scroll, "min_cutoff", "Smoothing (lower = steadier)", 0.2, 3.0)
        self._slider(scroll, "beta", "Responsiveness", 0.0, 0.6)
        self._slider(scroll, "wink_close", "Wink close threshold", 0.08, 0.35)
        self._slider(scroll, "wink_open", "Wink open threshold", 0.12, 0.45)
        self._slider(scroll, "cooldown_s", "Click cooldown (s)", 0.1, 3.0)
        self._slider(scroll, "dwell_s", "Dwell time (s)", 0.2, 2.5)

        self.switches = {}
        for key, label in (
            ("wink_click", "Wink to click"),
            ("dwell_click", "Dwell to click"),
            ("use_eyes", "Use eye tracking (iris)"),
            ("compensate_distance", "Compensate for leaning"),
            ("mirror_preview", "Mirror preview"),
        ):
            variable = ctk.BooleanVar(value=getattr(self.settings, key))
            self.switches[key] = variable
            ctk.CTkSwitch(
                scroll, text=label, variable=variable, command=self._save_settings
            ).pack(anchor="w", pady=4)

    def _slider(self, parent, key, label, low, high) -> None:
        variable = ctk.DoubleVar(value=float(getattr(self.settings, key)))
        self.vars[key] = variable
        text = ctk.CTkLabel(parent, text=f"{label}: {variable.get():.2f}")
        text.pack(anchor="w", pady=(10, 0))

        def on_change(value, _label=label, _text=text, _key=key):
            _text.configure(text=f"{_label}: {float(value):.2f}")
            setattr(self.settings, _key, float(value))
            if self.engine is not None:
                self.engine.apply_settings()
            self.settings.save(SETTINGS_PATH)

        ctk.CTkSlider(parent, from_=low, to=high, variable=variable, command=on_change).pack(
            fill="x", pady=(0, 4)
        )

    def _start_engine(self) -> None:
        self.engine = None
        try:
            model = CalibrationModel.load(CALIBRATION_PATH)
            self.engine = TrackingEngine(self.settings, model)
            self.controller_model = model
        except Exception as exc:  # pylint: disable=broad-except
            self.preview_label.configure(
                text=f"Could not start:\n\n{exc}\n\n"
                     "Download face_landmarker.task from the MediaPipe model\n"
                     "repository and put it in ./assets/"
            )

    # -- actions --------------------------------------------------------
    def toggle_enabled(self) -> None:
        if self.engine is None:
            return
        self.enabled = not self.enabled
        if self.enabled:
            self.toggle_button.configure(
                text="DISABLE MOUSE  (E)", fg_color="#388E3C", hover_color="#2E7D32"
            )
            self.engine.controller.reset()
        else:
            self.toggle_button.configure(
                text="ENABLE MOUSE  (E)", fg_color="#C2185B", hover_color="#9C1448"
            )

    def begin_calibration(self) -> None:
        if self.engine is None:
            return
        self.enabled = False
        self.toggle_button.configure(text="CALIBRATING...", fg_color="#616161")
        self.withdraw()
        self.session = self.engine.start_calibration(4, 4)
        self.session.start(time.monotonic())
        self.overlay = CalibrationOverlay(self, self.session, self.cancel_calibration)
        self.overlay.after(60, self.overlay.render)

    def cancel_calibration(self) -> None:
        if self.overlay is not None:
            self.overlay.render(cancelled=True)
            self.overlay.destroy()
            self.overlay = None
        self.session = None
        self.deiconify()
        self.toggle_button.configure(
            text="ENABLE MOUSE  (E)", fg_color="#C2185B", hover_color="#9C1448"
        )

    def _save_settings(self) -> None:
        for key, variable in self.switches.items():
            setattr(self.settings, key, bool(variable.get()))
        if self.engine is not None:
            self.engine.tracker.set_use_eyes(self.settings.use_eyes)
        self.settings.save(SETTINGS_PATH)

    # -- main loop ------------------------------------------------------
    def loop(self) -> None:
        if self.engine is None:
            return
        frame = self.engine.read_frame()
        if frame is None:
            self.after(15, self.loop)
            return

        now = time.monotonic()
        active = self.enabled and self.session is None
        sample = self.engine.step(frame, active, now)

        if self.session is not None:
            if sample.valid:
                self.session.add_sample(sample.yaw, sample.pitch, sample.distance)
            if self.session.update(now):
                self._finish_calibration()
            elif self.overlay is not None:
                self.overlay.render()

        if active and sample.valid and check_wink(sample, self.settings):
            if not self.is_winking and now - self.last_click > self.settings.cooldown_s:
                self.engine.mouse.click("left")
                self.is_winking = True
                self.last_click = now
        elif not check_wink(sample, self.settings):
            self.is_winking = False

        self._draw(frame, sample)
        self.after(10, self.loop)

    def _finish_calibration(self) -> None:
        try:
            model = self.session.build(self.settings.calibration_degree)
        except ValueError:
            self.cancel_calibration()
            return
        self.engine.install_calibration(model)
        if self.overlay is not None:
            self.overlay.destroy()
            self.overlay = None
        self.session = None
        self.deiconify()
        self.toggle_button.configure(
            text="ENABLE MOUSE  (E)", fg_color="#C2185B", hover_color="#9C1448"
        )

    def _draw(  # pylint: disable=too-many-locals
        self, frame: np.ndarray, sample: GazeSample
    ) -> None:
        if self.settings.mirror_preview:
            frame = cv2.flip(frame, 1)
        height, width = frame.shape[:2]

        colour = (63, 185, 80) if sample.valid else (229, 57, 53)
        cv2.putText(
            frame,
            f"{sample.yaw:+.1f} / {sample.pitch:+.1f} deg   {sample.source}",
            (12, height - 44), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2,
        )
        cv2.putText(
            frame,
            f"{self.engine.fps:.0f} fps   {width}x{height}   "
            f"{self.engine.camera_resolution[0]}x{self.engine.camera_resolution[1]} cam",
            (12, height - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2,
        )

        box_width = self.preview_box.winfo_width()
        box_height = self.preview_box.winfo_height()
        if box_width < 20 or box_height < 20:
            return
        scale = min(box_width / width, box_height / height, PREVIEW_MAX_WIDTH / width)
        size = (max(int(width * scale), 1), max(int(height * scale), 1))
        resized = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        image = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        photo = ctk.CTkImage(light_image=image, dark_image=image, size=size)
        self.preview_label.configure(image=photo, text="")
        self.preview_label.image = photo  # keep a reference alive

        width_px, height_px = self.engine.mouse.screen
        model = self.engine.controller.model
        if model.is_fitted and sample.valid:
            x, y = model.predict_one(sample.yaw, sample.pitch)
            self.status_label.configure(
                text=(
                    f"gaze   {sample.yaw:+6.1f} deg  {sample.pitch:+6.1f} deg\n"
                    f"target {int(x):4d}, {int(y):4d}  of {width_px}x{height_px}\n"
                    f"source {sample.source}    eyes {sample.left_eye_open:.2f}/"
                    f"{sample.right_eye_open:.2f}\n"
                    f"calibration  {model.report.describe()}"
                )
            )
        else:
            reason = sample.reason or "tracking"
            self.status_label.configure(
                text=(
                    f"gaze   {sample.yaw:+6.1f} deg  {sample.pitch:+6.1f} deg\n"
                    f"status {reason}\n"
                    f"eyes   {sample.left_eye_open:.2f} / {sample.right_eye_open:.2f}\n"
                    "calibration  not done -- press CALIBRATE"
                )
            )

    def on_closing(self) -> None:
        self.settings.save(SETTINGS_PATH)
        if self.engine is not None:
            self.engine.close()
        self.destroy()
