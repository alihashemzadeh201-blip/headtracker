# HeadTracker

Look at a point on the screen and the cursor goes there.

The cursor is driven by **where your eyes are pointing**, recovered from an
ordinary webcam: MediaPipe locates 478 face landmarks per frame, the iris
landmarks give the direction each eye is aimed, and a one-time calibration
turns that direction into a pixel on your monitor.

```
python head.py            # graphical window
python head.py --headless --calibrate   # no window, straight to calibration
```

---

## Why this version is accurate where the previous one was not

The original controller read the **pixel position of the nose tip** and fed it
to `pyautogui.moveRel`. Three properties of that design make "exactly where I
look" unreachable, and all three are fixed here.

| Problem | Consequence | Fix |
|---|---|---|
| The signal is a pixel position | It changes when you lean, even if your gaze never moves | The signal is now a **gaze angle in degrees** ([`geometry.py`](headtracker/geometry.py)) |
| Control is *relative* (`moveRel`) | Every frame's error accumulates; the cursor drifts and never returns to the same pixel | Control is **absolute** ([`mouse.py`](headtracker/mouse.py)) |
| No calibration | Nothing maps your face, your camera and your monitor onto each other | A **16-point polynomial fit** ([`calibration.py`](headtracker/calibration.py)) |

### The signal: gaze angle, not pixel position

The iris sits on the surface of the eyeball at the point the gaze exits, so its
offset from the socket centre measures the looking direction itself — in the
camera's frame, whatever the head is doing. It is normalised by the iris radius
(a rigid disc, so the one length on the eye that does not change when you move,
blink or squint) and converted with an arcsine, which saturates naturally at
the physiological limit instead of extrapolating past it.

Because it is an angle, distance drops out. Measured on a synthetic face with a
known 15° gaze:

| Distance from camera | Recovered gaze |
|---|---|
| 400 mm | 15.30° |
| 600 mm | 15.28° |
| 900 mm | 15.27° |

The same gaze expressed as a raw iris *pixel* position moves by more than 20 px
across that range. Sitting back no longer moves your cursor.

The head pose is still computed, with `cv2.solvePnP`, but as a **fallback** for
when the irises are not usable — not as something to add. Adding the two would
double-count: the iris already encodes the total gaze.

### Absolute control

`moveRel` integrates every frame's rounding error, so a cursor that has been
running for ten minutes is nowhere in particular. Recomputing the target from
the gaze each frame means a bad frame can be off by a pixel but can never
accumulate — returning your gaze to where it was returns the cursor to the same
pixel. `tests/test_controller.py::test_no_drift_under_repeated_identical_frames`
pins this down over 900 frames.

### Calibration

Sixteen points, about 27 seconds. The fit is degree 2 in `(yaw, pitch)`: enough
to capture the pincushion of a webcam view, not so much that it starts fitting
the noise in the samples. Outlier rejection drops any point where you blinked.

Measured on a simulated rig with a pinhole projection and a camera mounted
off-axis, at 1920×1080:

| | mean error | worst error |
|---|---|---|
| No calibration | 300 px | 559 px |
| After calibration | **3.2 px** | **5.8 px** |

Calibration is not the limiting factor — the gaze measurement is.

### Smoothing

Raw gaze from a webcam jitters. The One Euro filter (Casiez *et al.*, CHI 2012)
resolves the trade-off that a fixed low-pass cannot: a low cut-off while you are
still, opening up as you move. Defaults `min_cutoff = 0.8`, `beta = 0.05` were
chosen by measurement at 30 fps with 1 px of landmark jitter:

- resting jitter 2.4° → **1.2°**
- a 14° glance reaches 90% in **~100 ms**

A single frame that teleports — a landmark tracker failure — is dropped by a
speed gate rather than flickering the cursor across the screen.

---

## Accuracy you should expect

This is a webcam, not an infrared eye tracker. Honest numbers from
`tests/test_end_to_end.py`, at 1920×1080 on a 1920×1080 screen:

| Condition | Cursor lands | Residual wobble |
|---|---|---|
| Noise-free landmarks | 4 px from target | 0 px |
| 1 px landmark jitter | 10 px from target | 37 px |

The wobble is the resolution floor: the iris is only ~16 px across even at
1080p, so one pixel of landmark jitter *is* about a degree of gaze. The single
biggest lever is camera resolution — single-frame gaze error under 1 px of
landmark jitter:

| Resolution | Gaze error |
|---|---|
| 1280×720 | 2.25° |
| 1920×1080 | 1.52° |
| 2560×1440 | 1.15° |

1080p is requested by default. Also: good light on your face, and sit close
enough that your face fills a decent part of the frame.

---

## Install and run

```bash
pip install -r requirements.txt
python head.py
```

The MediaPipe model (`face_landmarker.task`) is downloaded automatically on
first run. If your network blocks it, the app prints the URL and the exact path
to save it to.

> **MediaPipe 1.0 removed `mp.solutions`.** A fresh `pip install mediapipe`
> breaks the original `head.py` with
> `AttributeError: module 'mediapipe' has no attribute 'solutions'`.
> This version uses the replacement Tasks API (`FaceLandmarker`) and falls back
> to the legacy solver if an older MediaPipe is installed.

### Controls

| Action | Effect |
|---|---|
| `E` | Enable / disable the cursor |
| **CALIBRATE** | Run the 16-point routine |
| Wink (one eye shut) | Left click |
| Dwell (optional) | Click after holding still |
| `Esc` | Cancel calibration |

Settings and calibration persist in `~/.config/headtracker/`.

---

## Layout

```
headtracker/
  geometry.py     landmarks -> gaze angles (solvePnP + iris)
  calibration.py  angles -> pixels (polynomial fit, collection session)
  filters.py      One Euro filter, glitch gate
  controller.py   gaze samples -> absolute cursor targets
  mouse.py        cross-platform absolute positioning (Win32 / X11 / Quartz)
  tracking.py     MediaPipe FaceLandmarker wrapper
  engine.py       camera -> gaze -> cursor, no GUI
  app.py          CLI
  gui.py          customtkinter window
tests/            155 tests, including a synthetic-face rig
```

The whole control path runs without a display server, so it is tested directly:
`tests/synthetic_face.py` builds a 3-D face, poses it at a known gaze, projects
it through a pinhole camera, and hands the pixels to the real estimator. The
accuracy numbers above come from that rig, not from hand-tuned assertions.

```bash
pytest          # 155 tests
pylint $(git ls-files '*.py')
```
