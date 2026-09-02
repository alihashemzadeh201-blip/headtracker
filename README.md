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

Thirty points on a 6×5 grid, about 51 seconds. The fit is degree 2 in
`(yaw, pitch)`: enough to capture the pincushion of a webcam view, not so much
that it starts fitting the noise in the samples. Outlier rejection drops any
point where you blinked.

**The grid density is chosen for stability, not for the best case.** Measured
whole-screen cursor error with the eyelids modelled, over three independent
noise seeds:

| Grid | Seed A | Seed B | Seed C | Mean |
|---|---|---|---|---|
| 5×4 | 15.6 px | 25.3 px | 16.8 px | 19.2 px |
| **6×5** | 15.8 px | 15.8 px | 15.9 px | **15.8 px** |
| 7×6 | 15.2 px | 17.6 px | 14.9 px | 15.9 px |

7×6 has the best single draw and 6×5 has none of the bad ones. 5×4 is the one to
avoid: on one seed in three it lands at 25 px, so which calibration you get is a
lottery. 6×5 costs 17 extra seconds to remove that.

**The grid has to reach the edges of the screen.** The margin sets the domain
the fitted polynomial is valid over, and a degree-2 surface extrapolates badly
outside it. It used to sit at 8%, so the outermost calibration points were 8% in
from the border and everything beyond that was extrapolation. The screen edges
need yaw of −20.0° and +26.7°, but the fitted domain only spanned −18.6° to
+26.2° — so the corners were predicted *off the display entirely* (x = 1972 on
a 1920 px screen, y = −16). That is what "it doesn't cover the whole screen"
actually was.

Filtered cursor error, measured at 1920×1080 with 1 px of landmark jitter,
averaged over the frames the way the running app does:

| Region | margin 8%, 4×4 | margin 2%, 5×4 |
|---|---|---|
| Centre (15–85%) | 10.9 px | 12.5 px |
| **Whole screen** | **23.8 px** | **13.1 px** |
| Outer edge band | 35.5 px | **17.2 px** |

The centre gives up 1.6 px to gain 10.7 px across the whole screen and 18.3 px
at the edges, which is the trade worth making: the error used to grow 3.3× as
you moved away from the middle, and now it is close to uniform.

An earlier revision of this README quoted 3.2 px after calibration. That number
was measured only over the central 15–85% of the screen with noise-free
landmarks — it was true of the region it covered and said nothing about the
edges, which is exactly where the problem was.

Calibration is not the limiting factor. Decomposed at 1 px of landmark jitter:
the fit contributes 9.7 px, the total is 35.5 px, so **73% of the error is gaze
noise**, not the mapping.

### Pose solving

The head rotation comes from `cv2.solvePnP` over six landmarks. A single
`SOLVEPNP_ITERATIVE` call is not reliable enough to run on: over 300 frames with
1 px of landmark jitter it returned a mirrored, negative depth on 75 of them — a
quarter of the stream, which reaches you as the cursor stuttering. The solver
now tries SQPNP, EPNP and ITERATIVE in order and keeps the most consistent
answer. Rejection over those same 300 frames: **0**, depth standard deviation
29.9 units against a true 4348, no negative depths.

The reprojection bound that rejects a pose outright is deliberately generous
(60 px). It measures how well the landmarks fit a *canonical* face model, and a
real skull is not that model: perturbing just the six pose landmarks by 15 px —
far less than the difference between two people — already measures 13.7 px. A
tight bound silently discards real faces, every frame, and the cursor stops
moving. Landmarks that have collapsed onto each other are caught separately by a
spread guard, because they reproject at 0.00 px with a depth of 1.4e16.

### The eye centre

The iris offset is measured against the centre of the eyeball, taken as the mean
of eight points around the eye rim rather than the two corners. The corners lie
on one horizontal line, so their midpoint says nothing about where the eye sits
vertically. Single-frame gaze noise at 12° of eye yaw with 1 px of landmark
jitter: **1.18°** from the rim against **1.63°** from the corners.

### Eyelids, and a blind spot this project had

A landmark detector cannot report the part of the iris that is behind a lid — it
reports the visible boundary instead. The synthetic rig used to place the iris
**independently of the eyelids**, so the iris was never occluded and the rig was
structurally incapable of showing any error that comes from looking far enough
up or down for a lid to cut across the iris. That is a real gap, and it hid the
vertical problem for several revisions. `make_face(..., occlude_iris=False)`
still produces the old behaviour so the two can be compared.

What the corrected rig measures is a **compression** of the vertical signal, not
a one-sided failure. Gain of measured pitch against asked pitch:

| Asked pitch | Lids off | Lids on |
|---|---|---|
| Up, −18° to −4° | 1.043 | 0.877 |
| Middle, −4° to +4° | 1.000 | 1.000 |
| Down, +4° to +18° | 1.021 | 0.840 |
| *Yaw, −18° to +18° (control)* | 1.026 | *1.026* |

Both extremes lose about 15% of their authority, and the yaw control is
untouched — which is what confirms the lids are the cause, since nothing cuts
the iris horizontally.

It is worth recording that an earlier note here claimed the *lower* lid made
downward gaze roughly fifteen times worse than the upper lid made upward gaze.
That was wrong. The lid geometry is nearly symmetric (8.6 units of slack above
the eye centre against 6.6 below), and the raw errors looked asymmetric only
because the compression was sitting on top of the estimator's own downward
pitch bias. Separating the two is what the gain table above does.

### Smoothing

Raw gaze from a webcam jitters. The One Euro filter (Casiez *et al.*, CHI 2012)
resolves the trade-off that a fixed low-pass cannot: a low cut-off while you are
still, opening up as you move. Defaults `min_cutoff = 0.8`, `beta = 0.05` were
chosen by measurement at 30 fps with 1 px of landmark jitter:

- resting jitter 2.4° → **1.2°**
- a 14° glance reaches 90% in **~100 ms**

A single frame that teleports — a landmark tracker failure — is dropped by a
speed gate rather than flickering the cursor across the screen.

`beta` is unit-dependent and worth understanding before tuning it. The filter
opens at `min_cutoff + beta * |derivative|`, and in pixel space the derivative
of landmark noise is large, so a `beta` of 1.1 swamps any `min_cutoff`: measured
cut-off 90.6 Hz against 5.5 Hz for the shipped 0.05, which is no filtering at
all. Keep `beta` small.

These settings only take effect if they reach the filters. They did not: the
filters were constructed with the library defaults and re-tuned only when
`apply_settings` was called explicitly, so a controller handed a `settings.json`
profile smoothed with the wrong coefficients. The constructor now routes through
`apply_settings` as well, and
`tests/test_controller.py::test_a_supplied_profile_actually_smooths_more_than_a_snappy_one`
pins the behaviour rather than just the stored values — measured 9.3 px of
resting jitter for `(0.3, 0.05)` against 19.9 px for `(0.8, 2.0)`.

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
| 1280×720 | 1.18° |
| 1920×1080 | 0.79° |
| 2560×1440 | 0.60° |

1080p is requested by default. Also: good light on your face, and sit close
enough that your face fills a decent part of the frame.

**Check that you actually got it.** `CAP_PROP_FRAME_WIDTH` is a request, and
many webcams quietly hand back something smaller. Both front ends now say so:
the headless runner prints a warning to stderr, and the preview window draws one
over the camera image. A camera stuck at 640×480 costs tens of pixels of
accuracy and is otherwise invisible.

There is a ceiling here, and it is worth knowing before buying a better camera
or trying to crop in on the eyes.

MediaPipe's iris refinement crops **twice the eye width** and resizes that to
**64×64**, whatever the input resolution; FaceMesh-V2 works at 256×256. So the
question is not "how big is the eye" but "is 2× the eye width already past 64?"
Measured on a synthetic face at a typical 600 mm:

| Setup | Eye width | Crop (2×) | Against the 64×64 input |
|---|---|---|---|
| 1280×720 @600 mm | 36.2 px | 72.5 px | above — downsampling |
| 1920×1080 @600 mm | 54.4 px | 108.7 px | above — downsampling |
| 2560×1440 @600 mm | 72.5 px | 145.0 px | above — downsampling |
| 1920×1080 @900 mm | 36.6 px | 73.3 px | above — downsampling |

**Zooming in on the eyes does not help.** In every ordinary setup the crop is
already larger than the model's 64×64 input, so MediaPipe is throwing detail
away, not starved for it. Cropping and upscaling adds no information the model
did not already discard. Moving closer or using a higher-resolution camera does
add real pixels — but only up to the point where 2× eye width reaches 64, which
even 720p clears. That is why the resolution table above flattens out
(1.18 → 0.79 → 0.60°) instead of scaling with pixel count.

### Lighting

Light is the other common reason a tracker is worse than its numbers, and it is
the one thing here that can be checked in seconds. Both front ends now measure
each frame and say so if the light is unusable:

- mean brightness below 55/255 — too dark
- mean brightness above 205/255 — blown out
- standard deviation below 32 — too flat, usually backlit (a face silhouetted
  against a window). This is the sneaky one: the preview looks fine and the
  brightness number looks fine, but the eyelid and iris boundary has no edge
  left for the landmarks to fit against.

Put the light in front of you, not behind.

Note that lighting is the one factor this project's test rig **cannot** measure:
the rig projects landmarks directly and never renders an image, so the
thresholds above are conventional rather than fitted.

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
tests/            172 tests, including a synthetic-face rig
```

The whole control path runs without a display server, so it is tested directly:
`tests/synthetic_face.py` builds a 3-D face, poses it at a known gaze, projects
it through a pinhole camera, and hands the pixels to the real estimator. The
accuracy numbers above come from that rig, not from hand-tuned assertions.

```bash
pytest          # 172 tests
pylint $(git ls-files '*.py')
```
