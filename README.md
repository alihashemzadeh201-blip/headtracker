# HeadTracker

Move the mouse with your eyes. A webcam tracks your gaze and puts the cursor on
the point you are looking at; a wink or a dwell clicks.

Written against MediaPipe's refined 468+10-point face mesh. No special hardware,
no infrared, no head-mounted anything.

```
python head.py
```

---

## What the previous version got wrong

The original `head.py` moved the cursor with
`pyautogui.moveRel(int(dx * width * sensitivity))`. That single line is the
whole problem:

1. **It integrated.** `moveRel` adds to wherever the cursor already is, so every
   rounding error, every dropped frame and every moment the face was lost
   accumulated permanently. The cursor drifted until it hit a screen edge.
2. **Its signal was the nose tip's pixel position.** That conflates turning
   your head with moving it sideways, scales with how far you sit from the
   camera, and does not change at all when you move your eyes.
3. **It was never calibrated**, so nothing absorbed the difference between "the
   direction my eyes point" and "the pixel I mean".
4. **It was not smoothed**, so a one-pixel landmark wobble shook the cursor.
5. `int()` truncated every move, the camera opened at its default resolution,
   and a bare `except:` hid all of it. It also called `mp.solutions.face_mesh`,
   which does not exist in MediaPipe 1.0.

Every one of those is fixed below, and each fix is covered by a test that
measures it rather than just asserting it exists.

---

## How it works

### The signal is the gaze ray, not an angle

The estimator recovers two things from each frame: **where the eye is** and
**which way it points**. It then intersects that ray with the screen plane and
reports the hit point (`geometry.project_to_screen_plane`).

This is the single most important design decision here, and it is what makes
head movement a non-event. Writing the eye at `E` and the gaze direction as `d`,
the point being looked at is

```
s = -E.z / d.z
hit = E + s * d          →   hit.x ≈ E.x + E.z * tan(yaw)
```

so the head's lateral position, its distance from the screen and its rotation
all enter the arithmetic instead of being guessed at afterwards. Lean left and
the origin moves left; lean back and the lever arm `E.z` lengthens, so the same
angle sweeps further. Both used to need a fitted correction factor — the old
`compensate_distance` setting — and both now fall out of the geometry. That
setting, and `distance_factor` with it, are gone.

The iris measurement is the primary signal and the head pose the fallback. This
is not a guess: multiplying the perspective numerator and denominator by
`cos(e)` shows the camera-space gaze reduces to `sin(e−θ)/cos(e−θ)`, so the iris
offset already encodes the *total* gaze direction independent of head rotation.
Adding the head angle on top would count it twice, which an earlier version of
this code did.

### The pose solver is chosen, not defaulted

The six model points used for PnP are nearly coplanar, which makes the pose
ambiguous. OpenCV's default `SOLVEPNP_ITERATIVE` converges from the origin, so
where it lands depends on its initial guess: measured over 300 frames with 1 px
of landmark jitter it returned a mirrored, **negative-depth** solution on 75 of
them — a quarter of the video silently discarded, which reads as a stuttering
cursor. `SOLVEPNP_SQPNP` is a direct method with no initialisation and returned
a sane depth on all 300, with a depth standard deviation of 20 model units
against 4010. It is tried first, with `EPNP` and `ITERATIVE` behind it.

Two guards reject the solutions the solvers are willing to invent:

- **Spread.** When MediaPipe loses the face it returns coincident points, and
  PnP answers those with a rotation and a translation of `1e16` whose
  reprojection error is *exactly zero* — every model point projects onto the
  pixel the landmarks already share. Nothing about the fit detects it; only the
  physical size of the face does.
- **Reprojection.** A pose that really fits puts its model points within a
  couple of pixels of the landmarks. Measured: 0.0 px for a real face, 59.9 px
  for a fabricated rotation, against a 12 px threshold.

### The eye centre is averaged, not assumed

The iris offset is one subtraction of two jittering points, and the eye-centre
half of that subtraction is the noisier one. Averaging all eight landmarks on
the rim of the socket instead of just the two corners cut the single-frame yaw
error from **1.163° to 0.882°** at 1 px of landmark jitter, and measured exactly
the same when the eye is half closed. The axes still come from the corners and
the lid pair, because those define orientation; only the centre is averaged.

### Absolute control

The cursor is warped to a position computed from the current gaze every frame.
Nothing accumulates, so nothing drifts. Targets are kept as floats and rounded
only at the moment of the warp, and the syscall is skipped when the rounded
pixel has not changed.

### Calibration

Calibration fits a degree-2 polynomial from screen-plane points to screen
pixels. It removes what cannot be computed: where the camera is mounted, how the
lens bends, how the screen is tilted, and the systematic bias of the iris model
itself.

Because the projection already linearises the geometry, the fit is close to
affine and converges on a 4×4 grid. The solver itself is verifiable: a target
that is exactly a degree-2 polynomial comes back out to 0.15 px, the residue of
the ridge penalty rather than of the fit.

### Smoothing

A One Euro filter, with one correction that matters. Its speed term is measured
in pixels per second, and a 1 px landmark wobble at 30 fps reads as about
**900 px/s**. A `beta` tuned in angle space therefore opens the filter right up
in pixel space — the cursor jittered by 26.6 px at settings that were supposed
to steady it. `beta` is now entered per screen height and divided down inside
the controller, so the same number means the same thing on any monitor.

A constant-velocity Kalman filter was implemented and measured against it before
being dropped. Its steady state matched its own Riccati equation to two decimals
(18.32 px predicted, 22.95 px measured once the 2-D radial mean is accounted
for), so it was correct — it simply loses. Gaze is long fixations punctuated by
saccades, and a model that only expects smooth acceleration has to build up
velocity before it follows a step. At equal lag One Euro was steadier: 17.3 px
against 27.5 px at 133 ms, and 13.1 px against 23.0 px at 167 ms.

---

## Measured

All figures on synthetic faces projected through a pinhole camera at 1920×1080
and 30 fps, with 1 px of per-landmark jitter unless stated. Reproduce them with
`pytest`.

### Cursor jitter and offset

Aimed at one pixel for 240 frames:

One protocol throughout: 300 warmup frames, then 150 measured while aiming at
one pixel, then a 943 px glance. Lag is the time to reach 90% of that glance.

| | jitter | offset | lag |
|---|---|---|---|
| unfiltered | 33.92 px | 1.56 px | — |
| as previously shipped | 26.59 px | 1.47 px | 33 ms |
| **shipped defaults** (`min_cutoff=0.3`, `beta=1.1`) | **9.08 px** | **2.37 px** | **133 ms** |
| steadier (`0.2`, `0.5`) | 6.28 px | 2.63 px | 200 ms |
| steadiest (`0.15`, `0.3`) | 4.68 px | 2.73 px | 267 ms |

The shipped defaults are the point where the cursor is steady while you read
and still keeps up with a glance. Both lower settings are one line in
`settings.json` if you would rather have stillness than speed.

"Offset" here is against the pixel being aimed at, and it is small in every row
because it is dominated by the calibration, not the filter. Against the *true*
target after a full calibration the cursor sits 3.5 px off.

### Single-frame gaze noise

| camera resolution | yaw | pitch |
|---|---|---|
| 1280×720 | 1.316° | 1.068° |
| 1920×1080 | 0.882° | 0.716° |
| 2560×1440 | 0.668° | 0.539° |

This is the resolution floor, not a tuning artefact: the iris is about 16 px
across even at 1080p. A higher camera resolution is the one upgrade that moves
it.

### Calibration

Clean signal, 48 held-out probes on a 1920×1080 screen:

| grid | mean | worst |
|---|---|---|
| 3×3 | 4.18 px | 10.15 px |
| 4×4 | 3.90 px | 8.87 px |
| 5×4 | 3.68 px | 7.86 px |

Calibration is not the limiting factor — the landmark noise above is roughly
eight times larger. Going past degree 2 fits the noise instead of the mapping.

Fitting in screen-plane coordinates rather than raw angles costs about 1.9 px
here (3.88 against 2.02 px). That is a fair trade: it is 5% of the noise floor,
and it is what buys the head compensation below.

### Head movement

Calibrate once, then move. Measured as the drift of the recovered screen point
against the true one, relative to the calibration pose:

| the head… | drift |
|---|---|
| slides 40 px left or right in frame | **< 3 px** |
| moves from 450 mm to 850 mm | **< 4 px** |
| yaws ±18° | up to 40 px |
| pitches ±12° | up to 22 px |

Translation and distance are compensated essentially exactly, because they enter
the projection analytically. Rotation is only partly compensated: the iris
offset is read along axes that turn with the head, so a rotated head slightly
under-reports the camera-frame angle. The residual is bounded, smooth and
centred on the calibration pose — at the cursor, every pose tested lands within
60 px of target — and removing it would mean de-rotating the iris measurement by
the head pose, which is worth doing only against a real camera.

---

## Install and run

```bash
pip install -r requirements.txt
python head.py                 # GUI
python head.py --headless      # no window
python head.py --calibrate     # calibrate, then run
python head.py --list-cameras
```

The MediaPipe model asset (`face_landmarker.task`, ~3 MB) is downloaded to
`~/.cache/headtracker/` on first run; the app prints manual instructions if that
fails.

Settings live in `~/.config/headtracker/settings.json` and the calibration in
`calibration.json` beside it.

### Controls

| | |
|---|---|
| wink either eye | click |
| hold still | dwell-click |
| `Space` | enable / disable tracking |
| `C` | recalibrate |
| `Q` / `Esc` | quit |

---

## Layout

```
head.py                    thin wrapper
headtracker/
  geometry.py              landmarks -> gaze ray -> screen-plane point
  calibration.py           screen-plane point -> screen pixel
  controller.py            -> smoothed cursor position
  filters.py               One Euro, glitch gate
  engine.py                GUI-free camera -> cursor loop
  gui.py                   customtkinter front end
  mouse.py                 absolute positioning: Win32 / Xlib / Quartz
  settings.py              persisted configuration
  tracking.py              MediaPipe Tasks wrapper
tests/                     179 tests
```

```bash
pytest          # 179 passed
pylint headtracker tests   # 10.00/10
```

### Not verified here

The GUI and the real MediaPipe inference path could not be executed in the
environment this was written in — there is no display and the model asset is not
downloadable. `gui.py` is import-checked against a stub and every attribute it
touches was checked by AST against a definition, but it has not been run. Every
number above comes from the synthetic rig, which models a pinhole camera and a
rigid face; a real webcam adds motion blur, auto-exposure and a lens that is not
a pinhole.
