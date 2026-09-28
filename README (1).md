# Hand Gesture Mouse Control

Control your mouse using hand gestures via webcam — built with OpenCV, MediaPipe (Tasks API), PyAutoGUI, and NumPy.

## Setup

```bash
pip install -r requirements.txt
python gesture_mouse.py
```

On first run the script downloads the ~7 MB `hand_landmarker.task` model next to
the script (internet required once). If the download fails, the error message
gives the URL so you can save the file manually.

**Quit:** press `q` or `ESC` in the video window, **or** move your physical mouse
into any screen corner (PyAutoGUI failsafe). Note: once the cursor clicks into
another app, the video window loses keyboard focus, so the corner failsafe is the
most reliable emergency stop.

> **macOS users:** grant your terminal/IDE both **Camera** and **Accessibility**
> permissions (System Settings → Privacy & Security), or PyAutoGUI can't move
> the mouse.

> **Python version:** use a Python version supported by your installed MediaPipe
> release (check its PyPI page); very new Python versions may not have wheels yet.

## Gestures

| Gesture | Action |
|---|---|
| Index finger up only | Move cursor |
| Thumb + Index pinch | Left click (fires once per pinch) |
| Thumb + Middle pinch | Right click (fires once per pinch) |
| Index + Middle fingers up | Scroll mode (move hand up/down) |
| Fist / no clear gesture | Idle |

## How it works

1. **Hand tracking** — MediaPipe's `HandLandmarker` returns 21 landmarks per hand
   per frame. `fingers_up()` compares each fingertip's y-position (x-position for
   the thumb) against its lower joint to decide if it's extended.

2. **Coordinate mapping** — Your hand only needs to move within a smaller
   "active zone" inside the camera frame (set by `FRAME_REDUCTION`) to reach
   the whole screen. `np.interp` maps camera-space to screen-space using the
   camera's *actual* frame size.

3. **Exponential smoothing** — `Smoother` blends each new point with the previous
   smoothed point (`smoothed = prev + (new - prev) * alpha`). Lower
   `SMOOTHING_FACTOR` = smoother but laggier; higher = snappier but noisier.

4. **Gesture priority** — Each frame: **fist (idle) → pinch (click) → two-finger
   (scroll) → index-only (move)**. Switching gesture types resets the smoother /
   scroll reference so leftover motion state doesn't leak between modes.

5. **Scale-independent, edge-triggered clicks** — Pinch distance is divided by
   hand size (wrist to middle knuckle), so it works at any distance from the
   camera. A click fires once when the pinch *starts*; hysteresis
   (`PINCH_ON_RATIO` < `PINCH_OFF_RATIO`) stops flicker, and `CLICK_COOLDOWN`
   adds a final debounce.

6. **Continuous scroll** — in scroll mode, the vertical distance between the
   current hand position and a rolling reference point becomes scroll ticks each
   frame, then the reference resets.

## Tuning

All constants are at the top of `gesture_mouse.py`:

- `FRAME_REDUCTION` — shrink/grow the "active zone" your hand needs to cover
- `SMOOTHING_FACTOR` — cursor smoothness vs. responsiveness
- `PINCH_ON_RATIO` / `PINCH_OFF_RATIO` — how close thumb/finger must get to start / end a click
- `CLICK_COOLDOWN` — minimum seconds between clicks
- `SCROLL_SENSITIVITY` / `SCROLL_DEADZONE` — scroll speed and jitter tolerance

## Troubleshooting

- **Cursor feels laggy:** check the FPS overlay; improve lighting, close other
  camera apps, or lower `CAM_WIDTH`/`CAM_HEIGHT`.
- **Clicks land slightly off target:** the index tip moves a little as your thumb
  approaches. Pinch with a quick, small motion, or raise `SMOOTHING_FACTOR` less.
- **Camera not found:** change `cv2.VideoCapture(0)` to `1` for a second webcam.

## Possible extensions

- Drag-and-drop: `mouseDown` on pinch start, `mouseUp` on pinch end (the pinch
  state tracking already exists in `handle_click`).
- Zoom gesture: two-hand pinch distance mapped to `Ctrl` + scroll.
- Volume control gesture using the distance between thumb and pinky.
