"""
Hand Gesture Mouse Control
---------------------------
Control your mouse cursor using hand gestures captured from a webcam.

Gestures:
    - Index finger up only          -> Move cursor
    - Thumb + Index pinch           -> Left click
    - Thumb + Middle pinch          -> Right click
    - Index + Middle fingers up     -> Scroll mode (move hand up/down to scroll)
    - Fist (no fingers up)          -> Idle / no action

Tech stack: OpenCV, MediaPipe (Tasks API - HandLandmarker), PyAutoGUI, NumPy

NOTE: Recent MediaPipe releases dropped the legacy `mp.solutions.hands` API.
This version uses the supported `mediapipe.tasks` HandLandmarker API.

Safety: PyAutoGUI's failsafe is ON. Slam your *physical* mouse into any screen
corner (or press 'q' / ESC in the video window) to stop the program.

Author: Sumit Chand
"""

import os
import time
import math
import urllib.request
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
import pyautogui

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

CAM_WIDTH, CAM_HEIGHT = 640, 480          # requested webcam resolution (camera may ignore it;
                                          # the code always uses the real frame size)
FRAME_REDUCTION = 100                     # margin (px) inside the camera frame that maps to
                                          # the full screen
SMOOTHING_FACTOR = 0.35                   # exponential smoothing weight (0-1);
                                          # lower = smoother but laggier
CLICK_COOLDOWN = 0.4                      # min seconds between clicks of the same type

# Pinch detection is relative to hand size (wrist -> middle-finger knuckle),
# so it works the same whether your hand is near or far from the camera.
PINCH_ON_RATIO = 0.25                     # fingertip distance / hand size below which a pinch STARTS
PINCH_OFF_RATIO = 0.35                    # ...and above which it ENDS (hysteresis stops flicker)

SCROLL_SENSITIVITY = 4                    # bigger = slower scroll
SCROLL_DEADZONE = 8                       # pixels of hand movement ignored to avoid jitter-scroll
SCREEN_EDGE_MARGIN = 5                    # keep cursor this many px away from screen edges so
                                          # PyAutoGUI's corner failsafe is not triggered by gestures

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

# FIX: PyAutoGUI sleeps 0.1 s after EVERY call by default, which capped the
# whole app at ~10 FPS and made the cursor feel laggy. Disable that delay.
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = True                 # FIX: keep the emergency stop enabled
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

# Standard 21-point hand skeleton connections (used only for drawing).
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                 # palm base
]


def ensure_model_downloaded():
    """Download the hand landmark model once, if it isn't already local."""
    if os.path.exists(MODEL_PATH):
        return
    print("Downloading hand landmark model (one-time, ~7 MB)...")
    tmp_path = MODEL_PATH + ".part"
    try:
        urllib.request.urlretrieve(MODEL_URL, tmp_path)
        os.replace(tmp_path, MODEL_PATH)      # only keep a fully-downloaded file
    except Exception as exc:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise SystemExit(
            f"Could not download the model: {exc}\n"
            f"Download it manually from:\n  {MODEL_URL}\n"
            f"and save it as:\n  {MODEL_PATH}"
        )
    print(f"Model saved to {MODEL_PATH}")


# ----------------------------------------------------------------------
# Helper: exponential smoother for (x, y) points
# ----------------------------------------------------------------------

class Smoother:
    """Simple exponential moving average smoother for 2D points."""

    def __init__(self, alpha=SMOOTHING_FACTOR):
        self.alpha = alpha
        self.prev = None

    def update(self, point):
        if self.prev is None:
            self.prev = point
            return point
        smoothed = (
            self.prev[0] + (point[0] - self.prev[0]) * self.alpha,
            self.prev[1] + (point[1] - self.prev[1]) * self.alpha,
        )
        self.prev = smoothed
        return smoothed

    def reset(self):
        self.prev = None


# ----------------------------------------------------------------------
# Hand landmark helper functions
# ----------------------------------------------------------------------

def fingers_up(landmarks, handedness_label):
    """
    landmarks: list of 21 NormalizedLandmark objects (x, y, z each 0-1).
    Returns [thumb, index, middle, ring, pinky] booleans (True = extended).
    """
    fingers = []

    # Thumb: compare x of tip vs the joint below it (direction depends on hand).
    if handedness_label == "Right":
        fingers.append(landmarks[4].x < landmarks[3].x)
    else:
        fingers.append(landmarks[4].x > landmarks[3].x)

    # Other 4 fingers: tip is "up" if it's above (smaller y) the PIP joint
    for tip_id in [8, 12, 16, 20]:
        fingers.append(landmarks[tip_id].y < landmarks[tip_id - 2].y)

    return fingers


def is_fist(landmarks):
    """
    True when all four fingers are curled tightly (tips below their knuckles).
    Stricter than fingers_up() so a relaxed pinch is not mistaken for a fist.
    """
    return all(landmarks[t].y > landmarks[t - 3].y for t in (8, 12, 16, 20))


def pixel_distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def landmark_to_pixel(landmark, w, h):
    return int(landmark.x * w), int(landmark.y * h)


def draw_hand(frame, landmarks_px):
    """Draw skeleton dots + connecting lines using plain OpenCV calls."""
    for (a, b) in HAND_CONNECTIONS:
        cv2.line(frame, landmarks_px[a], landmarks_px[b], (0, 200, 0), 2)
    for point in landmarks_px:
        cv2.circle(frame, point, 4, (0, 0, 255), cv2.FILLED)


# ----------------------------------------------------------------------
# Main application
# ----------------------------------------------------------------------

class GestureMouseController:
    def __init__(self):
        ensure_model_downloaded()

        self.cap = cv2.VideoCapture(0)
        # FIX: fail clearly if the webcam can't be opened
        if not self.cap.isOpened():
            raise SystemExit(
                "Could not open the webcam (index 0). Check that it is connected, "
                "not used by another app, and that camera permission is granted."
            )
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            running_mode=RunningMode.VIDEO,
        )
        self.landmarker = HandLandmarker.create_from_options(options)
        self._start_time = time.time()
        self._last_ts_ms = -1

        self.smoother = Smoother()
        self.last_left_click_time = 0.0
        self.last_right_click_time = 0.0
        self.left_pinched = False
        self.right_pinched = False
        self.scroll_ref_y = None
        self.fps_history = deque(maxlen=30)
        self.status_text = "Idle"

    # ------------------------------------------------------------------
    def next_timestamp_ms(self):
        """detect_for_video requires strictly increasing timestamps."""
        ts = int((time.time() - self._start_time) * 1000)
        if ts <= self._last_ts_ms:
            ts = self._last_ts_ms + 1
        self._last_ts_ms = ts
        return ts

    # ------------------------------------------------------------------
    def release_pinches(self):
        self.left_pinched = False
        self.right_pinched = False

    # ------------------------------------------------------------------
    def map_to_screen(self, x, y, w, h):
        """Map a point in the reduced camera frame to full screen space.
        FIX: uses the real frame size (w, h), not the requested one."""
        x = np.interp(x, (FRAME_REDUCTION, w - FRAME_REDUCTION), (0, SCREEN_WIDTH))
        y = np.interp(y, (FRAME_REDUCTION, h - FRAME_REDUCTION), (0, SCREEN_HEIGHT))
        return x, y

    # ------------------------------------------------------------------
    def handle_move(self, index_tip_px, w, h):
        screen_x, screen_y = self.map_to_screen(*index_tip_px, w, h)
        smooth_x, smooth_y = self.smoother.update((screen_x, screen_y))
        # Stay off the exact screen corners so the failsafe isn't tripped by gestures
        smooth_x = min(max(smooth_x, SCREEN_EDGE_MARGIN), SCREEN_WIDTH - 1 - SCREEN_EDGE_MARGIN)
        smooth_y = min(max(smooth_y, SCREEN_EDGE_MARGIN), SCREEN_HEIGHT - 1 - SCREEN_EDGE_MARGIN)
        pyautogui.moveTo(smooth_x, smooth_y)
        self.status_text = "Moving"

    # ------------------------------------------------------------------
    def handle_click(self, thumb_px, index_px, middle_px, hand_size):
        """
        Edge-triggered clicking with hysteresis:
        a click fires once when a pinch STARTS, not repeatedly while it is held.
        Returns True while any pinch is active.
        """
        now = time.time()
        ratio_left = pixel_distance(thumb_px, index_px) / hand_size
        ratio_right = pixel_distance(thumb_px, middle_px) / hand_size

        # Hysteresis: easier to stay pinched than to start a pinch
        left_now = ratio_left < (PINCH_OFF_RATIO if self.left_pinched else PINCH_ON_RATIO)
        right_now = (not left_now) and \
            ratio_right < (PINCH_OFF_RATIO if self.right_pinched else PINCH_ON_RATIO)

        if left_now and not self.left_pinched and now - self.last_left_click_time > CLICK_COOLDOWN:
            pyautogui.click()
            self.last_left_click_time = now
            self.status_text = "Left Click"
        elif right_now and not self.right_pinched and now - self.last_right_click_time > CLICK_COOLDOWN:
            pyautogui.click(button="right")
            self.last_right_click_time = now
            self.status_text = "Right Click"
        elif left_now or right_now:
            self.status_text = "Pinch held"

        self.left_pinched = left_now
        self.right_pinched = right_now
        return left_now or right_now

    # ------------------------------------------------------------------
    def handle_scroll(self, hand_y_px):
        self.status_text = "Scroll mode"      # FIX: status no longer stays stale in the deadzone
        if self.scroll_ref_y is None:
            self.scroll_ref_y = hand_y_px

        delta = self.scroll_ref_y - hand_y_px  # positive = moved up

        if abs(delta) > SCROLL_DEADZONE:
            pyautogui.scroll(int(delta / SCROLL_SENSITIVITY))
            self.status_text = f"Scrolling {'Up' if delta > 0 else 'Down'}"
            self.scroll_ref_y = hand_y_px

    # ------------------------------------------------------------------
    def process_frame(self, frame):
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self.landmarker.detect_for_video(mp_image, self.next_timestamp_ms())

        cv2.rectangle(
            frame,
            (FRAME_REDUCTION, FRAME_REDUCTION),
            (w - FRAME_REDUCTION, h - FRAME_REDUCTION),
            (255, 0, 255), 2,
        )

        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            handedness_label = result.handedness[0][0].category_name  # "Left" / "Right"

            landmarks_px = [landmark_to_pixel(lm, w, h) for lm in landmarks]
            draw_hand(frame, landmarks_px)

            up = fingers_up(landmarks, handedness_label)
            thumb_px = landmarks_px[4]
            index_px = landmarks_px[8]
            middle_px = landmarks_px[12]

            # Wrist -> middle knuckle: a stable measure of hand size in pixels
            hand_size = max(pixel_distance(landmarks_px[0], landmarks_px[9]), 1.0)

            if is_fist(landmarks):
                # FIX: a fist brings the thumb next to the fingers and used to
                # trigger phantom clicks. Fist is always idle.
                self.release_pinches()
                self.scroll_ref_y = None
                self.smoother.reset()
                self.status_text = "Idle"
            else:
                pinch_close = (
                    pixel_distance(thumb_px, index_px) / hand_size < PINCH_OFF_RATIO
                    or pixel_distance(thumb_px, middle_px) / hand_size < PINCH_OFF_RATIO
                )
                pinching = False
                if pinch_close or self.left_pinched or self.right_pinched:
                    pinching = self.handle_click(thumb_px, index_px, middle_px, hand_size)

                if pinching:
                    self.scroll_ref_y = None
                    cv2.circle(frame, index_px, 12, (0, 255, 0), cv2.FILLED)

                elif up[1] and up[2] and not up[3] and not up[4]:
                    self.smoother.reset()
                    mid_point_y = (index_px[1] + middle_px[1]) // 2
                    self.handle_scroll(mid_point_y)

                elif up[1] and not up[2] and not up[3] and not up[4]:
                    self.scroll_ref_y = None
                    self.handle_move(index_px, w, h)

                else:
                    self.scroll_ref_y = None
                    self.smoother.reset()
                    self.status_text = "Idle"
        else:
            self.status_text = "No Hand Detected"
            self.release_pinches()
            self.scroll_ref_y = None
            self.smoother.reset()

        return frame

    # ------------------------------------------------------------------
    def draw_overlay(self, frame, fps):
        cv2.rectangle(frame, (0, 0), (250, 70), (20, 20, 20), cv2.FILLED)
        cv2.putText(frame, f"Status: {self.status_text}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # ------------------------------------------------------------------
    def run(self):
        prev_time = time.time()
        print("Hand Gesture Mouse Control started.")
        print("Quit: press 'q' or ESC in the video window, or move the physical "
              "mouse into a screen corner (failsafe).")

        try:
            while True:
                success, frame = self.cap.read()
                if not success:
                    print("Failed to read from webcam.")
                    break

                frame = cv2.flip(frame, 1)  # mirror for natural interaction
                frame = self.process_frame(frame)

                curr_time = time.time()
                dt = curr_time - prev_time
                fps = 1 / dt if dt > 0 else 0
                prev_time = curr_time
                self.fps_history.append(fps)
                avg_fps = sum(self.fps_history) / len(self.fps_history)

                self.draw_overlay(frame, avg_fps)
                cv2.imshow("Hand Gesture Mouse Control", frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
        except pyautogui.FailSafeException:
            print("Failsafe triggered (mouse moved to a screen corner). Exiting.")
        except KeyboardInterrupt:
            print("Interrupted. Exiting.")
        finally:
            # FIX: always release the camera/model, even on errors
            self.cap.release()
            cv2.destroyAllWindows()
            self.landmarker.close()


# ----------------------------------------------------------------------
if __name__ == "__main__":
    controller = GestureMouseController()
    controller.run()
