"""
body_language.py
================
Body Language Analyser — MediaPipe Pose
Project: A Unified Behavioural Intelligence System for Academic Spaces

PURPOSE:
  Analyses the teacher's body movements and posture in each frame.
  Used as PRIMARY signal when face is not detected (side camera, back view),
  and as SECONDARY context signal alongside face expression.

KEY JOINTS TRACKED (wrist + elbow weighted highest — gesture-heavy teaching):
  nose, left/right shoulder, left/right elbow, left/right wrist

SCORING:
  weighted sum of per-joint frame-to-frame displacement
  → smoothed over a rolling 5-frame window
  → is_enthusiastic = smoothed_score ≥ movement_threshold

OUTPUTS (per frame):
  detected         (bool)     — pose landmarks found
  movement_score   (float)    — raw weighted displacement this frame
  smoothed_score   (float)    — rolling-averaged score
  is_enthusiastic  (bool)     — above/below threshold
  landmarks        (dict)     — named landmark positions
  joint_moves      (dict)     — per-joint displacement for combined model input
  annotated_frame  (BGR array) — frame with skeleton overlay
"""

import numpy as np
from collections import deque
from pathlib import Path
import urllib.request
import os

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────

# Key landmarks to track (MediaPipe 33-landmark model)
LANDMARK_IDS = {
    "nose":           0,
    "left_shoulder":  11,
    "right_shoulder": 12,
    "left_elbow":     13,
    "right_elbow":    14,
    "left_wrist":     15,
    "right_wrist":    16,
}

# Higher weights on wrists/elbows — teaching gestures are arm-heavy
JOINT_WEIGHTS = {
    "nose":           0.5,
    "left_shoulder":  0.6,
    "right_shoulder": 0.6,
    "left_elbow":     1.0,
    "right_elbow":    1.0,
    "left_wrist":     1.5,
    "right_wrist":    1.5,
}

# MediaPipe skeleton connections for drawing
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26),
    (25, 27), (26, 28), (27, 29), (28, 30), (29, 31), (30, 32),
]

MOVEMENT_THRESHOLD  = 0.08   # smoothed score threshold to call "enthusiastic"
ROLLING_WINDOW_SIZE = 5      # frames for temporal smoothing

# Colours (BGR)
COLOUR_SKELETON    = (0,  200, 255)
COLOUR_JOINT       = (0,  255, 0)


# ─────────────────────────────────────────────────────────────────────────────
#  Analyser class
# ─────────────────────────────────────────────────────────────────────────────

class BodyLanguageAnalyser:
    """
    Analyses teacher body language via MediaPipe Pose landmarker.

    Usage:
        analyser = BodyLanguageAnalyser()
        result   = analyser.analyse_frame(bgr_frame)
        analyser.close()
    """

    def __init__(self,
                 movement_threshold: float = MOVEMENT_THRESHOLD,
                 rolling_window: int       = ROLLING_WINDOW_SIZE,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence:  float = 0.5):

        self.threshold    = movement_threshold
        self.rolling      = deque(maxlen=rolling_window)
        self._prev_lm     = None
        self._joint_moves = {name: 0.0 for name in LANDMARK_IDS}

        model_path = self._get_model_path()

        base_opts = mp_python.BaseOptions(model_asset_path=model_path)
        options   = mp_vision.PoseLandmarkerOptions(
            base_options=base_opts,
            output_segmentation_masks=False,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._pose = mp_vision.PoseLandmarker.create_from_options(options)

    # ── Frame analysis ────────────────────────────────────────────────────

    def analyse_frame(self, bgr_frame: np.ndarray) -> dict:
        """
        Run pose detection on one frame, compute movement score.

        Returns dict with all body language signals for this frame.
        """
        annotated = bgr_frame.copy()
        rgb       = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result    = self._pose.detect(mp_image)

        # No pose detected → reset, return zero-score result
        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            self._prev_lm     = None
            self._joint_moves = {name: 0.0 for name in LANDMARK_IDS}
            self.rolling.append(0.0)
            return self._make_result(detected=False, raw_score=0.0,
                                     annotated=annotated, landmarks={})

        landmarks = result.pose_landmarks[0]
        h, w = bgr_frame.shape[:2]

        # Draw skeleton overlay
        for (s, e) in POSE_CONNECTIONS:
            if s < len(landmarks) and e < len(landmarks):
                pt1 = (int(landmarks[s].x * w), int(landmarks[s].y * h))
                pt2 = (int(landmarks[e].x * w), int(landmarks[e].y * h))
                cv2.line(annotated, pt1, pt2, COLOUR_SKELETON, 2)
        for lm in landmarks:
            cv2.circle(annotated,
                       (int(lm.x * w), int(lm.y * h)), 3,
                       COLOUR_JOINT, -1)

        # Extract named key landmarks
        curr_lm = {}
        for name, idx in LANDMARK_IDS.items():
            if idx < len(landmarks):
                lm  = landmarks[idx]
                vis = lm.visibility if hasattr(lm, "visibility") else 1.0
                curr_lm[name] = (lm.x, lm.y, vis)

        # Compute per-joint movement (frame-to-frame displacement)
        raw_score   = 0.0
        joint_moves = {name: 0.0 for name in LANDMARK_IDS}

        if self._prev_lm is not None:
            for name, (cx, cy, cvis) in curr_lm.items():
                if cvis < 0.3:          # low-confidence landmark → skip
                    continue
                px, py, _ = self._prev_lm.get(name, (cx, cy, 0))
                dist = np.hypot(cx - px, cy - py)
                joint_moves[name] = dist
                raw_score += dist * JOINT_WEIGHTS[name]

        self._prev_lm     = curr_lm
        self._joint_moves = joint_moves
        self.rolling.append(raw_score)
        smoothed = float(np.mean(self.rolling)) if self.rolling else 0.0

        return self._make_result(
            detected=True, raw_score=raw_score, smoothed=smoothed,
            annotated=annotated, landmarks=curr_lm, joint_moves=joint_moves
        )

    # ── Helpers ───────────────────────────────────────────────────────────

    def reset(self):
        """Reset temporal state (call between separate video clips)."""
        self._prev_lm     = None
        self._joint_moves = {name: 0.0 for name in LANDMARK_IDS}
        self.rolling.clear()

    def close(self):
        """Release MediaPipe resources."""
        self._pose.close()

    # ── Model download ────────────────────────────────────────────────────

    @staticmethod
    def _get_model_path() -> str:
        """Download pose landmarker model if not already cached."""
        model_path = "pose_landmarker.task"
        if not os.path.exists(model_path):
            print("[BodyLanguage] Downloading MediaPipe pose model (~7 MB) ...")
            url = ("https://storage.googleapis.com/mediapipe-models/"
                   "pose_landmarker/pose_landmarker_lite/float16/latest/"
                   "pose_landmarker_lite.task")
            urllib.request.urlretrieve(url, model_path)
            print(f"[BodyLanguage] Model saved → {model_path}")
        return model_path

    # ── Result builder ────────────────────────────────────────────────────

    def _make_result(self, detected: bool, raw_score: float,
                     annotated: np.ndarray, landmarks: dict,
                     smoothed: float = None,
                     joint_moves: dict = None) -> dict:
        if smoothed is None:
            smoothed = float(np.mean(self.rolling)) if self.rolling else 0.0
        if joint_moves is None:
            joint_moves = {name: 0.0 for name in LANDMARK_IDS}

        return {
            "detected":        detected,
            "movement_score":  raw_score,
            "smoothed_score":  smoothed,
            "is_enthusiastic": smoothed >= self.threshold,
            "landmarks":       landmarks,
            "joint_moves":     joint_moves,
            "annotated_frame": annotated,
        }