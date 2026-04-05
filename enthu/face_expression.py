"""
face_expression.py
==================
Facial Expression Classifier — Inference Module
Project: A Unified Behavioural Intelligence System for Academic Spaces

PURPOSE:
  Detects the teacher's face in a video frame and classifies it as
  ENTHUSIASTIC (happy/surprise) or NOT ENTHUSIASTIC (angry/neutral/sad/…).

APPROACH (matching design doc + literature survey):
  - YOLOv8-inspired region-of-interest: teacher zone (upper 60% of frame)
  - Haar cascade frontal + profile detection for robustness
  - Eye check for frontal faces (avoids false positives)
  - Domain-normalization preprocessing identical to training
  - Student faces detected outside teacher zone are ignored

INPUTS:  BGR video frame (numpy array)
OUTPUTS: dict with face_found, probability, is_enthusiastic, face_bbox, annotated_frame

NOTE: Live video capture is NOT handled here; this module processes individual
      frames supplied by analyze_video.py.
"""

import cv2
import os
import numpy as np
import tensorflow as tf

# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────
IMG_SIZE = 64

# Cascade paths (OpenCV built-in)
_FACE_CASCADE_PATH    = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_EYE_CASCADE_PATH     = cv2.data.haarcascades + "haarcascade_eye.xml"
_PROFILE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_profileface.xml"

# Teacher zone — upper portion of frame where instructor usually stands
# Horizontally wide to handle side-angle shots
ZONE_X1, ZONE_X2 = 0.10, 0.90
ZONE_Y1, ZONE_Y2 = 0.00, 0.60

# Visual colours (BGR)
COLOUR_ENT     = (0,  220, 80)
COLOUR_NOT     = (0,   60, 220)
COLOUR_YELLOW  = (0,  220, 220)
COLOUR_STUDENT = (50,  50, 200)
COLOUR_ZONE    = (255, 255, 0)
COLOUR_WHITE   = (255, 255, 255)


# ─────────────────────────────────────────────────────────────────────────────
#  Preprocessing — MUST BE IDENTICAL to train_model.py
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_face(img_bgr: np.ndarray, img_size: int = IMG_SIZE) -> np.ndarray:
    """
    Preprocessing identical to training (critical for accuracy).
    Works for both real classroom footage and AI-generated test videos.
    """
    # 1. Blur to reduce AI sharpness artefacts
    img_bgr = cv2.GaussianBlur(img_bgr, (3, 3), 0.5)

    # 2. Resolution round-trip — normalises texture between real and AI faces
    h, w  = img_bgr.shape[:2]
    small = cv2.resize(img_bgr, (max(w // 4, 8), max(h // 4, 8)),
                       interpolation=cv2.INTER_AREA)
    img_bgr = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    # 3. CLAHE grayscale — handles uneven classroom lighting
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    eq    = clahe.apply(gray)

    # 4. Resize to model input size
    resized = cv2.resize(eq, (img_size, img_size))

    # 5. Three channels + normalise to [-1, 1]
    rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    return rgb.astype(np.float32) / 127.5 - 1.0


# ─────────────────────────────────────────────────────────────────────────────
#  Eye check
# ─────────────────────────────────────────────────────────────────────────────

def _has_eyes(face_crop_bgr: np.ndarray) -> bool:
    """
    Check for at least one eye in a frontal face crop.
    Reduces false positives from wall-art, boards, etc.
    One eye is enough to handle partial occlusion.
    """
    gray        = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2GRAY)
    eye_cascade = cv2.CascadeClassifier(_EYE_CASCADE_PATH)
    eyes        = eye_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=2, minSize=(5, 5)
    )
    return len(eyes) >= 1


# ─────────────────────────────────────────────────────────────────────────────
#  Main classifier class
# ─────────────────────────────────────────────────────────────────────────────

class FaceExpressionClassifier:
    """
    Detects the teacher's face in a frame and classifies enthusiasm level.

    Decision rule:
      face ENTHUSIASTIC (happy/surprise prob ≥ threshold) → ENTHUSIASTIC
      face NOT enthusiastic                               → NOT ENTHUSIASTIC
      no face found (side cam / back of head)             → face_found=False
          → analyze_video.py falls back to body language
    """

    def __init__(self, model_path: str, conf_threshold: float = 0.5,
                 img_size: int = IMG_SIZE):
        self.img_size       = img_size
        self.conf_threshold = conf_threshold
        self.model          = self._load_model(model_path)
        self._face_det      = cv2.CascadeClassifier(_FACE_CASCADE_PATH)
        self._profile_det   = cv2.CascadeClassifier(_PROFILE_CASCADE_PATH)

    # ── Model loading ─────────────────────────────────────────────────────

    def _load_model(self, model_path: str) -> tf.keras.Model:
        """Try multiple candidate paths so .h5 / .keras both work."""
        candidates = [
            model_path,
            model_path.replace(".h5", ".keras"),
            model_path.replace(".keras", ".h5"),
            "models/face_model.keras",
            "models/face_model.h5",
            "face_model.keras",
            "face_model.h5",
        ]
        for path in candidates:
            if os.path.isfile(path):
                try:
                    print(f"[FaceClassifier] Loading model: {path}")
                    model = tf.keras.models.load_model(path, compile=False)
                    model.compile(optimizer="adam",
                                  loss="binary_crossentropy",
                                  metrics=["accuracy"])
                    print("[FaceClassifier] Model loaded successfully.")
                    return model
                except Exception as exc:
                    print(f"[FaceClassifier] Failed to load {path}: {exc}")
        raise RuntimeError(
            "[FaceClassifier] No valid model found. "
            "Run: python src/train_model.py --train_dir data/train --test_dir data/test"
        )

    # ── Face detection inside teacher zone ───────────────────────────────

    def _detect_in_roi(self, roi_gray: np.ndarray,
                       offset_x: int, offset_y: int) -> list[tuple]:
        """
        Frontal + profile detection inside the teacher zone ROI.
        Returns list of (fx, fy, fw, fh, det_type) in full-frame coordinates.
        """
        candidates = []

        # Frontal
        frontal = self._face_det.detectMultiScale(
            roi_gray, scaleFactor=1.05, minNeighbors=3, minSize=(15, 15)
        )
        for (x, y, w, h) in frontal:
            candidates.append((x + offset_x, y + offset_y, w, h, "frontal"))

        # Profile left
        profile_l = self._profile_det.detectMultiScale(
            roi_gray, scaleFactor=1.05, minNeighbors=3, minSize=(15, 15)
        )
        for (x, y, w, h) in profile_l:
            candidates.append((x + offset_x, y + offset_y, w, h, "profile_L"))

        # Profile right (flipped)
        roi_flipped = cv2.flip(roi_gray, 1)
        w_roi       = roi_gray.shape[1]
        profile_r   = self._profile_det.detectMultiScale(
            roi_flipped, scaleFactor=1.05, minNeighbors=3, minSize=(15, 15)
        )
        for (x, y, w, h) in profile_r:
            fx = (w_roi - x - w) + offset_x
            candidates.append((fx, y + offset_y, w, h, "profile_R"))

        return candidates

    # ── Main inference ────────────────────────────────────────────────────

    def analyse_frame(self, bgr_frame: np.ndarray) -> dict:
        """
        Analyse one frame and return face detection + classification result.

        Returns:
            dict:
              face_found      (bool)
              probability     (float, 0-1, higher = more enthusiastic)
              is_enthusiastic (bool)
              face_bbox       (x, y, w, h) or None
              annotated_frame (BGR numpy array with overlays)
        """
        annotated        = bgr_frame.copy()
        h_frame, w_frame = bgr_frame.shape[:2]

        # Draw teacher zone
        zx1 = int(w_frame * ZONE_X1); zx2 = int(w_frame * ZONE_X2)
        zy1 = int(h_frame * ZONE_Y1); zy2 = int(h_frame * ZONE_Y2)
        cv2.rectangle(annotated, (zx1, zy1), (zx2, zy2), COLOUR_ZONE, 1)
        cv2.putText(annotated, "TEACHER ZONE", (zx1 + 5, zy1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOUR_ZONE, 1)

        # Detect faces inside zone
        roi      = bgr_frame[zy1:zy2, zx1:zx2]
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        zone_faces = self._detect_in_roi(roi_gray, offset_x=zx1, offset_y=zy1)

        # Mark student faces outside zone (for UI clarity)
        full_gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        all_faces = self._face_det.detectMultiScale(
            full_gray, scaleFactor=1.05, minNeighbors=3, minSize=(15, 15)
        )
        for (fx, fy, fw, fh) in all_faces:
            cx = fx + fw / 2; cy = fy + fh / 2
            if not (zx1 <= cx <= zx2 and zy1 <= cy <= zy2):
                cv2.rectangle(annotated, (fx, fy), (fx + fw, fy + fh),
                              COLOUR_STUDENT, 1)
                cv2.putText(annotated, "student", (fx, fy - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLOUR_STUDENT, 1)

        if not zone_faces:
            cv2.putText(annotated, "Teacher not detected (side/back cam — body fallback)",
                        (zx1, zy2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOUR_YELLOW, 1)
            return self._empty(annotated)

        # Score each candidate and pick best
        cx_zone    = (zx2 - zx1) / 2
        candidates = []

        for (fx, fy, fw, fh, det_type) in zone_faces:
            fx = max(0, fx); fy = max(0, fy)
            fw = min(fw, w_frame - fx); fh = min(fh, h_frame - fy)
            if fw <= 0 or fh <= 0:
                continue

            face_crop = bgr_frame[fy:fy + fh, fx:fx + fw]
            if face_crop.size == 0:
                continue

            # Eye check for frontal detections only
            if det_type == "frontal" and not _has_eyes(face_crop):
                cv2.rectangle(annotated, (fx, fy), (fx + fw, fy + fh),
                              (80, 80, 80), 1)
                cv2.putText(annotated, "no_eyes", (fx, fy - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.28, (80, 80, 80), 1)
                continue

            # Scoring: prefer centred, upper, larger, frontal
            cx_rel       = (fx - zx1) + fw / 2
            cy_rel       = (fy - zy1) + fh / 2
            centre_score = 1.0 - abs(cx_rel - cx_zone) / max(cx_zone, 1)
            height_score = 1.0 - cy_rel / max(zy2 - zy1, 1)
            size_score   = fw * fh
            type_bonus   = 0.2 if det_type == "frontal" else 0.0
            score        = (centre_score * 0.4 + height_score * 0.3
                            + size_score * 0.3 + type_bonus)
            candidates.append(((fx, fy, fw, fh), score, det_type))

        if not candidates:
            cv2.putText(annotated, "No valid teacher face (eye check failed)",
                        (zx1, zy2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOUR_YELLOW, 1)
            return self._empty(annotated)

        # Best candidate
        (fx, fy, fw, fh), _, det_type = max(candidates, key=lambda c: c[1])
        face_crop = bgr_frame[fy:fy + fh, fx:fx + fw]

        # Classify
        processed = preprocess_face(face_crop, self.img_size)
        prob      = float(self.model.predict(processed[None], verbose=0)[0][0])
        is_enth   = prob >= self.conf_threshold

        # Draw result on frame
        col   = COLOUR_ENT if is_enth else COLOUR_NOT
        label = (f"TEACHER({det_type}): "
                 f"{'ENTHUSIASTIC' if is_enth else 'NOT enthus'} ({prob:.2f})")
        cv2.rectangle(annotated, (fx, fy), (fx + fw, fy + fh), col, 3)
        cv2.putText(annotated, label,
                    (max(fx - 10, 0), max(fy - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)

        return {
            "face_found":      True,
            "probability":     prob,
            "is_enthusiastic": is_enth,
            "face_bbox":       (fx, fy, fw, fh),
            "annotated_frame": annotated,
        }

    # ── Empty result (no face) ────────────────────────────────────────────

    def _empty(self, annotated: np.ndarray) -> dict:
        return {
            "face_found":      False,
            "probability":     0.0,
            "is_enthusiastic": False,
            "face_bbox":       None,
            "annotated_frame": annotated,
        }