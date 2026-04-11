"""
ai/pipeline.py — AI Processing Pipeline
──────────────────────────────────────────
THIS IS WHERE YOUR EXISTING CODE PLUGS IN.

run_ai_pipeline() takes raw video bytes and returns a dict
of scores. This function is called by the verify router
in a background thread.

To integrate your existing deepfake detector:
  1. Replace the logic inside _extract_features_from_video()
     with your actual OpenCV + MediaPipe code.
  2. The rest of the pipeline (scoring, result building) stays the same.

Current implementation:
  Reads video frames → MediaPipe face mesh → EAR → blink count
  → head pose → texture score → weighted confidence → verdict
"""

import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
from collections import deque
import time
import math
import tempfile
import os

# ── MediaPipe setup ───────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE  = [33,  160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
NOSE_TIP  = 4
CHIN      = 152

# Thresholds
EAR_THRESHOLD       = 0.22
BLINK_CONSEC_FRAMES = 2
REAL_BLINK_REQUIRED = 2


# ════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT — called by the router
# ════════════════════════════════════════════════════════

def run_ai_pipeline(video_bytes: bytes) -> dict:
    """
    Main entry point. Takes raw video bytes, returns result dict.
    
    Args:
        video_bytes: Raw bytes of the uploaded video file.
    
    Returns:
        {
            "verdict": "REAL" | "FAKE" | "INCONCLUSIVE",
            "confidence": 0-100,
            "authenticity_score": 0.0-1.0,
            "signals": { blink_count, avg_ear, ... },
            "ear_timeline": [0.31, 0.30, ...],
            "processing_ms": 1240
        }
    """
    start_time = time.time()

    # ── Write bytes to a temp file so OpenCV can read it ─────────
    # OpenCV VideoCapture needs a file path, not bytes.
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    try:
        features = _extract_features_from_video(tmp_path)
    finally:
        os.unlink(tmp_path)     # Always delete temp file

    # ── Compute final scores ──────────────────────────────────────
    confidence         = _compute_confidence(features)
    authenticity_score = round(confidence / 100.0, 3)
    verdict            = _determine_verdict(features, confidence)
    processing_ms      = int((time.time() - start_time) * 1000)

    return {
        "verdict":            verdict,
        "confidence":         confidence,
        "authenticity_score": authenticity_score,
        "signals": {
            "blink_count":      features["blink_count"],
            "avg_ear":          round(features["avg_ear"], 4),
            "min_ear":          round(features["min_ear"], 4),
            "head_yaw_deg":     round(features["yaw"], 2),
            "head_pitch_deg":   round(features["pitch"], 2),
            "texture_score":    round(features["texture_score"], 2),
            "motion_entropy":   round(features["motion_entropy"], 3),
            "frames_analyzed":  features["frames_analyzed"]
        },
        "ear_timeline":   features["ear_timeline"],
        "processing_ms":  processing_ms
    }


# ════════════════════════════════════════════════════════
# FEATURE EXTRACTION — your OpenCV + MediaPipe code here
# ════════════════════════════════════════════════════════

def _extract_features_from_video(video_path: str) -> dict:
    """
    Reads video file frame by frame and extracts all signals.
    This is where your existing detector code lives.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Sample at 5fps regardless of source fps to keep processing fast
    frame_interval = max(1, int(fps / 5))

    blink_count    = 0
    consec_closed  = 0
    ear_history    = []
    texture_scores = []
    prev_gray      = None
    flow_magnitudes= []
    yaw_readings   = []
    pitch_readings = []
    frames_analyzed= 0

    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # ── Only process every Nth frame (5fps sampling) ─────
        if frame_idx % frame_interval != 0:
            continue

        h, w = frame.shape[:2]
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        if not result.multi_face_landmarks:
            continue

        frames_analyzed += 1
        landmarks = result.multi_face_landmarks[0].landmark

        # ── EAR ──────────────────────────────────────────────
        left_ear  = _ear(landmarks, LEFT_EYE,  w, h)
        right_ear = _ear(landmarks, RIGHT_EYE, w, h)
        ear       = (left_ear + right_ear) / 2.0
        ear_history.append(round(ear, 4))

        # ── Blink logic ───────────────────────────────────────
        if ear < EAR_THRESHOLD:
            consec_closed += 1
        else:
            if consec_closed >= BLINK_CONSEC_FRAMES:
                blink_count += 1
            consec_closed = 0

        # ── Texture score (forehead region) ───────────────────
        tx = _texture_score(frame, landmarks, w, h)
        texture_scores.append(tx)

        # ── Head pose ─────────────────────────────────────────
        yaw, pitch = _head_pose(landmarks, w, h)
        yaw_readings.append(yaw)
        pitch_readings.append(pitch)

        # ── Optical flow (motion entropy) ─────────────────────
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            flow_magnitudes.append(float(np.mean(mag)))
        prev_gray = gray

    cap.release()
    face_mesh.close()

    # ── Aggregate ─────────────────────────────────────────────────
    avg_ear       = float(np.mean(ear_history))    if ear_history    else 0.0
    min_ear       = float(np.min(ear_history))     if ear_history    else 0.0
    avg_texture   = float(np.mean(texture_scores)) if texture_scores else 0.0
    motion_entropy= _motion_entropy(flow_magnitudes)
    avg_yaw       = float(np.mean(yaw_readings))   if yaw_readings   else 0.0
    avg_pitch     = float(np.mean(pitch_readings)) if pitch_readings else 0.0

    return {
        "blink_count":    blink_count,
        "avg_ear":        avg_ear,
        "min_ear":        min_ear,
        "texture_score":  avg_texture,
        "motion_entropy": motion_entropy,
        "yaw":            avg_yaw,
        "pitch":          avg_pitch,
        "ear_timeline":   ear_history,
        "frames_analyzed": frames_analyzed
    }


# ════════════════════════════════════════════════════════
# SCORING AND VERDICT
# ════════════════════════════════════════════════════════

def _compute_confidence(f: dict) -> int:
    """
    Weighted confidence score (0–100).
    Each signal contributes a percentage of the total.
    
    Weights:
      Blinks   40% — most reliable anti-spoofing signal
      Texture  25% — real skin has micro-texture; deepfakes are smooth
      Motion   20% — real faces have natural micro-movements
      Pose     15% — static deepfakes don't vary pose
    """
    # Blink score: full marks at REAL_BLINK_REQUIRED blinks
    blink_score   = min(f["blink_count"] / max(REAL_BLINK_REQUIRED, 1), 1.0) * 40

    # Texture score: full marks at variance > 30 (natural skin)
    texture_score = min(f["texture_score"] / 30.0, 1.0) * 25

    # Motion entropy: full marks at 0.7+ (natural head movement)
    motion_score  = min(f["motion_entropy"] / 0.7, 1.0) * 20

    # Pose variation: full marks if head moved more than 5 degrees
    pose_variation = abs(f["yaw"]) + abs(f["pitch"])
    pose_score    = min(pose_variation / 10.0, 1.0) * 15

    return int(blink_score + texture_score + motion_score + pose_score)


def _determine_verdict(f: dict, confidence: int) -> str:
    """
    Hard rules + confidence threshold → REAL / FAKE / INCONCLUSIVE.
    
    Logic:
      No blinks at all → FAKE (hard rule, regardless of confidence)
      Confidence < 40  → INCONCLUSIVE (not enough signal)
      Confidence >= 60 → REAL
      Else             → FAKE
    """
    if f["frames_analyzed"] < 10:
        return "INCONCLUSIVE"   # Too few frames to decide

    if f["blink_count"] == 0:
        return "FAKE"           # Hard rule — no blinks = fake

    if confidence < 40:
        return "INCONCLUSIVE"

    if confidence >= 60:
        return "REAL"

    return "FAKE"


# ════════════════════════════════════════════════════════
# SIGNAL HELPERS
# ════════════════════════════════════════════════════════

def _ear(landmarks, eye_indices, w, h) -> float:
    """Eye Aspect Ratio: measures how open the eye is."""
    pts = [(landmarks[p].x * w, landmarks[p].y * h) for p in eye_indices]
    A = dist.euclidean(pts[1], pts[5])
    B = dist.euclidean(pts[2], pts[4])
    C = dist.euclidean(pts[0], pts[3])
    return (A + B) / (2.0 * C) if C > 0 else 0.0


def _texture_score(frame, landmarks, w, h) -> float:
    """Laplacian variance of forehead skin — low = suspiciously smooth."""
    nx, ny = int(landmarks[NOSE_TIP].x * w), int(landmarks[NOSE_TIP].y * h)
    fx, fy = int(landmarks[10].x * w),       int(landmarks[10].y * h)
    cx, cy = (nx + fx) // 2,                 (ny + fy) // 2
    size   = max(30, min(w, h) // 8)
    x1, y1 = max(0, cx - size), max(0, cy - size)
    x2, y2 = min(w, cx + size), min(h, cy + size)
    roi    = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return float(np.var(cv2.Laplacian(gray_roi, cv2.CV_64F)))


def _head_pose(landmarks, w, h):
    """Estimate yaw and pitch from 6 facial landmarks using solvePnP."""
    model_pts = np.array([
        [0.0,    0.0,    0.0   ],
        [0.0,   -330.0, -65.0  ],
        [-225.0, 170.0, -135.0 ],
        [225.0,  170.0, -135.0 ],
        [-150.0,-150.0, -125.0 ],
        [150.0, -150.0, -125.0 ]
    ], dtype=np.float64)

    idxs      = [NOSE_TIP, CHIN, 33, 263, 61, 291]
    image_pts = np.array(
        [(landmarks[i].x * w, landmarks[i].y * h) for i in idxs],
        dtype=np.float64
    )
    cam = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(model_pts, image_pts, cam, np.zeros((4,1)),
                                flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0.0, 0.0
    rmat, _  = cv2.Rodrigues(rvec)
    angles, *_ = cv2.RQDecomp3x3(rmat)
    return round(angles[1], 2), round(angles[0], 2)


def _motion_entropy(magnitudes: list) -> float:
    """
    Normalised variance of optical flow magnitudes.
    Real people move; static images/loops have zero flow.
    """
    if len(magnitudes) < 2:
        return 0.0
    arr = np.array(magnitudes)
    variance = float(np.var(arr))
    return round(min(variance / 0.5, 1.0), 4)