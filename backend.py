"""
backend.py
==========
All ML inference, face-tracking logic, and CSV persistence.
Zero Streamlit imports — pure Python / OpenCV / NumPy.

Public API
----------
Detectors:
    get_detectors()                     → (face_detector, face_mesher, clahe)
    clear_inference_cache()

Inference:
    compute_au_scores(lm, w, h)         → np.ndarray (7,)
    run_deepface(face_bgr)              → np.ndarray (7,) | None
    calibrate(raw_probs)                → np.ndarray (7,)
    fuse(df_probs, au_scores)           → np.ndarray (7,)
    ewa_update(buffer, new_pred)        → np.ndarray (7,)

Person state:
    make_person()                       → dict
    close_current_emotion(person, t)
    get_dominant_emotion(duration_map)  → str

Frame processing:
    extract_face_crop(frame, det, w, h) → (crop | None, x1, y1, x2, y2)
    draw_face_overlay(frame, ...)
    process_frame(frame, now, persons, detector, mesher) → annotated_frame

CSV:
    save_session_to_csv(persons, session_id, session_start, session_end)
    load_csv()                          → pd.DataFrame | None

Utilities:
    fmt_dur(seconds)                    → str
"""

import cv2
import mediapipe as mp
import numpy as np
from deepface import DeepFace
from collections import deque
import math, time, base64, csv, os, uuid
from io import BytesIO
from PIL import Image
import pandas as pd

from config import (
    EMOTION_LABELS, MAX_FACES,
    BUFFER_LEN, EWA_ALPHA, CONFIDENCE_GATE,
    ANALYZE_EVERY, PADDING_RATIO, AU_FUSION_WEIGHT,
    TEMPERATURE, PRIOR_CORRECTION,
    PERSON_COLORS_BGR, PERSON_LABELS,
    EMOTION_EMOJIS, CSV_PATH, CSV_COLUMNS,
)


# MODULE-LEVEL INFERENCE CACHES


_face_buffers : dict[int, deque]       = {}
_face_last    : dict[int, np.ndarray]  = {}
_face_au_last : dict[int, np.ndarray]  = {}
_frame_num    : int                    = 0


def clear_inference_cache() -> None:
    #  Reset all inference state. Call when starting a new session.
    _face_buffers.clear()
    _face_last.clear()
    _face_au_last.clear()
    global _frame_num
    _frame_num = 0


# ─────────────────────────────────────────────────────────
# DETECTOR INITIALISATION
# ─────────────────────────────────────────────────────────

def get_detectors():
    
    # Initialise MediaPipe face detector, face mesher, and CLAHE.
    # Intended to be called once and cached (e.g. with @st.cache_resource).
    
    detector = mp.solutions.face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.55,
    )
    mesher = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=MAX_FACES,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return detector, mesher, clahe


# ACTION UNIT SCORING


def compute_au_scores(landmarks, img_w: int, img_h: int) -> np.ndarray:
    """
    Derive proxy Action Unit intensities from MediaPipe face landmarks.

    AUs computed:
        AU4  — Brow Lowerer        → Angry + Disgust
        AU9  — Nose Wrinkler       → Disgust
        AU17 — Chin Raiser         → Disgust + Sad
        AU23 — Lip Tightener       → Angry
        AU1/2 — Brow Raise         → Surprise + Fear

    Returns (7,) float32 array soft-normalised to sum ≈ 1.
    """
    scores = np.zeros(7, dtype=np.float32)
    try:
        def lm_y(idx):
            return landmarks[idx].y * img_h

        def lm_dist(a, b):
            return math.hypot(
                (landmarks[a].x - landmarks[b].x) * img_w,
                (landmarks[a].y - landmarks[b].y) * img_h,
            )

        face_h = max(lm_dist(10, 152), 1.0)

        # AU4 — Brow Depression
        lb = np.mean([lm_y(i) for i in [285, 295, 282, 283, 276]])
        rb = np.mean([lm_y(i) for i in [55,  65,  52,  53,  46]])
        bd = ((lm_y(362) - lb) + (lm_y(133) - rb)) / (2 * face_h)
        au4  = float(np.clip(1.0 - (bd / 0.12), 0, 1))

        # AU9 — Nose Wrinkler
        au9  = float(np.clip(((lm_y(94) - lm_y(4)) / face_h - 0.08) / 0.06, 0, 1))

        # AU17 — Chin Raiser
        au17 = float(np.clip(1.0 - ((lm_y(152) - lm_y(13)) / face_h / 0.20), 0, 1))

        # AU23 — Lip Tightener
        lip_w = lm_dist(61, 291) / face_h
        au23  = float(np.clip(1.0 - (lip_w / 0.28), 0, 1))

        # AU1/2 — Brow Raise
        bt   = np.mean([landmarks[i].y * img_h for i in [336, 296, 334, 293, 300, 107, 66, 105, 63, 70]])
        ea   = (lm_y(362) + lm_y(133)) / 2
        au12 = float(np.clip(((ea - bt) / face_h - 0.06) / 0.08, 0, 1))

        scores[0] = np.clip(0.60 * au4  + 0.40 * au23,                  0, 1)  # angry
        scores[1] = np.clip(0.45 * au4  + 0.40 * au9 + 0.15 * au17,    0, 1)  # disgust
        scores[2] = np.clip(0.55 * au12 + 0.45 * au4,                   0, 1)  # fear
        scores[3] = float(np.clip(lip_w / 0.35, 0, 1))                         # happy
        scores[4] = np.clip(0.60 * au17 + 0.40 * (1 - au4),             0, 1)  # sad
        scores[5] = np.clip(au12, 0, 1)                                         # surprise
        scores[6] = float(np.clip(1.0 - np.max(scores[:6]), 0, 1))             # neutral

        total = scores.sum()
        if total > 0:
            scores /= total
    except Exception:
        pass
    return scores


# DEEPFACE INFERENCE


def run_deepface(face_bgr: np.ndarray) -> np.ndarray | None:
    """
    Run DeepFace emotion analysis on a BGR face crop.
    Returns (7,) probability array or None on failure.
    """
    try:
        result = DeepFace.analyze(
            img_path=face_bgr,
            actions=['emotion'],
            enforce_detection=False,
            silent=True,
        )
        raw    = result[0]['emotion']
        scores = np.array([raw[e] for e in EMOTION_LABELS], dtype=np.float32)
        return scores / scores.sum()
    except Exception:
        return None


# CALIBRATION


def calibrate(raw_probs: np.ndarray) -> np.ndarray:
    """
    Temperature scaling + prior correction.
    Softens overconfident predictions and compensates for FER2013 class imbalance.
    """
    eps    = 1e-7
    logits = np.log(np.clip(raw_probs, eps, 1.0)) / TEMPERATURE
    logits -= logits.max()
    probs   = np.exp(logits);          probs /= probs.sum()
    probs   = probs * PRIOR_CORRECTION; probs /= probs.sum()
    return probs.astype(np.float32)



# FUSION


def fuse(deepface_probs: np.ndarray, au_scores: np.ndarray) -> np.ndarray:
    """Blend calibrated DeepFace probabilities with AU geometric scores."""
    blended  = (1 - AU_FUSION_WEIGHT) * deepface_probs + AU_FUSION_WEIGHT * au_scores
    blended /= blended.sum()
    return blended.astype(np.float32)




# SMOOTHING


def ewa_update(buffer: deque, new_pred: np.ndarray) -> np.ndarray:
    """Exponential weighted average — recent frames weighted more heavily."""
    if not buffer:
        return new_pred
    arr     = np.array(list(buffer))
    n       = len(arr)
    weights = np.array([(1 - EWA_ALPHA) ** (n - 1 - i) for i in range(n)])
    weights /= weights.sum()
    return np.dot(weights, arr)



# SNAPSHOT HELPER



def crop_to_b64(face_bgr: np.ndarray) -> str:
    """Convert BGR face crop to base64 PNG string for HTML embedding."""
    try:
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb).resize((80, 80))
        buf = BytesIO()
        pil.save(buf, format='PNG')
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""



# PERSON STATE MANAGEMENT



def make_person() -> dict:
    """
    Create a fresh per-person state dict.

    Keys:
        timeline        — list of emotion entry dicts (one per unique emotion)
        current_emotion — emotion label currently being tracked
        current_start   — epoch time when current emotion started
        snapshots       — {emotion: base64_png} — first capture only
        duration_map    — {emotion: total_seconds}
        last_probs      — latest (7,) probability array for live display
    """
    return {
        'timeline':        [],
        'current_emotion': None,
        'current_start':   None,
        'snapshots':       {},
        'duration_map':    {e: 0.0 for e in EMOTION_LABELS},
        'last_probs':      np.ones(7, dtype=np.float32) / 7,
    }


def close_current_emotion(person: dict, end_time: float) -> None:
    #  Finalise the running emotion segment and accumulate duration.
    ce, cs = person['current_emotion'], person['current_start']
    if ce and cs:
        dur = max(0.0, end_time - cs)
        person['duration_map'][ce] = person['duration_map'].get(ce, 0.0) + dur
        if person['timeline'] and person['timeline'][-1]['emotion'] == ce:
            person['timeline'][-1]['end_ts']   = end_time
            person['timeline'][-1]['duration'] = person['duration_map'][ce]
    person['current_emotion'] = None
    person['current_start']   = None


def _open_emotion_segment(person: dict, emotion: str, face_crop: np.ndarray, now: float) -> None:
    # Start a new emotion segment; capture snapshot on first occurrence.
    person['current_emotion'] = emotion
    person['current_start']   = now
    if emotion not in person['snapshots']:
        b64 = crop_to_b64(face_crop)
        if b64:
            person['snapshots'][emotion] = b64
        person['timeline'].append({
            'emotion':    emotion,
            'start_ts':   now,
            'end_ts':     now,
            'duration':   0.0,
            'snapshot':   b64 if b64 else '',
            'first_seen': time.strftime("%H:%M:%S", time.localtime(now)),
        })


def _accumulate_running_emotion(person: dict, emotion: str, now: float) -> None:
    # Add elapsed delta to duration_map and reset current_start.
    cs = person['current_start']
    if cs:
        delta = now - cs
        person['duration_map'][emotion] = person['duration_map'].get(emotion, 0.0) + delta
        person['current_start'] = now
        if person['timeline'] and person['timeline'][-1]['emotion'] == emotion:
            person['timeline'][-1]['duration'] = person['duration_map'][emotion]


def get_dominant_emotion(duration_map: dict) -> str:
    # Return the emotion with the highest accumulated duration.
    return max(duration_map, key=lambda e: duration_map[e]) \
        if any(duration_map.values()) else 'neutral'



# FACE CROP EXTRACTION

def extract_face_crop(
    frame: np.ndarray,
    detection,
    img_w: int,
    img_h: int,
) -> tuple:
    
    # Convert a MediaPipe detection into a padded face crop (BGR).
    # Returns (face_crop, x1, y1, x2, y2) or (None, 0, 0, 0, 0).
    
    bbox  = detection.location_data.relative_bounding_box
    bx    = int(bbox.xmin  * img_w);  by = int(bbox.ymin  * img_h)
    bw    = int(bbox.width * img_w);  bh = int(bbox.height * img_h)
    pad_x = int(bw * PADDING_RATIO);  pad_y = int(bh * PADDING_RATIO)
    x1    = max(0, bx - pad_x);       y1 = max(0, by - pad_y)
    x2    = min(img_w, bx+bw+pad_x);  y2 = min(img_h, by+bh+pad_y)
    crop  = frame[y1:y2, x1:x2]
    return (None, 0, 0, 0, 0) if crop.size == 0 else (crop, x1, y1, x2, y2)



# FRAME OVERLAY DRAWING


def draw_face_overlay(
    frame: np.ndarray,
    face_idx: int,
    emotion: str,
    confidence: float,
    x1: int, y1: int, x2: int, y2: int,
) -> None:
    # Draw bounding box + person label + emotion label + confidence in-place.
    color = PERSON_COLORS_BGR[face_idx % len(PERSON_COLORS_BGR)]
    label = PERSON_LABELS[face_idx]

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    tag = f"{label}  {emotion.upper()}"
    (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 8, y1), color, -1)
    cv2.putText(frame, tag, (x1 + 4, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)

    conf_label = f"{EMOTION_EMOJIS.get(emotion, '')}  {confidence:.0%}"
    cv2.putText(frame, conf_label, (x1 + 4, y2 + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)



# MAIN FRAME PROCESSOR


def process_frame(
    frame: np.ndarray,
    now: float,
    persons: dict,
    face_detector,
    face_mesher,
) -> np.ndarray:
    """
    Full inference cycle on one BGR frame:
      1. Detect faces
      2. Compute AU scores from landmarks
      3. Run DeepFace every ANALYZE_EVERY frames
      4. Calibrate → fuse → EWA smooth
      5. Update per-person tracking state
      6. Draw overlays

    Returns annotated BGR frame (modified in-place).
    """
    global _frame_num
    _frame_num += 1

    img_h, img_w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Landmark extraction
    mesh_res = face_mesher.process(rgb)
    lm_map: dict[int, list] = {}
    if mesh_res.multi_face_landmarks:
        for idx, face_lm in enumerate(mesh_res.multi_face_landmarks):
            lm_map[idx] = face_lm.landmark

    # Face detection
    det_res = face_detector.process(rgb)
    if not det_res.detections:
        return frame

    for face_idx, detection in enumerate(det_res.detections[:MAX_FACES]):
        face_crop, x1, y1, x2, y2 = extract_face_crop(frame, detection, img_w, img_h)
        if face_crop is None:
            continue

        # AU scores
        if face_idx in lm_map:
            au = compute_au_scores(lm_map[face_idx], img_w, img_h)
            _face_au_last[face_idx] = au
        else:
            au = _face_au_last.get(face_idx, np.ones(7, dtype=np.float32) / 7)

        # DeepFace (throttled)
        if _frame_num % ANALYZE_EVERY == 0:
            raw = run_deepface(face_crop)
            if raw is not None and np.max(raw) >= CONFIDENCE_GATE:
                calibrated = calibrate(raw)
                fused      = fuse(calibrated, au)
                if face_idx not in _face_buffers:
                    _face_buffers[face_idx] = deque(maxlen=BUFFER_LEN)
                _face_buffers[face_idx].append(fused)
                _face_last[face_idx] = ewa_update(_face_buffers[face_idx], fused)

        if face_idx not in _face_last:
            continue

        smooth     = _face_last[face_idx]
        emotion    = EMOTION_LABELS[int(np.argmax(smooth))]
        confidence = float(smooth[int(np.argmax(smooth))])

        # Ensure person record exists
        if face_idx not in persons:
            persons[face_idx] = make_person()

        person = persons[face_idx]
        person['last_probs'] = smooth.copy()

        # Emotion transition tracking
        if person['current_emotion'] != emotion:
            close_current_emotion(person, now)
            _open_emotion_segment(person, emotion, face_crop, now)
        else:
            _accumulate_running_emotion(person, emotion, now)

        draw_face_overlay(frame, face_idx, emotion, confidence, x1, y1, x2, y2)

    return frame



# CSV STORAGE


def save_session_to_csv(
    persons: dict,
    session_id: str,
    session_start: float,
    session_end: float,
) -> str:
    """
    Append one row per unique emotion per person to the CSV file.

    Columns written  (see CSV_COLUMNS in config.py):
        session_id, person_label, emotion, first_seen,
        duration_sec, dominant, session_date, session_duration_sec

    Creates the file with a header row if it doesn't exist yet.
    Returns the CSV file path.
    """
    file_exists = os.path.isfile(CSV_PATH)
    session_dur = round(session_end - session_start, 2)
    session_date = time.strftime("%Y-%m-%d", time.localtime(session_start))

    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)

        # Write header only once
        if not file_exists:
            writer.writeheader()

        for p_idx in sorted(persons.keys()):
            person   = persons[p_idx]
            dur_map  = person['duration_map']
            timeline = person['timeline']
            dominant = get_dominant_emotion(dur_map)

            for entry in timeline:
                emo = entry['emotion']
                writer.writerow({
                    "session_id":           session_id,
                    "person_label":         PERSON_LABELS[p_idx],
                    "emotion":              emo,
                    "first_seen":           entry.get("first_seen", ""),
                    "duration_sec":         round(dur_map.get(emo, 0.0), 2),
                    "dominant":             emo == dominant,
                    "session_date":         session_date,
                    "session_duration_sec": session_dur,
                })

    return CSV_PATH


def load_csv() -> "pd.DataFrame | None":
    """
    Load all saved sessions from CSV into a DataFrame.
    Returns None if the file doesn't exist or is empty.
    """
    if not os.path.isfile(CSV_PATH):
        return None
    try:
        df = pd.read_csv(CSV_PATH)
        return df if not df.empty else None
    except Exception:
        return None


def generate_session_id() -> str:
    """Generate a short unique session identifier."""
    return str(uuid.uuid4())[:8].upper()



# UTILITY




def fmt_dur(seconds: float) -> str:
    """Format seconds into a human-readable duration string."""
    if seconds < 1:
        return f"{seconds:.1f}s"
    if seconds < 60:
        return f"{int(seconds)}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s"
