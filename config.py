"""
config.py
=========
All shared constants, labels, colors, and model hyperparameters.
Import this in every other module — never hardcode values elsewhere.
"""

import numpy as np

# ─────────────────────────────────────────────────────────
# EMOTION METADATA
# ─────────────────────────────────────────────────────────

EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
DISPLAY_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

EMOTION_EMOJIS = {
    'angry':    '😠',
    'disgust':  '🤢',
    'fear':     '😨',
    'happy':    '😄',
    'sad':      '😢',
    'surprise': '😲',
    'neutral':  '😐',
}

EMOTION_COLORS = {
    'angry':    '#ef4444',
    'disgust':  '#a855f7',
    'fear':     '#f59e0b',
    'happy':    '#22c55e',
    'sad':      '#3b82f6',
    'surprise': '#06b6d4',
    'neutral':  '#6b7280',
}

# ─────────────────────────────────────────────────────────
# PERSON TRACKING
# ─────────────────────────────────────────────────────────

MAX_FACES         = 4
PERSON_LABELS     = ['Person A', 'Person B', 'Person C', 'Person D']

PERSON_COLORS_HEX = ['#f97316', '#06b6d4', '#a855f7', '#4ade80']

# BGR colors for OpenCV overlays (R and B swapped vs hex)
PERSON_COLORS_BGR = [
    (34,  115, 249),
    (212, 182,   6),
    (168,  85, 168),
    (128, 222,  78),
]

# ─────────────────────────────────────────────────────────
# INFERENCE HYPERPARAMETERS
# ─────────────────────────────────────────────────────────

BUFFER_LEN       = 10
EWA_ALPHA        = 0.45
CONFIDENCE_GATE  = 0.20
ANALYZE_EVERY    = 2
PADDING_RATIO    = 0.28
AU_FUSION_WEIGHT = 0.35
TEMPERATURE      = 1.6

PRIOR_CORRECTION = np.array(
    [1.55, 2.80, 1.40, 0.85, 1.10, 1.00, 0.80],
    dtype=np.float32
)

# ─────────────────────────────────────────────────────────
# CSV STORAGE
# ─────────────────────────────────────────────────────────

CSV_PATH = "emotion_sessions.csv"

# Columns written to CSV
CSV_COLUMNS = [
    "session_id",       # unique ID per START→STOP cycle
    "person_label",     # Person A / B / C / D
    "emotion",          # emotion label
    "first_seen",       # HH:MM:SS timestamp string
    "duration_sec",     # total seconds this emotion was held
    "dominant",         # True if this was the dominant emotion for that person
    "session_date",     # YYYY-MM-DD
    "session_duration_sec",  # total session length in seconds
]
