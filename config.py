# config.py
# Configuration constants for the face recognition application.

# Recognition thresholds
T_KNOWN = 0.5
BLUR_THRESHOLD = 100.0
BLUR_MARGIN_UPGRADE = 50.0  # how much sharper than threshold to trigger upgrade

# Detection settings
DETECT_SCALE = 0.5
DETECT_EVERY_K = 5

# Tracking settings
TRACK_IOU_THRESHOLD = 0.3
MAX_TRACK_MISSED = 10

# Unknown tracker settings
# Embedding threshold for matching unknown faces to tracks (slightly higher than T_KNOWN
# to allow for more variance in unknown face tracking before registration)
UNKNOWN_EMBEDDING_THRESHOLD = 0.6

# Recognition cooldown (seconds)
RECOG_COOLDOWN = 1.5
