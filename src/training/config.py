#Central config - all settings in one place.

import os

# paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
AUGMENTED_DIR = os.path.join(DATA_DIR, "augmented")
LABELS_CSV = os.path.join(DATA_DIR, "labels.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# image settings
CAMERA_RES = (640, 480)
INPUT_SHAPE = (224, 224, 3)
CROP_TOP_FRAC = 0.40  # remove top 40% of frame

# model
BACKBONE = "v2"
BACKBONE_ALPHA = 0.5
FREEZE_FRACTION = 0.5

NAV_HIDDEN = [128, 64]
OBS_HIDDEN = [64]
GRID_SIZE = 9  # 3x3 flattened
NAV_DROPOUT = 0.3
OBS_DROPOUT = 0.2

# training
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3
LR_DECAY_FACTOR = 0.5
LR_PATIENCE = 5
EARLY_STOP_PATIENCE = 10
VALIDATION_SPLIT = 0.15

# augmentation
AUG_BRIGHTNESS_LIMIT = 0.4
AUG_BLUR_LIMIT = (3, 7)
AUG_SHADOW_PROB = 0.3
AUG_FLIP_PROB = 0.5

# quantization
QUANT_ACCURACY_THRESHOLD = 0.05

# safety thresholds
SAFE_DARK_THRESHOLD = 10
SAFE_GLARE_THRESHOLD = 245
SAFE_VARIANCE_THRESHOLD = 5.0
SAFE_MIN_CONFIDENCE = 0.4
SAFE_MAX_FAILURES = 3
SAFE_OBSTACLE_THRESHOLD = 0.85

# direction: right=+1, left=-1 (per organizer)
DIR_RIGHT = +1.0
DIR_LEFT = -1.0
DIR_STRAIGHT = 0.0
