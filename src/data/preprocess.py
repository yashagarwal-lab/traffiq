#Preprocessing pipeline - crop, enhance, normalize.

import cv2
import numpy as np
from src.training import config


def crop_roi(frame):
    # Crop the top portion of the frame (ceiling/sky is useless).
    h = frame.shape[0]
    crop_y = int(h * config.CROP_TOP_FRAC)
    return frame[crop_y:, :]


def apply_clahe(frame):
    # Apply CLAHE for consistent contrast under different lighting.
    lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def preprocess(frame, add_batch_dim=True):
    # Full preprocessing: crop -> resize -> CLAHE -> normalize to [-1,1].
    h, w = config.INPUT_SHAPE[:2]
    
    img = crop_roi(frame)
    img = cv2.resize(img, (w, h))
    img = apply_clahe(img)
    img = img.astype(np.float32) / 127.5 - 1.0
    
    if add_batch_dim:
        img = np.expand_dims(img, 0)
    return img
