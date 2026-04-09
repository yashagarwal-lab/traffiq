#Augmentation pipeline using albumentations.

import albumentations as A
import numpy as np
from src.training import config


def get_augmentation_pipeline():
    return A.Compose([
        A.RandomBrightnessContrast(brightness_limit=config.AUG_BRIGHTNESS_LIMIT, contrast_limit=0.3, p=0.7),
        A.GaussianBlur(blur_limit=config.AUG_BLUR_LIMIT, p=0.3),
        A.GaussNoise(std_range=(0.02, 0.1), p=0.4),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=20, p=0.4),
        A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_limit=(1, 3), shadow_dimension=5, p=config.AUG_SHADOW_PROB),
        A.Rotate(limit=5, border_mode=0, p=0.3),
    ])


def augment_sample(image, speed, direction, grid):
    # Augment one sample. Handles flip specially - negate direction and mirror grid.
    pipeline = get_augmentation_pipeline()
    
    if image.dtype != np.uint8:
        img = ((image + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    else:
        img = image
    
    augmented = pipeline(image=img)
    aug_img = augmented["image"]
    
    # random horizontal flip - need to also flip labels
    if np.random.random() < config.AUG_FLIP_PROB:
        aug_img = np.fliplr(aug_img).copy()
        direction = -direction  # left becomes right
        grid_3x3 = grid.reshape(3, 3)
        grid_3x3 = grid_3x3[:, ::-1]  # mirror columns
        grid = grid_3x3.flatten()
    
    if image.dtype != np.uint8:
        aug_img = (aug_img.astype(np.float32) / 127.5) - 1.0
    
    return aug_img, speed, direction, grid
