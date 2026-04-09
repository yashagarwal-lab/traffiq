#Dataset loader - reads labels.csv and builds tf.data pipelines.

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from src.training import config
from src.data.preprocess import preprocess
from src.data.augment import augment_sample


def load_labels(csv_path=None):
    csv_path = csv_path or config.LABELS_CSV
    return pd.read_csv(csv_path)


def _parse_row(row, augment=False):
    import cv2
    
    img_path = row["image_path"]
    speed = float(row["speed"])
    direction = float(row["direction"])
    grid = np.array([float(row[f"g{r}{c}"]) for r in range(3) for c in range(3)], dtype=np.float32)
    
    frame = cv2.imread(img_path)
    if frame is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = preprocess(frame, add_batch_dim=False)
    
    if augment:
        image, speed, direction, grid = augment_sample(image, speed, direction, grid)
    
    return image, np.array([speed, direction], dtype=np.float32), grid


def build_dataset(csv_path=None, augment=False, batch_size=None, shuffle=True):
    batch_size = batch_size or config.BATCH_SIZE
    df = load_labels(csv_path)
    
    images, nav_labels, grid_labels = [], [], []
    for _, row in df.iterrows():
        img, nav, grid = _parse_row(row, augment=augment)
        images.append(img)
        nav_labels.append(nav)
        grid_labels.append(grid)
    
    images = np.array(images, dtype=np.float32)
    nav_labels = np.array(nav_labels, dtype=np.float32)
    grid_labels = np.array(grid_labels, dtype=np.float32)
    
    dataset = tf.data.Dataset.from_tensor_slices(
        (images, {"navigation": nav_labels, "obstacle_grid": grid_labels})
    )
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(images))
    
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def split_dataset(csv_path=None):
    df = load_labels(csv_path)
    val_size = int(len(df) * config.VALIDATION_SPLIT)
    
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df_val = df_shuffled[:val_size]
    df_train = df_shuffled[val_size:]
    
    train_csv = os.path.join(config.DATA_DIR, "train_split.csv")
    val_csv = os.path.join(config.DATA_DIR, "val_split.csv")
    df_train.to_csv(train_csv, index=False)
    df_val.to_csv(val_csv, index=False)
    
    train_ds = build_dataset(train_csv, augment=True, shuffle=True)
    val_ds = build_dataset(val_csv, augment=False, shuffle=False)
    return train_ds, val_ds
