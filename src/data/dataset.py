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


def _load_and_preprocess(img_path, speed, direction, grid, augment=False):
    import cv2

    frame = cv2.imread(img_path)
    if frame is None:
        return None, None, None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = preprocess(frame, add_batch_dim=False)

    if augment:
        image, speed, direction, grid = augment_sample(image, speed, direction, grid)

    return image, np.array([speed, direction], dtype=np.float32), grid


def build_dataset(csv_path=None, augment=False, batch_size=None, shuffle=True):
    # uses a generator so we only load one image at a time instead of all at once
    batch_size = batch_size or config.BATCH_SIZE
    df = load_labels(csv_path)

    def generator():
        indices = np.arange(len(df))
        if shuffle:
            np.random.shuffle(indices)
        for idx in indices:
            row = df.iloc[idx]
            img_path = str(row["image_path"])
            speed = float(row["speed"])
            direction = float(row["direction"])
            grid = np.array([float(row[f"g{r}{c}"]) for r in range(3) for c in range(3)], dtype=np.float32)

            img, nav, grd = _load_and_preprocess(img_path, speed, direction, grid, augment=augment)
            if img is None:
                continue
            yield img, {"navigation": nav, "obstacle_grid": grd}

    output_sig = (
        tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
        {
            "navigation": tf.TensorSpec(shape=(2,), dtype=tf.float32),
            "obstacle_grid": tf.TensorSpec(shape=(9,), dtype=tf.float32),
        }
    )

    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_sig)
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

    print(f"Training: {len(df_train)} samples, Validation: {len(df_val)} samples")

    train_ds = build_dataset(train_csv, augment=True, shuffle=True)
    val_ds = build_dataset(val_csv, augment=False, shuffle=False)
    return train_ds, val_ds
