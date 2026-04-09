#Training script.

import os
import argparse
import json
import numpy as np
import tensorflow as tf
from datetime import datetime

from src.model.traffiq_model import build_traffiq_model
from src.data.dataset import split_dataset
from src.training import config


def train(backbone_variant=None, csv_path=None):
    backbone_variant = backbone_variant or config.BACKBONE
    
    print(f"Training with backbone: {backbone_variant}")
    print(f"GPU: {tf.config.list_physical_devices('GPU')}")
    
    model = build_traffiq_model(backbone_variant)
    model.summary()
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss={
            "navigation": tf.keras.losses.MeanSquaredError(),
            "obstacle_grid": tf.keras.losses.BinaryCrossentropy(),
        },
        loss_weights={"navigation": 1.0, "obstacle_grid": 0.5},
        metrics={"navigation": ["mae"], "obstacle_grid": ["accuracy"]},
    )
    
    train_ds, val_ds = split_dataset(csv_path)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(config.MODELS_DIR, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(run_dir, "best_model.keras"),
            monitor="val_loss", save_best_only=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=config.LR_DECAY_FACTOR,
            patience=config.LR_PATIENCE, min_lr=1e-6, verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=config.EARLY_STOP_PATIENCE,
            restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(log_dir=os.path.join(run_dir, "logs")),
    ]
    
    history = model.fit(
        train_ds, validation_data=val_ds,
        epochs=config.EPOCHS, callbacks=callbacks, verbose=1,
    )
    
    # save model and history
    final_path = os.path.join(run_dir, "final_model.keras")
    model.save(final_path)
    print(f"\nFinal model saved to: {final_path}")
    
    history_path = os.path.join(run_dir, "history.json")
    hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(history_path, "w") as f:
        json.dump(hist_dict, f, indent=2)
    print(f"Training history saved to: {history_path}")
    
    # save config snapshot
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({
            "backbone": backbone_variant, "batch_size": config.BATCH_SIZE,
            "epochs": config.EPOCHS, "learning_rate": config.LEARNING_RATE,
            "input_shape": list(config.INPUT_SHAPE),
        }, f, indent=2)
    
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, default=None, choices=["v3_small", "v2"])
    parser.add_argument("--csv", type=str, default=None)
    args = parser.parse_args()
    train(args.backbone, args.csv)
