#Evaluation - loss curves and prediction plots.

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from src.training import config


def plot_training_curves(history_path, save_dir=None):
    with open(history_path) as f:
        history = json.load(f)
    
    save_dir = save_dir or os.path.dirname(history_path)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    axes[0, 0].plot(history["loss"], label="train")
    axes[0, 0].plot(history["val_loss"], label="val")
    axes[0, 0].set_title("Total Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    if "navigation_mae" in history:
        axes[0, 1].plot(history["navigation_mae"], label="train")
        axes[0, 1].plot(history["val_navigation_mae"], label="val")
        axes[0, 1].set_title("Navigation MAE")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    if "navigation_loss" in history:
        axes[1, 0].plot(history["navigation_loss"], label="train")
        axes[1, 0].plot(history["val_navigation_loss"], label="val")
        axes[1, 0].set_title("Nav Loss (MSE)")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    if "obstacle_grid_accuracy" in history:
        axes[1, 1].plot(history["obstacle_grid_accuracy"], label="train")
        axes[1, 1].plot(history["val_obstacle_grid_accuracy"], label="val")
        axes[1, 1].set_title("Grid Accuracy")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Training curves saved to: {plot_path}")


def evaluate_model(model, val_dataset, save_dir=None):
    save_dir = save_dir or config.MODELS_DIR
    os.makedirs(save_dir, exist_ok=True)
    
    all_nav_true, all_nav_pred = [], []
    for images, labels in val_dataset:
        nav_pred, obs_pred = model.predict(images, verbose=0)
        all_nav_true.append(labels["navigation"].numpy())
        all_nav_pred.append(nav_pred)
    
    nav_true = np.concatenate(all_nav_true)
    nav_pred = np.concatenate(all_nav_pred)
    
    speed_mae = np.mean(np.abs(nav_true[:, 0] - nav_pred[:, 0]))
    dir_mae = np.mean(np.abs(nav_true[:, 1] - nav_pred[:, 1]))
    
    print(f"Speed MAE: {speed_mae:.4f}")
    print(f"Direction MAE: {dir_mae:.4f}")
    
    return {"speed_mae": float(speed_mae), "direction_mae": float(dir_mae)}
