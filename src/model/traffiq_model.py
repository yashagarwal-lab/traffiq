#Model architecture - MobileNetV2/V3 backbone with two output heads.

import tensorflow as tf
from src.model.backbone import build_backbone
from src.training import config


def build_traffiq_model(backbone_type=None):
    # Build the full model with navigation + obstacle grid heads.
    backbone_type = backbone_type or config.BACKBONE
    
    inp = tf.keras.Input(shape=config.INPUT_SHAPE, name="input_image")
    backbone = build_backbone(backbone_type)
    
    features = backbone(inp)
    pooled = tf.keras.layers.GlobalAveragePooling2D()(features)
    
    # navigation head -> [speed, direction]
    x = pooled
    for units in config.NAV_HIDDEN:
        x = tf.keras.layers.Dense(units, activation="relu")(x)
    x = tf.keras.layers.Dropout(config.NAV_DROPOUT)(x)
    nav_out = tf.keras.layers.Dense(2, activation="tanh", name="navigation")(x)
    
    # obstacle grid head -> 3x3 probabilities
    y = pooled
    for units in config.OBS_HIDDEN:
        y = tf.keras.layers.Dense(units, activation="relu")(y)
    y = tf.keras.layers.Dropout(config.OBS_DROPOUT)(y)
    grid_out = tf.keras.layers.Dense(config.GRID_SIZE, activation="sigmoid", name="obstacle_grid")(y)
    
    model = tf.keras.Model(inputs=inp, outputs=[nav_out, grid_out], name="traffiq")
    return model
