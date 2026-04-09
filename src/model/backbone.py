#Backbone factory - builds MobileNetV3 or V2 with frozen layers.

import tensorflow as tf
from src.training import config


def build_backbone(backbone_type="v2"):
    # Load pretrained MobileNet and freeze early layers.
    
    if backbone_type == "v3_small":
        base = tf.keras.applications.MobileNetV3Small(
            input_shape=config.INPUT_SHAPE,
            include_top=False,
            weights="imagenet",
            minimalistic=True,
        )
    else:
        base = tf.keras.applications.MobileNetV2(
            input_shape=config.INPUT_SHAPE,
            include_top=False,
            weights="imagenet",
            alpha=config.BACKBONE_ALPHA,
        )
    
    # freeze first half of layers (keep low-level features fixed)
    freeze_up_to = int(len(base.layers) * config.FREEZE_FRACTION)
    for layer in base.layers[:freeze_up_to]:
        layer.trainable = False
    
    return base
