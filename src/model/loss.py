#Loss functions for multi-task training.

import tensorflow as tf


class AdaptiveMultiTaskLoss(tf.keras.layers.Layer):
    # Learns how to weight the two losses automatically.
    # Based on the idea of using uncertainty to balance tasks.
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # learnable log-variance for each task
        self.log_var_nav = self.add_weight(name="log_var_nav", initializer="zeros")
        self.log_var_obs = self.add_weight(name="log_var_obs", initializer="zeros")
    
    def call(self, nav_loss, obs_loss):
        # higher uncertainty = lower weight (model is unsure, don't penalize as hard)
        weighted_nav = nav_loss / (2 * tf.exp(self.log_var_nav)) + self.log_var_nav / 2
        weighted_obs = obs_loss / (2 * tf.exp(self.log_var_obs)) + self.log_var_obs / 2
        return weighted_nav + weighted_obs
