# tf.random.uniform((B, 1), dtype=tf.float32)  ← input_shape=(1,) based on issue example

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Replicating the original simple network:
        # Dense(256), relu, Dense(2), softmax
        self.dense1 = tf.keras.layers.Dense(256, activation='relu', input_shape=(1,))
        self.dense2 = tf.keras.layers.Dense(2, activation='softmax')
        # BinaryCrossentropy loss instance with default settings
        # from the issue: same as keras.losses.BinaryCrossentropy()
        self.loss_fn = tf.keras.losses.BinaryCrossentropy()

    def call(self, x, training=False):
        # Forward pass
        logits = self.dense1(x)
        probs = self.dense2(logits)
        return probs

    def compute_losses(self, y_true, y_pred):
        # This function reproduces the issue's comparison of losses:
        # 1. Loss as computed by loss function call (self.loss_fn)
        loss_func_value = self.loss_fn(y_true, y_pred)

        # 2. Manual binary cross-entropy computed element-wise
        #    BCE = - [ y*log(p) + (1-y)*log(1-p) ]
        #    averaged over batch & classes.
        manual_loss = tf.reduce_mean(- (y_true * tf.math.log(y_pred + 1e-7) + (1 - y_true) * tf.math.log(1 - y_pred + 1e-7)))

        # This aligns with the manual calculation printed in the issue.
        return loss_func_value, manual_loss

def my_model_function():
    return MyModel()

def GetInput():
    # Based on the example in the issue:
    # input is shape (batch=1, features=1), e.g. np.array([0.4])
    # Outputs a single batch with one float feature.
    # Use tf.float32 dtype for compatibility with model weights (default).
    return tf.random.uniform((1, 1), minval=0.0, maxval=1.0, dtype=tf.float32)

