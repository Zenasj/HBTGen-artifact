# tf.random.normal((B=5, H=3), dtype=tf.float32) ← inferred input shape (batch size 5, feature dimension 3)

import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow import math, dtypes
from tensorflow.keras.optimizers import Adam

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simple two-layer dense network resembling the original Sequential structure
        self.dense1 = layers.Dense(3)
        self.dense2 = layers.Dense(3)

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return x

def MMSE_loss(y_true, y_pred, mask_value=0.0):
    """
    Masked Mean Squared Error loss function with signature (y_true, y_pred),
    applying mask to ignore positions where target == mask_value.

    This implements the correction described in the issue:
    The correct order of args is (y_true, y_pred).
    """
    mask = dtypes.cast(tf.not_equal(y_true, mask_value), tf.float32)
    num_valid = math.reduce_sum(mask)
    # Avoid division by zero if no valid targets
    num_valid = tf.maximum(num_valid, 1.0)
    diff = mask * (y_pred - y_true)
    loss = math.reduce_sum(tf.square(diff)) / num_valid
    return loss

def my_model_function():
    """
    Returns an instance of MyModel compiled with Adam optimizer and MMSE_loss,
    mimicking the example in the issue.
    """
    model = MyModel()
    # Compile with Adam and the custom masked MSE loss expecting (y_true, y_pred) order
    model.compile(optimizer=Adam(learning_rate=0.01), loss=MMSE_loss)
    return model

def GetInput():
    """
    Returns a random tensor input of shape (5,3) matching the batch and feature dims used in example.
    Using tf.random.normal and rounding to approximate the example data.
    """
    data = tf.math.round(tf.random.normal(shape=[5, 3]))
    return data

