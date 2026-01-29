# tf.random.uniform((B, 3), dtype=tf.float32)  ← Input shape inferred as (batch_size, 3)

import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np

class BatchMeanSquaredError(tf.keras.losses.Loss):
    def __init__(self, reduction='auto', name='batch_mean_squared_error'):
        # Use 'auto' reduction for compatibility with tf.keras Loss defaults
        super().__init__(reduction=reduction, name=name)

    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        # Compute mean squared error averaged over the batch dimension axis=0,
        # this is non-standard (normally axis=-1), consistent with original code.
        L = K.mean((y_pred - y_true) ** 2, axis=0)
        return L

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define three dense layers with relu activations except last layer
        self.dense1 = tf.keras.layers.Dense(3, activation='relu')
        self.dense2 = tf.keras.layers.Dense(3, activation='relu')
        self.dense3 = tf.keras.layers.Dense(3)

        # Use the custom loss as an attribute (not for computation here,
        # but to align with issue's context)
        self.custom_loss = BatchMeanSquaredError()

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

def my_model_function():
    # Instantiate MyModel and compile using the custom loss.
    model = MyModel()
    # Compile so that usage of the custom loss is consistent with example
    model.compile(optimizer='sgd', loss=BatchMeanSquaredError())
    return model

def GetInput():
    # Generate a random tensor input with shape (batch_size, 3)
    # Assumes float32 dtype as typical for tf.keras model inputs
    batch_size = 10  # example batch size inferred from original fit(batch_size=10)
    return tf.random.uniform((batch_size, 3), dtype=tf.float32)

