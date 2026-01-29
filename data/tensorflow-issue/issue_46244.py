# tf.random.uniform((1, 5, 1), dtype=tf.float32) ← Input shape is (batch=1, timesteps=5, features=1)

import tensorflow as tf
from tensorflow import keras

class AddsLossLayer(keras.layers.Layer):
    """Identity layer which calls add_loss on mean of input (shape: batch x features)"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        # inputs shape: (batch_size, features)
        # Add to losses the mean over features dimension (axis=1 if inputs was (batch, features))
        # Here inputs shape is (batch, features), reduce mean over last axis
        loss = tf.reduce_mean(inputs, axis=-1)
        # Note: original code uses reduce_mean over inputs (batch x ninputs),
        # but in TD layer, call is run on timestep slices: shape (batch, features),
        # so axis=1 or -1 is features axis.
        # add_loss expects scalar or batch-wise loss; add_loss aggregates losses.
        self.add_loss(loss)
        return inputs

    def compute_output_shape(self, input_shape):
        return input_shape

class MyModel(tf.keras.Model):
    """
    Model wrapping the AddsLossLayer inside TimeDistributed.
    This reproduces the behavior from the issue:
    - Inputs shape: (batch, timesteps, features)
    - AddsLossLayer is applied per timestep with TimeDistributed
    - Outputs are same shape as inputs
    """

    def __init__(self):
        super().__init__()
        self.adds_loss_layer = AddsLossLayer()
        self.td_layer = keras.layers.TimeDistributed(self.adds_loss_layer, name="td")

    def call(self, inputs):
        # inputs shape: (batch, timesteps, features)
        # Run TimeDistributed AddsLossLayer
        return self.td_layer(inputs)

def my_model_function():
    # Return an instance of MyModel 
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel:
    # Shape: (batch=1, timesteps=5, features=1)
    # As in the issue: inputs at each timestep = timestep + 1
    batch, timesteps, features = 1, 5, 1
    # Create a float32 tensor with values [[1], [2], [3], [4], [5]] per timestep, repeated for batch
    import numpy as np
    timevec = np.arange(timesteps, dtype=np.float32) + 1  # [1., 2., 3., 4., 5.]
    inputs_np = np.broadcast_to(timevec[np.newaxis, :, np.newaxis], (batch, timesteps, features))
    return tf.convert_to_tensor(inputs_np, dtype=tf.float32)

