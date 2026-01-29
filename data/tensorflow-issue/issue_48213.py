# tf.random.uniform((B, None, 5), dtype=tf.float32) and tf.random.uniform((B, 1), dtype=tf.float32)
import tensorflow as tf
from tensorflow.linalg import matmul
import numpy as np

class MinimalRNNCell(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        super(MinimalRNNCell, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape: (batch, timesteps, input_dim), but this build is called per timestep: input_shape[-1]
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='uniform',
                                      name='kernel')
        self.recurrent_kernel = self.add_weight(shape=(self.units, self.units),
                                                initializer='uniform',
                                                name='recurrent_kernel')
        self.built = True

    def call(self, inputs, states=None, constants=None, *args, **kwargs):
        """
        inputs: (batch_size, input_dim)
        states: list with previous output state tensor(s)
        constants: list of constant tensors passed in via RNN layer call
        """
        prev_output = states[0]
        # constants[0] shape: (batch_size, 1)
        # matmul(inputs, kernel) shape: (batch_size, units)
        # Add constants[0] directly (broadcast) since shape (batch_size, 1)
        # Matches units?
        # According to original error, dimensions mismatched, but this is how original code was.
        h = matmul(inputs, self.kernel) + constants[0]  # Broadcast constants over units axis
        output = h + matmul(prev_output, self.recurrent_kernel)
        return output, [output]

    def get_config(self):
        return dict(super().get_config(), **{'units': self.units})

# To workaround the known bug with default tf.keras.layers.RNN shape inference for custom cells,
# define a subclass of RNN without modifications.
class RNN(tf.keras.layers.RNN):
    pass

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.cell = MinimalRNNCell(32)
        self.rnn_layer = RNN(self.cell, name='rnn')

    def call(self, inputs):
        """
        Inputs: a list/tuple of two tensors:
          - inputs[0]: input sequence tensor of shape (batch_size, timesteps, 5)
          - inputs[1]: constants tensor of shape (batch_size, 1)
        Returns:
          - output tensor from RNN: shape (batch_size, units)
        """
        x, z = inputs
        # Pass constants argument as list with single tensor
        y = self.rnn_layer(x, constants=[z])
        return y

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Create batch_size B=1 for sanity; sequence length 1 (timesteps); input_dim=5 per original example
    B = 1
    timesteps = 1
    input_dim = 5
    # Generate input tensor shape (B, timesteps, input_dim)
    x = tf.random.uniform((B, timesteps, input_dim), dtype=tf.float32)
    # Generate constants tensor shape (B, 1)
    z = tf.random.uniform((B, 1), dtype=tf.float32)
    return [x, z]

