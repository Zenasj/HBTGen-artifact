# tf.random.uniform((1, 5), dtype=tf.float32)
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Matching the original model structure: input shape (1,5)
        # Sequential Equivalent:
        # Dense(10, sigmoid) -> Dense(2, linear)
        self.dense1 = layers.Dense(10, activation='sigmoid', name="dense_sigmoid")
        self.dense2 = layers.Dense(2, activation='linear', name="dense_linear")

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return x

def my_model_function():
    # Returns a freshly initialized MyModel instance.
    # Since the original code compiled with loss and optimizer, we add compile here to mimic training setup.
    model = MyModel()
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    return model

def GetInput():
    # Input expected is a batch with shape (1, 5), dtype float32.
    # The original code uses a one-hot identity vector of length 5 as input representing states.
    # So create a random one-hot like vector batch with shape (1, 5)
    # We create a random integer in [0..4] and convert to one-hot vector
    batch_index = 1
    num_states = 5
    # Random choice of state index
    state_idx = np.random.randint(0, num_states)
    # Create one-hot input as float32 tensor
    input_array = np.eye(num_states, dtype=np.float32)[state_idx:state_idx+1]
    # Convert to tf.Tensor
    input_tensor = tf.convert_to_tensor(input_array)
    return input_tensor

