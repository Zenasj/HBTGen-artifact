# tf.random.uniform((B, 120, 2), dtype=tf.float32)  # B = batch size, 120 timesteps (5 sec * 60 Hz), 2 channels (rri & ampl)

import tensorflow as tf
from tensorflow.keras import layers

# Since original data input shape is (batch, 120, 2)
# The two channels correspond to interpolated and normalized RRI and amplitude signals
# From the issue context, the original model likely expects (None, 120, 2) inputs

# The user’s issue revolves around a Lambda layer causing NameErrors when loading due to
# undefined functions. To resolve this in a robust way for TensorFlow 2.x,
# we encapsulate any custom lambda behavior as a method inside the model.

# We'll create a MyModel class with:
# - An example Lambda layer implemented as a method (doing identity here since original func unknown)
# - A basic architecture that could resemble something used for apnea detection on this input shape
# - For demonstration, a simple Conv1D + Dense classifier as a plausible structure

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # We'll assume the original model used some Lambda layer with custom padding or scaling
        # Here we define an identity lambda (placeholder), to prevent NameErrors
        # In practice, you would replace this with the actual pre-processing logic
        self.lambda_layer = layers.Lambda(self._custom_lambda, name="custom_lambda")

        # Sample stack of layers similar to an apnea detection model given 1D time series input
        self.conv1 = layers.Conv1D(filters=32, kernel_size=5, activation='relu', padding='same')
        self.conv2 = layers.Conv1D(filters=64, kernel_size=5, activation='relu', padding='same')
        self.global_pool = layers.GlobalAveragePooling1D()
        self.dense1 = layers.Dense(64, activation='relu')
        self.out = layers.Dense(2, activation='softmax')  # binary classification output
        
    def _custom_lambda(self, x):
        # Placeholder for the original Lambda function that caused the NameError.
        # For example, this could be symmetric padding or scaling.
        # Here: just pass x through as is to keep shape consistent.
        return x

    def call(self, inputs, training=False):
        x = self.lambda_layer(inputs)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.global_pool(x)
        x = self.dense1(x)
        return self.out(x)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor matching the expected input: (batch, 120, 2)
    # Batch size chosen arbitrarily as 8 for example
    batch_size = 8
    input_shape = (120, 2)
    # Use uniform random floats in [0,1), dtype float32 (matching normalized signal data)
    return tf.random.uniform((batch_size,) + input_shape, dtype=tf.float32)

