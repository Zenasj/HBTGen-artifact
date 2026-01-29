# tf.random.uniform((1, 10, 64), dtype=tf.float32) ← Input shape inferred from issue: batch_size=1, sequence_length=10, input_size=64

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # GRU as described in the issue snippet: units=16, input shape (batch=1, 10, 64)
        # No explicit batch_first because Keras expects (batch, timesteps, features)
        self.gru = tf.keras.layers.GRU(units=16)
    
    def call(self, inputs, training=False):
        # Simply run the GRU layer and return the output
        # The output is the last output state (shape: [batch_size, 16])
        return self.gru(inputs, training=training)


def my_model_function():
    # Returns an instance of the defined MyModel
    # No weights provided or loaded since issue snippet did not mention pretrained weights
    return MyModel()


def GetInput():
    # Generate a random input tensor matching the expected input shape (batch=1, timesteps=10, features=64)
    # Using float32 uniform in [-1, 1], similar to the dataset generator in the issue
    return tf.random.uniform(shape=(1, 10, 64), minval=-1.0, maxval=1.0, dtype=tf.float32)

