# tf.random.uniform((B, 784), dtype=tf.float32) ← Input shape inferred from example: vector of length 784 (e.g. flattened 28x28 images)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Following the example model from the issue:
        # Input shape: (784,)
        # Architecture: Activation('relu') applied, then Dense(10, softmax)

        # Important: The issue highlighted that using tf.keras.activations.relu
        # directly as a layer (functional) causes serialization issues,
        # while layers.Activation('relu') works correctly.
        # So here we use layers.Activation('relu') to avoid serialization problems.

        self.relu = tf.keras.layers.Activation('relu')
        self.dense = tf.keras.layers.Dense(10, activation='softmax', name='predictions')

    def call(self, inputs):
        x = self.relu(inputs)     # Use keras Activation layer (not tf.keras.activations.relu) to avoid serialization issue
        return self.dense(x)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor of shape (batch_size, 784).
    # The example code input shape was (784,), so batch dimension is needed.

    batch_size = 32  # Default batch size for testing
    return tf.random.uniform((batch_size, 784), dtype=tf.float32)

