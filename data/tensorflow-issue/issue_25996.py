# tf.random.uniform((B, 20), dtype=tf.float32)  ← inferred input shape from original example (batch size B, feature vector of length 20)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define a simple Dense network matching the example in the issue
        self.dense = tf.keras.layers.Dense(2)

    def call(self, inputs, training=False):
        # Forward pass
        x = self.dense(inputs)
        return x

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    # Shape: (batch_size, feature_length) e.g. (32,20)
    batch_size = 32  # arbitrary batch size
    feature_length = 20
    # Use float32 for compatibility with Dense layer weights
    return tf.random.uniform((batch_size, feature_length), dtype=tf.float32)

