# tf.random.uniform((B, 1), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simple identity model (input shape=(None,1))
        self.dummy_layer = tf.keras.layers.Lambda(lambda x: x)  # Identity layer

    def call(self, inputs, training=None):
        # Forward pass just returns inputs unchanged
        return self.dummy_layer(inputs)


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Generate a batch of 10 samples with shape (10, 1)
    # Values are random floats, matching the example from the issue
    return tf.random.uniform((10, 1), dtype=tf.float32)

