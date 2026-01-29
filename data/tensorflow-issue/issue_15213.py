# tf.random.uniform((B, H, W, C), dtype=tf.float32) ← input shape is ambiguous as no specific model or shape is described in issue content
import tensorflow as tf

class MyModel(tf.keras.Model):
    """
    Since the issue discusses XLA/AOT compilation on Windows but does not specify
    any particular model architecture, this model serves as a minimal placeholder
    compatible with tf.function jit_compile=True.

    It simply applies a single Dense layer to flatten and project input features,
    demonstrating a minimal meaningful forward pass.
    """

    def __init__(self):
        super().__init__()
        # Assume input is at least 4D - batch, height, width, channels (typical image-like tensor)
        # We'll flatten and pass through a simple Dense layer.
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(10, activation='relu')

    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        x = self.dense(x)
        return x

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    """
    Return a random input tensor compatible with MyModel.
    Based on typical image input assumption: batch_size=8, height=28, width=28, channels=1 (like MNIST)
    dtype=float32 as standard for models.
    """
    B, H, W, C = 8, 28, 28, 1
    return tf.random.uniform((B, H, W, C), dtype=tf.float32)

