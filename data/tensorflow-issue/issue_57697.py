# tf.random.uniform((1, 1, 1, 1), dtype=tf.float32)
import tensorflow as tf
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Dense layer with one unit, applied after an Add operator
        self.dense = layers.Dense(1)

    def call(self, inputs):
        # inputs shape assumed (1, 1, 1, 1) to avoid TFLite shape issue
        # Add a constant tensor broadcasted to inputs shape
        x = tf.add(inputs, tf.constant(1.1, shape=[1,1,1,1]))
        # Apply dense layer, which expects the last dimension to match units
        # Since the input shape (1,1,1,1) last dim=1, Dense(1) will work
        x = self.dense(x)
        return x

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random input tensor compatible with MyModel
    # Using shape (1,1,1,1) to match what the TFLite converter expects (to avoid rank 0 scalar issues)
    return tf.random.uniform((1, 1, 1, 1), dtype=tf.float32)

