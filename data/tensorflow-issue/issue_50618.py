# tf.random.uniform((B, 1, 1, 5), dtype=tf.float32) ← inferred input shape based on example shape [[[1,2,3,4,5]]]

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # The example uses a ReLU layer with threshold=3 and input_shape=(2,)
        # But the input shape in code is (1,1,5), which doesn't match (2,)
        # We'll reconcile it by implementing a ReLU threshold=3 applied element-wise,
        # assuming inputs shaped (B, 1, 1, 5).
        # This matches tf.keras.layers.ReLU(threshold=3) behavior.
        self.relu = tf.keras.layers.ReLU(threshold=3.0)

    def call(self, inputs, training=False):
        # inputs shape: (B, 1, 1, 5) float32 tensor expected
        # Apply ReLU threshold=3 element-wise
        return self.relu(inputs)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input matching expected shape (B,1,1,5)
    # We'll use batch size = 2 as a reasonable default for testing
    B = 2
    H = 1
    W = 1
    C = 5
    # Uniform values in 0-5 range to span both below and above ReLU threshold 3
    return tf.random.uniform((B, H, W, C), minval=0.0, maxval=5.0, dtype=tf.float32)

