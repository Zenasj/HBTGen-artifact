# tf.random.uniform((B, 5, 16), dtype=tf.float32) ← input shape inferred from the original code (batch size unknown)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # The model performs slicing on axis=1 from index 3 onward, then Conv1D with 2 filters and kernel size 1
        self.conv1d = tf.keras.layers.Conv1D(2, kernel_size=1)

    def call(self, inputs):
        # Slice input as inp[:, 3:, :] in the original code, i.e. take last 2 time steps (indices 3 and 4 of length 5)
        x = inputs[:, 3:, :]
        x = self.conv1d(x)
        return x

def my_model_function():
    # Return an instance of MyModel, weights uninitialized (random initialization)
    return MyModel()

def GetInput():
    # Return a random tensor with batch size 1 (arbitrary choice), shape (1, 5, 16), dtype float32
    # Matches expected input for MyModel (B, 5, 16)
    return tf.random.uniform((1, 5, 16), dtype=tf.float32)

