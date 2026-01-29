# tf.random.set_seed does not produce reproducible results on tensorflow-macos, but tf.keras.utils.set_random_seed does
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Single Dense layer model, matching original issue example
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs):
        return self.dense(inputs)

def my_model_function():
    # Use tf.keras.utils.set_random_seed to ensure reproducibility on macOS as per issue resolution
    tf.keras.utils.set_random_seed(42)
    return MyModel()

def GetInput():
    # The original example input was a 1D tensor expanded to shape (batch_size, 1)
    # Use a shape consistent with the example: 8 samples, each with 1 feature
    import tensorflow as tf
    import numpy as np
    X = np.array([-7.0, -4.0, -1.0, 2.0, 5.0, 8.0, 11.0, 14.0], dtype=np.float32)
    # Expand dims to (8,1) to match model input shape
    return tf.convert_to_tensor(X.reshape(-1, 1))

