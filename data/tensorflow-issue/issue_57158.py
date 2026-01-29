# tf.random.uniform((B, 20, 20, 3), dtype=tf.uint8)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # A simple Conv2D model similar to the example in the issue
        self.conv = tf.keras.layers.Conv2D(
            filters=3,
            kernel_size=3,
            activation="relu",
            padding="same"
        )
        # Optional Flatten and Dense layer to produce a scalar output,
        # since the issue compared both 3D (Conv2D output) and scalar outputs.
        # We'll keep only the Conv2D output as the main output to reflect issue scenario.
        # If needed, these can be uncommented for testing 1D scalar outputs.
        # self.flatten = tf.keras.layers.Flatten()
        # self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs):
        # Forward pass, simply apply conv layer
        x = self.conv(inputs)
        # To reproduce the exact issue context: output is 4D tensor (B, H, W, filters)
        return x

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input matching the expected input shape (B, 20, 20, 3)
    # and dtype uint8 to mirror the original issue
    batch_size = 16  # batch size used in failing example
    input_tensor = tf.random.uniform(
        shape=(batch_size, 20, 20, 3),
        minval=0,
        maxval=256,
        dtype=tf.uint8
    )
    return input_tensor

