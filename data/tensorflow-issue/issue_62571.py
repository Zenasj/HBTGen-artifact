# tf.random.uniform((B, H, W, C), dtype=tf.float32) ← Assumed input shape for testing: batch size 4, height 32, width 32, channels 3

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Conv2D layer with 10 filters, kernel size 3, activation relu
        self.conv = tf.keras.layers.Conv2D(10, kernel_size=3, activation="relu")
        # GlobalAveragePooling2D layer known to cause freezing on multi-GPU in certain TF versions
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        # Dense layer with 10 units and relu activation
        self.dense = tf.keras.layers.Dense(10, activation="relu")

        # For comparison, an alternative Flatten-based model to contrast behavior
        self.flatten = tf.keras.layers.Flatten()
        self.dense_alt = tf.keras.layers.Dense(10, activation="relu")

    def call(self, inputs, training=False):
        # Primary path (the one that causes freezing on multi-GPU according to the issue)
        x1 = self.conv(inputs)
        x1 = self.global_avg_pool(x1)
        out1 = self.dense(x1)

        # Alternative path (Flatten instead of GlobalAveragePooling2D)
        x2 = self.conv(inputs)
        x2 = self.flatten(x2)
        out2 = self.dense_alt(x2)

        # Compare outputs numerically with a tolerance and output boolean indicating approximate equality
        # The forward output is a boolean tensor of shape (batch_size, 10)
        # True means outputs are close within atol=1e-5 and rtol=1e-3, False otherwise
        # This responds to the "fuse them and compare" requirement if multiple models discussed
        comparison = tf.math.abs(out1 - out2) < 1e-3 + 1e-5 * tf.math.abs(out2)
        return comparison

def my_model_function():
    # Returns an instance of MyModel with initialized layers and weights
    return MyModel()

def GetInput():
    # Return a random uniform float32 tensor shaped like a typical image batch (B=4, H=32, W=32, C=3)
    # This input shape matches Conv2D input expectations and fits with the model definition
    batch_size = 4
    height = 32
    width = 32
    channels = 3
    return tf.random.uniform((batch_size, height, width, channels), dtype=tf.float32)

