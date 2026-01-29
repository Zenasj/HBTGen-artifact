# tf.random.uniform((B, 1000, 1000, 3), dtype=tf.float32) ← Input shape inferred from data used in TFRecord (1000x1000 RGB images)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simple conv layer with sigmoid activation
        self.conv = tf.keras.layers.Conv2D(1, 3, activation='sigmoid')
        # Global average pooling to reduce spatial dims
        self.pool = tf.keras.layers.GlobalAveragePooling2D()

    def call(self, inputs, training=None):
        x = self.conv(inputs)
        x = self.pool(x)
        return x

def my_model_function():
    # Returns an instance of our model
    return MyModel()

def GetInput():
    # Generate a random batch of images
    # Batch size is assumed to be 3 based on the example batching in the issue
    batch_size = 3
    height = 1000
    width = 1000
    channels = 3
    # Uniform random float input, mimicking normalized image data
    return tf.random.uniform((batch_size, height, width, channels), dtype=tf.float32)

