# tf.random.uniform((B, 360, 640, 3), dtype=tf.uint8)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Replicating the simple conv net as in the issue:
        # Conv2D with 64 filters, 3x3 kernel, ReLU activation
        # Flatten layer
        # Dense 128 with ReLU
        # Output Dense 2 with softmax
        self.conv = tf.keras.layers.Conv2D(
            64, kernel_size=3, activation=tf.nn.relu, input_shape=(360, 640, 3))
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(2, activation=tf.nn.softmax)

    def call(self, inputs):
        # Inputs assumed to be uint8 images with shape (B, 360, 640, 3)
        # Convert to float32 for the network after scaling to [0,1]
        x = tf.cast(inputs, tf.float32) / 255.0
        x = self.conv(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)

def my_model_function():
    # Return an instance of the simple CNN model
    return MyModel()

def GetInput():
    # Return a batch of random uint8 images with batch size 2, height 360, width 640, channels 3
    # Chose batch size 2 for demonstration and reasonable memory use
    return tf.random.uniform((2, 360, 640, 3), maxval=256, dtype=tf.int32).numpy().astype('uint8')

