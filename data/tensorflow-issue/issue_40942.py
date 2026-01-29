# tf.random.uniform((batch_size=8, height=512, width=512, channels=1), dtype=tf.uint16)

import tensorflow as tf

# This model is reconstructed from the issue: it processes single-channel 512x512 inputs,
# normalizes input dividing by 5000.0, applies a 3x3 Conv2D with sigmoid activation.

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.max_signal = 5000.0
        self.norm_layer = tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32) / self.max_signal)
        self.conv = tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')

    def call(self, inputs, training=False):
        x = self.norm_layer(inputs)
        x = self.conv(x)
        return x

def my_model_function():
    # Return an instance of MyModel.
    return MyModel()

def GetInput():
    # The original training input has shape (K, 512, 512, 1) with uint16 values [0..5000].
    # The batch size used during training was 8; here we produce a single batch of input.
    # Using same input dtype uint16, values in [0..5000].
    batch_size = 8
    height = 512
    width = 512
    channels = 1
    max_signal = 5000

    # Generate random input tensor matching training input characteristics.
    input_tensor = tf.random.uniform(
        shape=(batch_size, height, width, channels),
        minval=0,
        maxval=max_signal + 1,
        dtype=tf.dtypes.uint16
    )
    return input_tensor

