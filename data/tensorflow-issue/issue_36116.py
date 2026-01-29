# tf.random.uniform((B, 540, 960, 3), dtype=tf.float32) ← Inferred input shape for each input image tensor

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        # Two conv layers as per the example code
        self.conv1 = tf.keras.layers.Conv2D(
            filters=2, kernel_size=(7, 7), strides=(2, 2), padding="same", name="conv1"
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=1, kernel_size=(7, 7), strides=(2, 2), padding="same", name="conv2"
        )

    def call(self, inputs):
        # inputs is a tuple of two tensors (image_l, image_r), each shape (B, 540, 960, 3)
        # Concatenate along channel axis as original model does
        x = tf.concat(inputs, axis=-1, name="concat")  # shape (B, 540, 960, 6)
        x = self.conv1(x)  # e.g. shape (B, 270, 480, 2)
        y = self.conv2(x)  # e.g. shape (B, 135, 240, 1)
        # Return list of outputs with different resolutions (matching user's usage)
        return [x, y]

def my_model_function():
    # Instantiate and return MyModel
    return MyModel()

def GetInput():
    # Return a tuple of two random float32 tensors with shape (batch, height, width, channels)
    # Normalized between 0 and 1 to resemble images, batch size = 1 for simplicity
    img_l = tf.random.uniform(shape=(1, 540, 960, 3), dtype=tf.float32)
    img_r = tf.random.uniform(shape=(1, 540, 960, 3), dtype=tf.float32)
    return (img_l, img_r)

