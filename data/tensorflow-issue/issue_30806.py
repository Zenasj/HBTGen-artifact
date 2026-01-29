# tf.random.uniform((B, 28, 28, 3), dtype=tf.float32)

import tensorflow as tf
from tensorflow import keras

# Custom 2D upsampling function as a tf.keras layer-compatible Lambda
def UpSamplingCustom2D(scale=(2, 2)):
    if isinstance(scale, int):
        scale = (scale, scale)

    def upsampling(x):
        # x shape is expected to be [batch, height, width, channels]
        shape = tf.shape(x)
        # Repeat along height dimension scale[0] times
        x = tf.concat([x] * scale[0], axis=1)
        new_height = shape[1] * scale[0]
        new_width = shape[2]
        x = tf.reshape(x, [shape[0], new_height, new_width, shape[3]])
        # Repeat along width dimension scale[1] times
        x = tf.concat([x] * scale[1], axis=2)
        new_width = shape[2] * scale[1]
        x = tf.reshape(x, [shape[0], new_height, new_width, shape[3]])
        return x

    # Wrap as a keras layer Lambda to preserve graph structure
    return keras.layers.Lambda(upsampling)

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Assuming input shape: (28, 28, 3)
        # Two Dense layers applied per spatial location across channels via TimeDistributed.
        # However, original code used Dense layers directly on 4D tensor which is unusual;
        # here we will implement Dense layers as Conv2D with kernel_size=1 to mimic Dense applied on channel dim.
        self.dense1 = keras.layers.Conv2D(filters=28, kernel_size=1, activation='relu')
        self.dense2 = keras.layers.Conv2D(filters=28, kernel_size=1, activation='relu')
        self.upsample = UpSamplingCustom2D(scale=(2, 2))
        self.dense3 = keras.layers.Conv2D(filters=3, kernel_size=1, activation='relu')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.upsample(x)
        x = self.dense3(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    # Return a random float tensor with batch size 1 and shape (28, 28, 3)
    # dtype float32 to match training inputs
    return tf.random.uniform((1, 28, 28, 3), dtype=tf.float32)

