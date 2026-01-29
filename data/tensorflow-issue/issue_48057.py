# tf.random.uniform((B, 224, 224, 3), dtype=tf.float32) ← inferred input shape and dtype

import tensorflow as tf
from tensorflow import keras

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv = keras.layers.Conv2D(1, (2, 2), strides=(1, 1), padding='same')
        # We use a Lambda layer to apply tf.nn.local_response_normalization as in original repro.
        # Note: This is the operation that causes crashes on multi-GPU evaluation in TF 2.4.x
        self.lrn = keras.layers.Lambda(tf.nn.local_response_normalization)
        self.relu = keras.layers.Activation('relu')
        self.gap = keras.layers.GlobalAveragePooling2D()
        self.dense = keras.layers.Dense(5, activation='softmax')

    def call(self, inputs, training=False):
        x = tf.cast(inputs, tf.float32)
        x = self.conv(x)
        x = self.lrn(x)
        x = self.relu(x)
        x = self.gap(x)
        x = self.dense(x)
        return x

def my_model_function():
    # Return an instance of MyModel initialized with random weights
    return MyModel()

def GetInput():
    # Return a random input tensor with shape [batch_size, 224, 224, 3]
    # Use batch_size=32 as standard default for image dataset batches
    batch_size = 32
    return tf.random.uniform((batch_size, 224, 224, 3), dtype=tf.float32)

