# tf.random.uniform((B, 32, 32, 3), dtype=tf.float32) ← Input shape inferred from Conv2D input_shape in original code

import tensorflow as tf
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Reconstructing the model architecture from original snippet:
        # Conv2D(128, (3,3)), input_shape=(32,32,3)
        self.conv = layers.Conv2D(128, (3, 3), padding='valid')  
        self.pool = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
        self.elu = layers.Activation('elu')

        self.flatten = layers.Flatten()
        self.dense = layers.Dense(10)
        # Softmax activation with explicit float32 dtype as in original example
        self.softmax = layers.Activation('softmax', dtype='float32')

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.pool(x)
        x = self.elu(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.softmax(x)
        return x


def my_model_function():
    # Return an instance of MyModel, constructed similarly to the original Sequential model
    return MyModel()


def GetInput():
    # Return a random float32 tensor matching input_shape (batch, height, width, channels)
    # Batch size arbitrary; choosing 8 as a typical small batch size
    BATCH_SIZE = 8
    return tf.random.uniform((BATCH_SIZE, 32, 32, 3), dtype=tf.float32)

