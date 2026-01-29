# tf.random.uniform((BATCH_SIZE, 28, 28, 1), dtype=tf.float32) ← inferred input shape: grayscale MNIST images batch

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Define the convolutional base
        self.conv = tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1))
        self.pool = tf.keras.layers.MaxPooling2D()
        self.flatten = tf.keras.layers.Flatten()
        # Dense layers
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)


def my_model_function():
    # Returns an instance of MyModel (untrained)
    return MyModel()


def GetInput():
    # Return a random tensor input compatible with MyModel: (BATCH_SIZE, 28, 28, 1)
    # BATCH_SIZE is chosen arbitrarily as 64 matching typical batch size from the issue example
    BATCH_SIZE = 64
    return tf.random.uniform(shape=(BATCH_SIZE, 28, 28, 1), minval=0, maxval=1, dtype=tf.float32)

