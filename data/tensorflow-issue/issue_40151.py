# tf.random.uniform((B, 28, 28, 1), dtype=tf.float32) ← Input shape inferred as (28,28,1) grayscale images

import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # This is the LeNet style architecture from the issue example
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.conv2 = Conv2D(64, 3, activation='relu')
        self.conv3 = Conv2D(128, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(64)
        self.out = Dense(10, activation='softmax')

    def call(self, x):
        # The original code checked x shape and optionally expanded dims,
        # here we require input with batch dimension so no need to expand dims.
        # If a 3D input (H,W,C) was passed, expand dims to batchify.
        if len(x.shape) == 3:
            x = tf.expand_dims(x, axis=0)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        return self.out(x)

def my_model_function():
    # Instantiate and return the LeNet-like model
    return MyModel()

def GetInput():
    # Return a batch of random grayscale images with shape (B=1,28,28,1),
    # dtype float32 matches Conv2D expectation
    return tf.random.uniform((1, 28, 28, 1), dtype=tf.float32)

