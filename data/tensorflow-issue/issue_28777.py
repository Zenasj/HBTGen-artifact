# tf.random.uniform((B, 32, 32, 3), dtype=tf.float32) ← Input shape and dtype inferred from the usage in issue

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation
from tensorflow.keras.layers import BatchNormalization, Flatten

class MyModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        # Following the fix suggested in the issue: 
        # separate Conv2D layers are defined for each distinct call with input channel mismatch
        self.conv1 = Conv2D(32, (3, 3), padding='same')  # For input channels=3
        self.conv2 = Conv2D(32, (3, 3), padding='same')  # For input channels=32 (second conv with 32 filters)
        self.conv3 = Conv2D(64, (3, 3), padding='same')  # For input channels=32 (first conv with 64 filters)
        self.conv4 = Conv2D(64, (3, 3), padding='same')  # For input channels=64 (second conv with 64 filters)

        self.pool = MaxPooling2D(pool_size=(2, 2))
        self.bn = BatchNormalization()
        self.relu = Activation("relu")
        self.softmax = Activation("softmax")
        self.drop1 = Dropout(0.25)
        self.drop2 = Dropout(0.5)
        self.dense1 = Dense(512)
        self.dense2 = Dense(10)
        self.flat = Flatten()

    def call(self, inputs, train=False):
        # Note: The `train` argument controls behaviour of BN and Dropout layers
        z = self.conv1(inputs)           # input channels 3 -> 32 filters
        z = self.bn(z, training=train)
        z = self.relu(z)

        z = self.conv2(z)               # input channels 32 -> 32 filters
        z = self.bn(z, training=train)
        z = self.relu(z)
        z = self.pool(z)
        z = self.drop1(z, training=train)

        z = self.conv3(z)               # input channels 32 -> 64 filters
        z = self.bn(z, training=train)
        z = self.relu(z)

        z = self.conv4(z)               # input channels 64 -> 64 filters
        z = self.bn(z, training=train)
        z = self.relu(z)
        z = self.pool(z)
        z = self.drop1(z, training=train)

        z = self.flat(z)
        z = self.dense1(z)
        z = self.relu(z)
        z = self.drop2(z, training=train)
        z = self.dense2(z)
        z = self.softmax(z)

        return z

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Returns a random input tensor matching the expected input: batch size 1, 32x32 image with 3 channels
    # Use tf.random.uniform with dtype float32 as used in issue examples
    return tf.random.uniform((1, 32, 32, 3), dtype=tf.float32)

