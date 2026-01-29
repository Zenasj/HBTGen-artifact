# tf.random.uniform((BATCH, 32, 32, 3), dtype=tf.float32)  # inferred input shape from CIFAR-10 images

import tensorflow as tf
from tensorflow import keras as k

BATCH = 128
CLASSES = 10

class MyModel(tf.keras.Model):
    """
    A fused model encapsulating the Inception-style toy classifier from the issue.
    This model replaces the Keras functional model definition with a subclassed Keras Model.
    """

    def __init__(self, input_shape=(32, 32, 3), num_classes=10):
        super().__init__()
        # Input shape is fixed (CIFAR-10) by default: 32x32x3, num_classes=10

        # Define layers corresponding to the three towers of the inception block
        self.conv1x1_tower1 = k.layers.Conv2D(64, (1, 1), padding="same", activation="relu")
        self.conv3x3_tower1 = k.layers.Conv2D(64, (3, 3), padding="same", activation="relu")

        self.conv1x1_tower2 = k.layers.Conv2D(64, (1, 1), padding="same", activation="relu")
        self.conv5x5_tower2 = k.layers.Conv2D(64, (5, 5), padding="same", activation="relu")

        self.maxpool_tower3 = k.layers.MaxPooling2D((3, 3), strides=(1, 1), padding="same")
        self.conv1x1_tower3 = k.layers.Conv2D(64, (1, 1), padding="same", activation="relu")

        self.concat = k.layers.Concatenate(axis=3)
        self.flatten = k.layers.Flatten()
        self.classifier = k.layers.Dense(num_classes, activation="softmax")

        # Build dummy call to create weights, optional for subclassed model inference
        self.build((None,) + input_shape)

    @tf.function(jit_compile=True)
    def call(self, inputs, training=False):
        # Forward pass implementing the inception block
        x = inputs

        tower1 = self.conv1x1_tower1(x)
        tower1 = self.conv3x3_tower1(tower1)

        tower2 = self.conv1x1_tower2(x)
        tower2 = self.conv5x5_tower2(tower2)

        tower3 = self.maxpool_tower3(x)
        tower3 = self.conv1x1_tower3(tower3)

        merged = self.concat([tower1, tower2, tower3])

        flat = self.flatten(merged)
        out = self.classifier(flat)

        return out

def my_model_function():
    """
    Returns an instance of MyModel initialized to match the inception-like model.
    """
    return MyModel(input_shape=(32, 32, 3), num_classes=CLASSES)

def GetInput():
    """
    Return a random input tensor matching the CIFAR-10 data shape expected by MyModel.
    The tensor dtype is float32 and values in [0,1] (as in original data preprocessing).
    """
    # shape: (batch_size, height, width, channels)
    return tf.random.uniform((BATCH, 32, 32, 3), dtype=tf.float32)

