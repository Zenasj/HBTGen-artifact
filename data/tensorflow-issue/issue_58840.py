# tf.random.uniform((B, H, W, C), dtype=tf.float32) ← Input shape is assumed as (batch_size, 180, 180, 3)

import tensorflow as tf
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self, num_classes=5):  # tf_flowers dataset has 5 classes
        super().__init__()
        # This model reflects the Keras Sequential model architecture described in the issue:
        self.conv1 = layers.Conv2D(16, 3, padding='same', activation='relu')
        self.pool1 = layers.MaxPooling2D()
        self.conv2 = layers.Conv2D(32, 3, padding='same', activation='relu')
        self.pool2 = layers.MaxPooling2D()
        self.conv3 = layers.Conv2D(64, 3, padding='same', activation='relu')
        self.pool3 = layers.MaxPooling2D()
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(128, activation='relu')
        self.out = layers.Dense(num_classes)

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.out(x)
        return x

def my_model_function():
    # Return a MyModel instance initialized for 5 classes (tf_flowers dataset)
    return MyModel(num_classes=5)

def GetInput():
    # Returns a random float tensor simulating a batch of images from tf_flowers dataset
    # Assumed batch size is 32, image size 180x180, 3 channels as per issue code
    batch_size = 32
    height = 180
    width = 180
    channels = 3
    # Random values between 0 and 1 to simulate normalized images
    return tf.random.uniform((batch_size, height, width, channels), dtype=tf.float32)

