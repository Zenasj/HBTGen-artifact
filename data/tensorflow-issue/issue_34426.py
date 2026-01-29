# tf.random.uniform((B, 32, 32, 3), dtype=tf.float32)  # CIFAR-10 input shape and type assumed

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras import Sequential

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Backbone based on ResNet50 excluding top layer
        self.resnet = ResNet50(include_top=False, weights='imagenet', input_shape=(32, 32, 3))
        self.pool = GlobalAveragePooling2D()
        self.classifier = Dense(100, activation='softmax')

    def call(self, inputs, training=False):
        # Forward pass through ResNet base, global pooling, and final dense layer
        x = self.resnet(inputs, training=training)
        x = self.pool(x)
        return self.classifier(x)


def my_model_function():
    # Return a fresh instance of MyModel
    return MyModel()


def GetInput():
    # Return a batch of random inputs matching CIFAR-10 images (B=64 assumed)
    return tf.random.uniform((64, 32, 32, 3), dtype=tf.float32)

