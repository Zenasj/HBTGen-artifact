# tf.random.uniform((10, 110, 75, 3), dtype=tf.float32) ← inferred input shape from ImageDataGenerator target_size=(110,75), color_mode='rgb', batch_size=10

import tensorflow as tf
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        c1 = 16
        c2 = 32
        h1 = 64
        activation = 'relu'

        # Following the original architecture recreated from Sequential using Functional API layers here as submodules
        self.conv1 = layers.Conv2D(c1, (3, 3), padding='same', activation=activation)
        self.conv2 = layers.Conv2D(c2, (3, 3), padding='same', activation=activation)
        self.pool = layers.MaxPooling2D((2, 2))
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(h1, activation=activation)
        # Output units is 2 in example because original train_classes not retrievable here - 
        # assuming 2 classes based on categorical_class_mode typical scenario
        self.dense2 = layers.Dense(2, activation='softmax')  # This should match number of classes in training data

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dense1(x)
        out = self.dense2(x)
        return out


def my_model_function():
    # Return an instance of MyModel
    # No custom weights to load, compile later on usage
    return MyModel()


def GetInput():
    # Return a random tensor input matching model input shape (batch_size=10, 110 height, 75 width, 3 channels)
    # Use float32 since image data are rescaled to floats via ImageDataGenerator
    return tf.random.uniform(shape=(10, 110, 75, 3), minval=0, maxval=1, dtype=tf.float32)

